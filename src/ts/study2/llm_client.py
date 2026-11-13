"""
Unified LLM Client with Persistent SQLite Cache (`DeepSeek-R1-Distill-Qwen-32B`).

Supports:
1. `"openai_compat"`: Local vLLM / SGLang / OpenAI-compatible HTTP endpoint (`http://localhost:8000/v1/chat/completions`)
2. `"ollama"`: Local Ollama server (`http://localhost:11434/api/chat`)
3. `"deterministic_econometric"`: Explicit econometric rule-grounded JSON generator strictly for
   offline unit/integration testing when no GPU LLM server is running (clearly tagged `is_simulated_llm=True`).

Strips `<think>...</think>` reasoning blocks emitted by DeepSeek-R1 before extracting JSON,
and persists every response in SQLite keyed by `(role, model, sha256(system + user), retry_idx)`.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import logging
import time
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

import subprocess

import requests


# DeepSeek-R1 always opens its answer with a reasoning block and Ollama's `think=False`
# does not switch it off for this model. Sending this text as the start of the assistant
# turn closes the block immediately, so the token budget goes to the JSON answer.
EMPTY_THINK_PREFILL = "<think>" + chr(10) * 2 + "</think>" + chr(10) * 2

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
JSON_OBJ_RE = re.compile(r"\{.*\}", flags=re.DOTALL)
LOGGER = logging.getLogger(__name__)


class TruncatedResponseError(ValueError):
    """The answer reached the output token limit before the JSON object was closed."""


# Added to the user prompt only after an answer was cut off. The role prompt and the token
# limit stay as they are; the model is told why its answer was unusable and asked for a
# shorter one in the same schema.
LENGTH_NOTICE = (
    "\n[Length limit] Your previous answer was cut off at the output token limit. "
    "Answer again with the same JSON schema and keep the rationale within two sentences."
)


class LLMCallError(RuntimeError):
    """Transport or response parsing failed after the configured retries."""


def validate_role_payload(role, payload):
    """Reject structurally incomplete role answers instead of neutral defaults."""
    if not isinstance(payload, dict):
        raise ValueError("The model answer is not a JSON object")
    if role == "edge_verifier":
        if str(payload.get("decision", "")).upper() not in ("ACCEPT", "REJECT"):
            raise ValueError("Missing/invalid edge decision")
        score = float(payload["confidence_score"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid edge confidence")
        rationale = payload.get("economic_rationale")
    elif role in ("alpha_analyst", "risk_manager"):
        weight_key = "Proposed Portfolio Weights" if role == "alpha_analyst" else "Revised Portfolio Weights"
        weights = payload[weight_key]
        if not isinstance(weights, dict) or not {"Asset", "Cash"}.issubset(weights):
            raise ValueError("Missing portfolio weights")
        if not all(math.isfinite(float(v)) for v in weights.values()):
            raise ValueError("Portfolio weights must be finite")
        if role == "risk_manager" and payload.get("Rebuttal Decision") not in ("AGREE", "MODIFY", "STRONG REJECT"):
            raise ValueError("Invalid risk rebuttal decision")
        rationale = payload.get("Grounded Rationale" if role == "alpha_analyst" else "Risk Rationale")
    else:
        return
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("Role rationale is missing")


def extract_json_object(raw_text: str) -> Dict[str, Any]:
    if raw_text.lower().count("<think>") != raw_text.lower().count("</think>"):
        raise ValueError("Truncated reasoning block; no complete final JSON answer")
    cleaned = THINK_BLOCK_RE.sub("", raw_text).strip()
    # Try direct parse first
    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            raise ValueError("LLM response must be a JSON object")
        return parsed
    except json.JSONDecodeError:
        LOGGER.debug("Trying JSON object extraction from surrounding response text")
    match = JSON_OBJ_RE.search(cleaned)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"No valid JSON object found in LLM output: {raw_text[:200]}")


class LLMClient:
    """Cached JSON LLM interface for Study 1 sentiment & Study 2 Verifier/Debate."""

    def __init__(
        self,
        backend: str = "openai_compat",
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        endpoint_url: str = "http://localhost:8000/v1/chat/completions",
        temperature: float = 0.2,
        max_tokens: int = 256,
        cache_db_path: Optional[Path] = None,
        timeout_sec: int = 60,
        max_request_attempts: int = 3,
        retry_backoff_sec: float = 1.0,
        ollama_num_ctx: int = 4096,
        skip_thinking: bool = True,
        gpu_temp_limit_c: Optional[int] = None,
    ) -> None:
        self.backend = backend
        self.model_name = model_name
        self.endpoint_url = endpoint_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_sec = timeout_sec
        if max_request_attempts < 1 or retry_backoff_sec < 0:
            raise ValueError("Retry attempts must be positive and backoff nonnegative")
        self.max_request_attempts = max_request_attempts
        self.retry_backoff_sec = retry_backoff_sec
        # Ollama's default context (32k for this model) does not fit in 24 GB next to the
        # weights and spills layers to the CPU. The longest project prompt is under 2k tokens.
        self.ollama_num_ctx = int(ollama_num_ctx)
        self.skip_thinking = bool(skip_thinking)
        # Optional hardware protection for long runs on a laptop: before a request, wait while
        # the GPU is at or above this temperature. It only delays requests; answers are unaffected.
        self.gpu_temp_limit_c = gpu_temp_limit_c
        self.telemetry = Counter()
        self.use_cache = True
        self.is_simulated_llm = backend == "deterministic_econometric"

        self.cache_db_path = Path(cache_db_path) if cache_db_path else None
        if self.cache_db_path is not None:
            self.cache_db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()

    def _wait_for_gpu_to_cool(self) -> None:
        if self.gpu_temp_limit_c is None:
            return
        waited = 0.0
        while waited < 300.0:
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10).stdout
                temp = int(out.strip().splitlines()[0])
            except Exception:
                return  # no reading available: do not block the run
            # Resume a few degrees below the limit so the run does not oscillate at the limit.
            if temp < self.gpu_temp_limit_c - (4 if waited > 0 else 0):
                break
            time.sleep(5.0)
            waited += 5.0
        if waited:
            self.telemetry["gpu_cooldown_ms"] += int(waited * 1000)

    def _init_db(self) -> None:
        assert self.cache_db_path is not None
        with sqlite3.connect(str(self.cache_db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_cache (
                    cache_key TEXT PRIMARY KEY,
                    role TEXT,
                    model TEXT,
                    response_json TEXT
                )
                """
            )
            conn.commit()

    def _lookup_cache(self, key: str) -> Optional[Dict[str, Any]]:
        if self.cache_db_path is None:
            return None
        with sqlite3.connect(str(self.cache_db_path)) as conn:
            row = conn.execute(
                "SELECT response_json FROM llm_cache WHERE cache_key = ?", (key,)
            ).fetchone()
            if row:
                return json.loads(row[0])
        return None

    def _store_cache(self, key: str, role: str, payload: Dict[str, Any]) -> None:
        if self.cache_db_path is None:
            return
        with sqlite3.connect(str(self.cache_db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO llm_cache (cache_key, role, model, response_json) VALUES (?, ?, ?, ?)",
                (key, role, self.model_name, json.dumps(payload, ensure_ascii=False)),
            )
            conn.commit()

    def generate_json(
        self,
        role: str,
        system_prompt: str,
        user_prompt: str,
        retry_attempt: int = 0,
        context_hint: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Retry transient transport and malformed JSON failures; never cache failures.

        ``use_cache=False`` bypasses both reads and writes, including during a
        teacher latency benchmark. Permanent failures are explicit exceptions.
        """
        self.telemetry["logical_calls"] += 1
        was_truncated = False
        for attempt in range(self.max_request_attempts):
            try:
                parsed = self._generate_json_once(
                    role, system_prompt, user_prompt + (LENGTH_NOTICE if was_truncated else ""),
                    retry_attempt, context_hint, use_cache and self.use_cache)
                if was_truncated:
                    self.telemetry["answers_after_length_notice"] += 1
                validate_role_payload(role, parsed)
                return parsed
            except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
                self.telemetry["failed_attempts"] += 1
                LOGGER.warning("LLM call failed: role=%s attempt=%d/%d error=%s: %s",
                               role, attempt + 1, self.max_request_attempts, type(exc).__name__,
                               str(exc)[:160])
                self.telemetry["failed_" + type(exc).__name__] += 1
                was_truncated = was_truncated or isinstance(exc, TruncatedResponseError)
                if attempt + 1 == self.max_request_attempts:
                    self.telemetry["failed_calls"] += 1
                    raise LLMCallError(f"{role}: {type(exc).__name__} after {attempt+1} attempts") from exc
                self.telemetry["retries"] += 1
                # Only transport errors need a growing pause; a malformed or cut-off answer
                # is simply sampled again.
                if isinstance(exc, requests.RequestException):
                    time.sleep(min(30., self.retry_backoff_sec * 2 ** attempt))
        raise AssertionError("Unreachable retry state")

    def _generate_json_once(
        self,
        role: str,
        system_prompt: str,
        user_prompt: str,
        retry_attempt: int = 0,
        context_hint: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        key_src = json.dumps([self.backend, self.endpoint_url, self.model_name, role,
                              retry_attempt, self.temperature, self.max_tokens,
                              system_prompt, user_prompt, context_hint if self.is_simulated_llm else None,
                              [self.ollama_num_ctx, self.skip_thinking] if self.backend == "ollama" else None],
                             sort_keys=True)
        cache_key = hashlib.sha256(key_src.encode("utf-8")).hexdigest()

        if use_cache and retry_attempt == 0:
            cached = self._lookup_cache(cache_key)
            if cached is not None:
                try:
                    validate_role_payload(role, cached)
                except (ValueError, KeyError, TypeError):
                    LOGGER.warning("Ignoring malformed cached role answer: %s", role)
                    self.telemetry["invalid_cache_entries"] += 1
                    with sqlite3.connect(str(self.cache_db_path)) as conn:
                        conn.execute("DELETE FROM llm_cache WHERE cache_key=?", (cache_key,))
                else:
                    self.telemetry["cache_hits"] += 1
                    return cached

        if self.backend == "openai_compat":
            self.telemetry["http_calls"] += 1
            resp = requests.post(
                self.endpoint_url,
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": self.temperature + 0.05 * retry_attempt,
                    "max_tokens": self.max_tokens,
                    "response_format": {"type": "json_object"},
                },
                timeout=self.timeout_sec,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            parsed = extract_json_object(text)
        elif self.backend == "ollama":
            self.telemetry["http_calls"] += 1
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            if self.skip_thinking:
                messages.append({"role": "assistant", "content": EMPTY_THINK_PREFILL})
            self._wait_for_gpu_to_cool()
            t_call = time.perf_counter()
            resp = requests.post(
                self.endpoint_url,
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False,
                    "think": False,
                    "options": {
                        "temperature": self.temperature + 0.05 * retry_attempt,
                        "num_predict": self.max_tokens,
                        "num_ctx": self.ollama_num_ctx,
                    },
                },
                timeout=self.timeout_sec,
            )
            resp.raise_for_status()
            body = resp.json()
            self.telemetry["http_ms"] += int((time.perf_counter() - t_call) * 1000.0)
            self.telemetry["output_tokens"] += int(body.get("eval_count", 0) or 0)
            self.telemetry["prompt_tokens"] += int(body.get("prompt_eval_count", 0) or 0)
            if int(body.get("prompt_eval_count", 0) or 0) + self.max_tokens > self.ollama_num_ctx:
                # The server silently drops the start of a prompt that does not fit.
                self.telemetry["context_overflows"] += 1
                raise ValueError("Prompt plus answer budget does not fit in the context window")
            if body.get("done_reason") == "length":
                # The answer hit the token limit. A cut-off JSON object is not an answer.
                self.telemetry["truncated_responses"] += 1
                raise TruncatedResponseError("Response truncated at the output token limit")
            text = body["message"]["content"]
            parsed = extract_json_object(text)
        elif self.backend == "deterministic_econometric":
            parsed = self._deterministic_econometric_response(role, context_hint or {})
        else:
            raise ValueError(f"Unsupported LLM backend: {self.backend}")

        validate_role_payload(role, parsed)
        if use_cache and retry_attempt == 0:
            self._store_cache(cache_key, role, parsed)
        return parsed

    @staticmethod
    def _deterministic_econometric_response(role: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deterministic, citation-faithful econometric rule generator used ONLY in automated
        tests (`backend='deterministic_econometric'`).
        """
        if role == "edge_verifier":
            te = float(ctx.get("te_score", 0.0))
            p_gc = float(ctx.get("granger_p", 1.0))
            accept = (te >= 0.02) or (p_gc < 0.05)
            conf = float(min(0.95, max(0.35, 0.5 * te + (0.4 if p_gc < 0.05 else 0.1))))
            return {
                "decision": "ACCEPT" if accept else "REJECT",
                "confidence_score": round(conf, 4),
                "economic_rationale": (
                    f"Cross-asset transmission from {ctx.get('source', 'X')} to {ctx.get('target', 'Y')} "
                    f"is supported by statistical information flow. Liquidity spillover channel is plausible."
                ),
            }

        if role == "alpha_analyst":
            pos_edges = ctx.get("pos_edges", [])
            ens_w = float(ctx.get("study1_target_weight", 0.0))
            asset_w = float(min(1.0, max(0.0, 0.5 + 0.5 * ens_w)))
            if pos_edges:
                e0 = pos_edges[0]
                cite = f"[Edge ID: #{e0['edge_id']}, TE: {e0['te_norm']:.4f}]"
            else:
                cite = "No active positive edge available."
            return {
                "Proposed Portfolio Weights": {"Asset": round(asset_w, 4), "Cash": round(1.0 - asset_w, 4)},
                "Grounded Rationale": f"{cite} indicates constructive momentum transmission aligned with ensemble baseline.",
            }

        if role == "risk_manager":
            neg_edges = ctx.get("neg_edges", [])
            ens_w = float(ctx.get("study1_target_weight", 0.0))
            if neg_edges:
                e0 = neg_edges[0]
                max_w = float(e0.get("weight", 0.0))
                cite = f"[Edge ID: #{e0['edge_id']}, TE: {e0['te_norm']:.4f}]"
                if max_w > 0.65:
                    dec = "STRONG REJECT"
                    short_w = min(1.0, round(max_w, 4))
                    asset_w = 0.0
                elif max_w > 0.35:
                    dec = "MODIFY"
                    short_w = round(0.5 * max_w, 4)
                    asset_w = 0.0
                else:
                    dec = "AGREE"
                    short_w = 0.0
                    asset_w = max(0.0, round(ens_w, 4))
            else:
                dec = "AGREE"
                short_w = 0.0
                asset_w = max(0.0, round(ens_w, 4))
                cite = "No active tail-risk edge triggered."
            cash_w = max(0.0, round(1.0 - asset_w - short_w, 4))
            return {
                "Rebuttal Decision": dec,
                "Revised Portfolio Weights": {"Asset": asset_w, "Cash": cash_w, "Short": short_w},
                "Risk Rationale": f"{cite} governs downside contagion assessment and drawdown protection.",
            }

        raise ValueError(f"Unknown deterministic role: {role}")
