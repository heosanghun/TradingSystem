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
from pathlib import Path
from typing import Any, Dict, Optional

import requests


THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
JSON_OBJ_RE = re.compile(r"\{.*\}", flags=re.DOTALL)


def extract_json_object(raw_text: str) -> Dict[str, Any]:
    cleaned = THINK_BLOCK_RE.sub("", raw_text).strip()
    # Try direct parse first
    try:
        return json.loads(cleaned)
    except Exception:
        pass
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
    ) -> None:
        self.backend = backend
        self.model_name = model_name
        self.endpoint_url = endpoint_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_sec = timeout_sec
        self.is_simulated_llm = backend == "deterministic_econometric"

        self.cache_db_path = Path(cache_db_path) if cache_db_path else None
        if self.cache_db_path is not None:
            self.cache_db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()

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
    ) -> Dict[str, Any]:
        key_src = f"{self.backend}|{self.model_name}|{role}|{retry_attempt}|{system_prompt}|{user_prompt}"
        cache_key = hashlib.sha256(key_src.encode("utf-8")).hexdigest()

        if retry_attempt == 0:
            cached = self._lookup_cache(cache_key)
            if cached is not None:
                return cached

        if self.backend == "openai_compat":
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
            resp = requests.post(
                self.endpoint_url,
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "format": "json",
                    "stream": False,
                    "options": {
                        "temperature": self.temperature + 0.05 * retry_attempt,
                        "num_predict": self.max_tokens,
                    },
                },
                timeout=self.timeout_sec,
            )
            resp.raise_for_status()
            text = resp.json()["message"]["content"]
            parsed = extract_json_object(text)
        elif self.backend == "deterministic_econometric":
            parsed = self._deterministic_econometric_response(role, context_hint or {})
        else:
            raise ValueError(f"Unsupported LLM backend: {self.backend}")

        if retry_attempt == 0:
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
