"""
Teacher Signal Conversion & Temporal Alignment (Ch. V Sec 2 & Sec 4.4, Eqs. V.2–V.4).

1. Soft Target Policy Distribution (`P_teacher^τ`, Eqs. V.2–V.3):
     `z_i^(T) = -(w_t^* - w(a_i))^2 / (2 * σ_w^2)`,   `σ_w = 0.25`
     `P_teacher^τ = softmax(z^(T) / τ)`,               `τ = 2.0`
2. Reasoning Latent Vector (`E_reasoning ∈ R^768`, Eq. V.4):
     `FinBERT_CLS(C_t)` from `ProsusAI/finbert`
3. Temporal Broadcast Alignment (Sec 4.4):
     Copies each hourly bar `t`'s Teacher signal across `10` sampled `(40, 100)` LOB windows
     strictly inside bar `t`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from ts.study1.ppo_expert import ACTION_WEIGHTS


@dataclass
class TeacherSignalPack:
    lob_windows: np.ndarray        # (N_pairs, 40, 100) float32
    teacher_logits: np.ndarray     # (N_pairs, 5) float32 — raw z^(T) before /tau!
    teacher_probs: np.ndarray      # (N_pairs, 5) float32 — softmax(z^(T) / tau)
    teacher_reasoning: np.ndarray  # (N_pairs, 768) float32
    bar_returns: np.ndarray        # (N_pairs,) float32 — next-bar asset return for L_task
    cost_rates: np.ndarray         # (N_pairs,) float32 — bar friction rate for L_task
    bar_indices: np.ndarray        # (N_pairs,) int64 — originating 1h bar index


def compute_teacher_soft_logits_and_probs(
    w_final: np.ndarray | float,
    sigma_w: float = 0.25,
    tau: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact Eqs. (V.2)–(V.3).
    Returns `(z_logits, probs_tau)` of shapes `(..., 5)`.
    Note: `z_logits` is NOT divided by `tau` so `ReasoningDistillationLoss` divides by `tau` once!
    """
    w_arr = np.asarray(w_final, dtype=np.float64)[..., None]  # (..., 1)
    grid = ACTION_WEIGHTS[None, :] if w_arr.ndim == 2 else ACTION_WEIGHTS
    denom = 2.0 * (sigma_w ** 2)  # 2 * 0.25^2 = 0.125
    z_logits = -((w_arr - grid) ** 2) / denom

    scaled = z_logits / tau
    scaled = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp_s = np.exp(scaled)
    probs = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
    return z_logits.astype(np.float32), probs.astype(np.float32)


class FinBERTReasoningEncoder:
    """
    Extracts `768-d` `[CLS]` embeddings from Meta PM consensus text `C_t` (Eq. V.4).

    In offline unit tests where HuggingFace model weights are intentionally skipped
    (`offline_hash_mode=True`), produces a deterministic SHA-512 projected 768-d vector
    (with both positive and negative coordinates, matching FinBERT `[CLS]` geometry).
    In real runs (`offline_hash_mode=False`), loads `ProsusAI/finbert` onto GPU/CPU.
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        max_length: int = 256,
        offline_hash_mode: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.offline_hash_mode = offline_hash_mode
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is None and not self.offline_hash_mode:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self._model.eval()

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        n = len(texts)
        out = np.empty((n, 768), dtype=np.float32)
        if n == 0:
            return out

        if self.offline_hash_mode:
            for i, txt in enumerate(texts):
                seed_int = int.from_bytes(
                    hashlib.sha256((txt or "").encode("utf-8")).digest()[:8], "little"
                )
                rng = np.random.default_rng(seed_int)
                vec = rng.standard_normal(768).astype(np.float32)
                out[i] = vec / max(float(np.linalg.norm(vec)), 1e-6)
            return out

        self._ensure_loaded()
        assert self._tokenizer is not None and self._model is not None

        for s in range(0, n, batch_size):
            batch_txt = [str(t) for t in texts[s : s + batch_size]]
            enc = self._tokenizer(
                batch_txt,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            hidden = self._model(**enc).last_hidden_state[:, 0, :]  # [CLS] (B, 768)
            out[s : s + batch_size] = hidden.cpu().numpy().astype(np.float32)
        return out


def build_aligned_distillation_dataset(
    per_bar_lob_windows: Sequence[Sequence[np.ndarray]],
    w_final_per_bar: np.ndarray,
    consensus_texts_per_bar: Sequence[str],
    next_bar_returns: np.ndarray,
    one_way_cost_rates: np.ndarray,
    finbert_encoder: FinBERTReasoningEncoder,
    sigma_w: float = 0.25,
    tau: float = 2.0,
) -> TeacherSignalPack:
    """
    Implements Chapter V Sec 4.4 Step 3:
    Copies each 1-hour bar `t`'s `(P_teacher, E_reasoning)` across all `M` (default 10)
    sampled `(40, 100)` LOB windows within bar `t`.
    """
    n_bars = len(w_final_per_bar)
    bar_z, bar_p = compute_teacher_soft_logits_and_probs(w_final_per_bar, sigma_w=sigma_w, tau=tau)
    bar_e = finbert_encoder.encode_texts(consensus_texts_per_bar)

    lob_list: List[np.ndarray] = []
    z_list: List[np.ndarray] = []
    p_list: List[np.ndarray] = []
    e_list: List[np.ndarray] = []
    r_list: List[float] = []
    c_list: List[float] = []
    idx_list: List[int] = []

    for t in range(n_bars):
        wins = per_bar_lob_windows[t]
        for w_arr in wins:
            lob_list.append(np.asarray(w_arr, dtype=np.float32))
            z_list.append(bar_z[t])
            p_list.append(bar_p[t])
            e_list.append(bar_e[t])
            r_list.append(float(next_bar_returns[t]))
            c_list.append(float(one_way_cost_rates[t]))
            idx_list.append(t)

    return TeacherSignalPack(
        lob_windows=np.stack(lob_list, axis=0),
        teacher_logits=np.stack(z_list, axis=0),
        teacher_probs=np.stack(p_list, axis=0),
        teacher_reasoning=np.stack(e_list, axis=0),
        bar_returns=np.asarray(r_list, dtype=np.float32),
        cost_rates=np.asarray(c_list, dtype=np.float32),
        bar_indices=np.asarray(idx_list, dtype=np.int64),
    )
