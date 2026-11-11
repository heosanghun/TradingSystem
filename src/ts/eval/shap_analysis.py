"""
KernelSHAP Modality Attribution Analysis (Chapter VI Sec 8.1, Eq. VI.12, Table 6.9 & Table 6.15).

Computes exact Shapley attributions `φ_j` via `shap.KernelExplainer` on Study 1's
Cross-Modal Fusion + Active Regime Ensemble policy logits during Period II (`2022.01~2022.12`)
over the 3 input modalities `[f_visual (128), f_tech (64), f_senti (32)]`:

  `ρ_m = ( Σ_{j ∈ m} E[|φ_j|] ) / ( Σ_k E[|φ_k|] ) × 100 (%)`   (Eq. VI.12)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, Optional

import numpy as np
import shap
import torch

from ts.study1.encoders import CrossModalStateFusion
from ts.study1.ensemble import DynamicRegimeEnsemble
from ts.study1.regime_labeling import REGIME_BEAR


@dataclass
class ModalitySHAPResult:
    visual_pct: float
    tech_pct: float
    sentiment_pct: float
    tech_plus_sentiment_pct: float
    n_background: int
    n_explained: int

    def to_dict(self) -> Dict[str, float | int]:
        return asdict(self)


def compute_modality_kernel_shap(
    concat_modalities: np.ndarray,
    fusion_module: CrossModalStateFusion,
    ensemble: DynamicRegimeEnsemble,
    target_regime: int = REGIME_BEAR,
    d_visual: int = 128,
    d_tech: int = 64,
    d_senti: int = 32,
    n_background: int = 32,
    n_explain: int = 64,
    nsamples: int = 128,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> ModalitySHAPResult:
    """
    Compute grouped KernelSHAP / feature-level KernelSHAP modality shares matching Eq. (VI.12).
    """
    dev = device or next(fusion_module.parameters()).device
    fusion_module.eval()

    total_dim = d_visual + d_tech + d_senti
    if concat_modalities.shape[1] != total_dim:
        raise ValueError(f"Expected modality vector dim {total_dim}, got {concat_modalities.shape[1]}")

    rng = np.random.default_rng(seed)
    n_avail = len(concat_modalities)
    bg_idx = rng.choice(n_avail, size=min(n_background, n_avail), replace=False)
    ex_idx = rng.choice(n_avail, size=min(n_explain, n_avail), replace=False)

    bg_data = concat_modalities[bg_idx].astype(np.float64)
    ex_data = concat_modalities[ex_idx].astype(np.float64)

    def _model_expected_weight(x_224: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            xt = torch.as_tensor(x_224, dtype=torch.float32, device=dev)
            f_v = xt[:, :d_visual]
            f_t = xt[:, d_visual : d_visual + d_tech]
            f_s = xt[:, d_visual + d_tech :]
            s_t = fusion_module(f_v, f_t, f_s).cpu().numpy()

        # Ensemble action logits / expected directional score across the active regime pool
        experts = ensemble.pools[target_regime]
        logits_sum = np.zeros((len(s_t), 5), dtype=np.float64)
        with torch.no_grad():
            st_dev = torch.as_tensor(s_t, dtype=torch.float32, device=experts[0].device)
            for exp in experts:
                exp.actor.eval()
                logits_sum += exp.actor(st_dev).cpu().numpy()
        mean_logits = logits_sum / max(len(experts), 1)
        # Max action logit magnitude captures decision sensitivity across the 5 actions
        return np.max(mean_logits, axis=-1)

    explainer: Callable = shap.KernelExplainer(_model_expected_weight, bg_data, silent=True)
    shap_vals = np.asarray(explainer.shap_values(ex_data, nsamples=nsamples, silent=True), dtype=np.float64)

    mean_abs_phi = np.mean(np.abs(shap_vals), axis=0)
    vis_sum = float(np.sum(mean_abs_phi[:d_visual]))
    tech_sum = float(np.sum(mean_abs_phi[d_visual : d_visual + d_tech]))
    senti_sum = float(np.sum(mean_abs_phi[d_visual + d_tech :]))
    denom = max(vis_sum + tech_sum + senti_sum, 1e-12)

    vis_pct = (vis_sum / denom) * 100.0
    tech_pct = (tech_sum / denom) * 100.0
    senti_pct = (senti_sum / denom) * 100.0

    return ModalitySHAPResult(
        visual_pct=vis_pct,
        tech_pct=tech_pct,
        sentiment_pct=senti_pct,
        tech_plus_sentiment_pct=tech_pct + senti_pct,
        n_background=len(bg_data),
        n_explained=len(ex_data),
    )
