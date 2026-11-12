"""
Group-level Shapley attribution for Study 1 multimodal representations (Chapter VI Sec 4.2, Eq. VI.12).

Computes closed-form 3-player Shapley values (`2^3 = 8` coalitions) over the three modality groups:
  - Visual (`128-d` ResNet-18 + CBAM embedding)
  - Technical (`64-d` indicator embedding)
  - Sentiment (`32-d` news EWMA embedding)

For each evaluated bar `t`, the attribution target is the ensemble-weighted logit of the selected
action `a_t^*` within bar `t`'s active causal regime pool `r_t`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from ts.study1.ensemble import DynamicRegimeEnsemble
from ts.study1.regime_labeling import REGIME_BEAR, REGIME_BULL, REGIME_SIDEWAYS


@dataclass(frozen=True)
class ModalityAttributionResult:
    visual_pct: float
    tech_pct: float
    sentiment_pct: float
    tech_plus_sentiment_pct: float
    mean_abs_visual: float
    mean_abs_tech: float
    mean_abs_sentiment: float
    n_background: int
    n_explained: int

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


# Precomputed Shapley weights |S|!(3 - |S| - 1)! / 3! for 3 players
_SHAPLEY_WEIGHT_BY_SIZE = {
    0: math.factorial(0) * math.factorial(2) / math.factorial(3),  # 1/3
    1: math.factorial(1) * math.factorial(1) / math.factorial(3),  # 1/6
    2: math.factorial(2) * math.factorial(0) / math.factorial(3),  # 1/3
}


@torch.no_grad()
def compute_modality_group_shapley(
    concat_modalities: np.ndarray,
    fusion_module: nn.Module,
    ensemble: DynamicRegimeEnsemble,
    active_regimes: np.ndarray,
    chosen_actions: np.ndarray,
    expert_weights: Optional[np.ndarray] = None,
    num_experts_subset: Optional[int] = None,
    d_visual: int = 128,
    d_tech: int = 64,
    d_senti: int = 32,
    n_background: int = 256,
    n_explain: int = 512,
    eval_chunk_size: int = 8192,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> ModalityAttributionResult:
    """
    Compute closed-form 3-group Shapley shares (`visual_pct`, `tech_pct`, `sentiment_pct`) targeting
    each bar's active regime pool ensemble logit for its chosen action `chosen_actions[t]`.
    """
    n_total = len(concat_modalities)
    expected_dim = d_visual + d_tech + d_senti
    if n_total == 0:
        raise ValueError("concat_modalities must be non-empty.")
    if concat_modalities.shape[1] != expected_dim:
        raise ValueError(f"Expected modality width {expected_dim}, got {concat_modalities.shape[1]}.")
    if len(active_regimes) != n_total or len(chosen_actions) != n_total:
        raise ValueError("active_regimes and chosen_actions must match len(concat_modalities).")

    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fusion_module = fusion_module.to(dev)
    fusion_module.eval()

    m_exp = min(num_experts_subset or ensemble.experts_per_regime, ensemble.experts_per_regime)
    if expert_weights is not None:
        if expert_weights.shape != (n_total, m_exp):
            raise ValueError(f"Expected expert_weights shape {(n_total, m_exp)}, got {expert_weights.shape}.")
        exp_w_all = np.asarray(expert_weights, dtype=np.float32)
    else:
        exp_w_all = np.full((n_total, m_exp), 1.0 / max(m_exp, 1), dtype=np.float32)

    rng = np.random.default_rng(seed)
    bg_cnt = min(n_background, n_total)
    exp_cnt = min(n_explain, n_total)
    bg_idx = rng.choice(n_total, size=bg_cnt, replace=False)
    exp_idx = rng.choice(n_total, size=exp_cnt, replace=False)

    bg_x = torch.as_tensor(concat_modalities[bg_idx], dtype=torch.float32, device=dev)
    fg_x = torch.as_tensor(concat_modalities[exp_idx], dtype=torch.float32, device=dev)
    fg_regimes = np.asarray(active_regimes[exp_idx], dtype=np.int64)
    fg_actions = torch.as_tensor(chosen_actions[exp_idx], dtype=torch.long, device=dev)
    fg_exp_w = torch.as_tensor(exp_w_all[exp_idx], dtype=torch.float32, device=dev)

    s0 = slice(0, d_visual)
    s1 = slice(d_visual, d_visual + d_tech)
    s2 = slice(d_visual + d_tech, expected_dim)

    bg_vis, bg_tech, bg_senti = bg_x[:, s0], bg_x[:, s1], bg_x[:, s2]
    fg_vis, fg_tech, fg_senti = fg_x[:, s0], fg_x[:, s1], fg_x[:, s2]

    for r_id in (REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS):
        for k in range(m_exp):
            ensemble.pools[r_id][k].actor.to(dev)
            ensemble.pools[r_id][k].actor.eval()

    def eval_coalition(mask_bits: int) -> torch.Tensor:
        """
        Evaluate v_t(S) = E_{b ~ bg}[ f_t(z(S, x_t, b)) ] for all `exp_cnt` foreground bars.
        `mask_bits` bit 0 = visual, bit 1 = tech, bit 2 = sentiment.
        Returns tensor of shape `(exp_cnt,)` on CPU float64.
        """
        use_vis = bool(mask_bits & 1)
        use_tech = bool(mask_bits & 2)
        use_senti = bool(mask_bits & 4)

        out_vals = torch.zeros(exp_cnt, dtype=torch.float64, device=dev)

        # Pair count = exp_cnt * bg_cnt; process in chunks of foreground bars
        bars_per_chunk = max(1, eval_chunk_size // max(bg_cnt, 1))
        for c_start in range(0, exp_cnt, bars_per_chunk):
            c_end = min(exp_cnt, c_start + bars_per_chunk)
            c_len = c_end - c_start

            if use_vis:
                v_in = fg_vis[c_start:c_end].unsqueeze(1).expand(c_len, bg_cnt, d_visual)
            else:
                v_in = bg_vis.unsqueeze(0).expand(c_len, bg_cnt, d_visual)

            if use_tech:
                t_in = fg_tech[c_start:c_end].unsqueeze(1).expand(c_len, bg_cnt, d_tech)
            else:
                t_in = bg_tech.unsqueeze(0).expand(c_len, bg_cnt, d_tech)

            if use_senti:
                s_in = fg_senti[c_start:c_end].unsqueeze(1).expand(c_len, bg_cnt, d_senti)
            else:
                s_in = bg_senti.unsqueeze(0).expand(c_len, bg_cnt, d_senti)

            flat_n = c_len * bg_cnt
            fused = fusion_module(
                v_in.reshape(flat_n, d_visual),
                t_in.reshape(flat_n, d_tech),
                s_in.reshape(flat_n, d_senti),
            ).reshape(c_len, bg_cnt, -1)

            chunk_reg = fg_regimes[c_start:c_end]
            chunk_act = fg_actions[c_start:c_end]
            chunk_w = fg_exp_w[c_start:c_end]

            chunk_target = torch.zeros((c_len, bg_cnt), dtype=torch.float32, device=dev)
            for r_id in (REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS):
                r_loc = np.where(chunk_reg == r_id)[0]
                if len(r_loc) == 0:
                    continue
                r_loc_t = torch.as_tensor(r_loc, dtype=torch.long, device=dev)
                sub_states = fused[r_loc_t].reshape(len(r_loc) * bg_cnt, -1)
                act_idx = chunk_act[r_loc_t].unsqueeze(1).expand(len(r_loc), bg_cnt).reshape(-1, 1)

                ens_logit = torch.zeros(len(r_loc), bg_cnt, dtype=torch.float32, device=dev)
                for k in range(m_exp):
                    logits_k = ensemble.pools[r_id][k].actor(sub_states)
                    chosen_logit_k = logits_k.gather(1, act_idx).reshape(len(r_loc), bg_cnt)
                    w_k = chunk_w[r_loc_t, k].unsqueeze(1)
                    ens_logit = ens_logit + w_k * chosen_logit_k
                chunk_target[r_loc_t] = ens_logit

            out_vals[c_start:c_end] = chunk_target.to(dtype=torch.float64).mean(dim=1)

        return out_vals.cpu()

    coalition_values: Dict[int, np.ndarray] = {}
    for mask_bits in range(8):
        coalition_values[mask_bits] = eval_coalition(mask_bits).numpy()

    phi = np.zeros((exp_cnt, 3), dtype=np.float64)
    for player in range(3):
        p_bit = 1 << player
        for s_mask in range(8):
            if s_mask & p_bit:
                continue
            s_size = int(bin(s_mask).count("1"))
            weight = _SHAPLEY_WEIGHT_BY_SIZE[s_size]
            marginal = coalition_values[s_mask | p_bit] - coalition_values[s_mask]
            phi[:, player] += weight * marginal

    abs_means = np.mean(np.abs(phi), axis=0)
    denom = max(float(np.sum(abs_means)), 1e-12)
    vis_pct = float(abs_means[0] / denom * 100.0)
    tech_pct = float(abs_means[1] / denom * 100.0)
    senti_pct = float(abs_means[2] / denom * 100.0)

    return ModalityAttributionResult(
        visual_pct=vis_pct,
        tech_pct=tech_pct,
        sentiment_pct=senti_pct,
        tech_plus_sentiment_pct=tech_pct + senti_pct,
        mean_abs_visual=float(abs_means[0]),
        mean_abs_tech=float(abs_means[1]),
        mean_abs_sentiment=float(abs_means[2]),
        n_background=bg_cnt,
        n_explained=exp_cnt,
    )
