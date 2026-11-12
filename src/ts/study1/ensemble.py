"""
Dynamic Regime-Specialized PPO Ensemble (Chapter III Sec 4, Eqs. III.23-III.24).

Manages `3 regimes x K experts` (trains up to `K_max = 7` experts per regime once per fold
so `K in {3, 5, 7}` in Table 6.13 and `T in {1, 5, 10, 20, 'static_1_over_N'}` in Table 6.12
are evaluated on a shared set of trained experts).
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ts.study1.ppo_expert import ACTION_WEIGHTS, NEUTRAL_ACTION_INDEX, RegimePPOExpert
from ts.study1.regime_labeling import REGIME_BEAR, REGIME_BULL, REGIME_SIDEWAYS


class DynamicRegimeEnsemble:
    """
    Implements Eqs. (III.23)-(III.24):
      w_{i,t} = exp(SR_{i,K} / T) / sum_{j=1}^M exp(SR_{j,K} / T)
      pi_ensemble(a | s_t) = sum_{i=1}^M w_{i,t} pi_i(a | s_t)
      a_t^* = argmax_a pi_ensemble(a | s_t)
    """

    def __init__(
        self,
        state_dim: int = 128,
        experts_per_regime: int = 5,
        rolling_sharpe_window_bars: int = 720,
        bars_per_year: int = 8760,
        base_seed: int = 1000,
        ppo_kwargs: Optional[dict] = None,
    ) -> None:
        self.state_dim = state_dim
        self.experts_per_regime = experts_per_regime
        self.rolling_sharpe_window_bars = rolling_sharpe_window_bars
        self.bars_per_year = bars_per_year
        ppo_kwargs = dict(ppo_kwargs or {})
        self.include_prev_weight = bool(ppo_kwargs.get("include_prev_weight", False))
        self.last_rollout_expert_weights_: Optional[np.ndarray] = None
        self.last_rollout_prev_weights_: Optional[np.ndarray] = None
        self.last_final_action_: int = NEUTRAL_ACTION_INDEX

        self.pools: Dict[int, List[RegimePPOExpert]] = {}
        for r_id in (REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS):
            self.pools[r_id] = [
                RegimePPOExpert(
                    regime_id=r_id,
                    seed=base_seed + r_id * 100 + k,
                    state_dim=state_dim,
                    **ppo_kwargs,
                )
                for k in range(experts_per_regime)
            ]

    def train_all_pools(
        self,
        train_states: np.ndarray,
        train_regimes: np.ndarray,
        next_bar_returns: np.ndarray,
        cost_rates: np.ndarray,
        total_rollouts: int = 8,
        log_dir: Optional[Union[str, Path]] = None,
        fold_id: Optional[int] = None,
        tag_prefix: str = "",
        sideways_turnover_penalty: Optional[float] = None,
    ) -> Dict[int, List[float]]:
        # `sideways_turnover_penalty` is C of Eq. (III.19). None keeps the expert default.
        reward_kwargs = ({} if sideways_turnover_penalty is None
                         else {"sideways_turnover_penalty": float(sideways_turnover_penalty)})
        summary: Dict[int, List[float]] = {}
        log_root = Path(log_dir) if log_dir is not None else None
        if log_root is not None:
            log_root.mkdir(parents=True, exist_ok=True)

        for r_id, experts in self.pools.items():
            mask = train_regimes == r_id
            r_trajectory: List[float] = []
            for exp in experts:
                hist = exp.train_on_fold(
                    states=train_states,
                    next_bar_returns=next_bar_returns,
                    cost_rates=cost_rates,
                    regime_mask=mask,
                    total_rollouts=total_rollouts,
                    **reward_kwargs,
                )
                r_trajectory.append(hist[-1] if hist else 0.0)
                if log_root is not None and hasattr(exp, "last_training_logs_"):
                    prefix = f"{tag_prefix}_" if tag_prefix else ""
                    f_str = f"fold{fold_id}_" if fold_id is not None else ""
                    log_file = log_root / f"{prefix}{f_str}regime{r_id}_seed{exp.seed}.json"
                    payload = {
                        "fold_id": fold_id,
                        "regime_id": int(r_id),
                        "seed": int(exp.seed),
                        "total_rollouts": int(total_rollouts),
                        "matched_regime_bars": int(np.sum(mask)),
                        "rollout_logs": exp.last_training_logs_,
                    }
                    log_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            summary[r_id] = r_trajectory
        return summary

    @staticmethod
    def compute_boltzmann_weights(
        rolling_sharpes: np.ndarray,
        temperature: Union[float, str] = 10.0,
    ) -> np.ndarray:
        """
        Eq. (III.23): Softmax over trailing annualized Sharpe ratios with temperature `T`.
        If `temperature == 'static_1_over_N'` (or `<= 0`), returns uniform `1 / M` weights.
        """
        m = len(rolling_sharpes)
        if m == 0:
            return np.array([], dtype=np.float64)
        if isinstance(temperature, str) or float(temperature) <= 0.0:
            return np.full(m, 1.0 / m, dtype=np.float64)

        t_val = float(temperature)
        sr_clipped = np.clip(np.asarray(rolling_sharpes, dtype=np.float64), -10.0, 10.0)
        scaled = sr_clipped / t_val
        scaled -= np.max(scaled)
        exp_s = np.exp(scaled)
        return exp_s / np.sum(exp_s)

    def rollout_causal_policy(
        self,
        states: np.ndarray,
        active_regimes: np.ndarray,
        bar_asset_returns: np.ndarray,
        one_way_cost_rates: np.ndarray,
        temperature: Union[float, str] = 10.0,
        num_experts_subset: Optional[int] = None,
        warmup_states: Optional[np.ndarray] = None,
        warmup_returns: Optional[np.ndarray] = None,
        warmup_costs: Optional[np.ndarray] = None,
        initial_action: int = NEUTRAL_ACTION_INDEX,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Causally evaluate the ensemble bar-by-bar on out-of-sample `states`, updating each
        expert's trailing 720-bar counterfactual Sharpe only after bar `t`'s return is realized.

        Also records per-bar Boltzmann expert weights `(N, m)` in `self.last_rollout_expert_weights_`
        and per-bar prior portfolio weights `(N,)` in `self.last_rollout_prev_weights_`.
        Returns `(ensemble_probs_n5, chosen_actions_n, target_weights_n)`.
        """
        m = min(num_experts_subset or self.experts_per_regime, self.experts_per_regime)
        sqrt_ann = float(np.sqrt(self.bars_per_year))

        hist_ret: Dict[Tuple[int, int], deque[float]] = {}
        prev_exp_a: Dict[Tuple[int, int], int] = {}
        for r_id in (REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS):
            for k in range(m):
                hist_ret[(r_id, k)] = deque(maxlen=self.rolling_sharpe_window_bars)
                prev_exp_a[(r_id, k)] = NEUTRAL_ACTION_INDEX

        if warmup_states is not None and warmup_returns is not None and len(warmup_states) > 0:
            w_tail = slice(max(0, len(warmup_states) - self.rolling_sharpe_window_bars), len(warmup_states))
            ws = warmup_states[w_tail]
            wr = warmup_returns[w_tail]
            wc = warmup_costs[w_tail] if warmup_costs is not None else np.full(len(ws), 0.0005)
            for r_id in (REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS):
                for k in range(m):
                    probs_w_all = self.pools[r_id][k].predict_probs_all_prev(ws)
                    argmax_w = np.argmax(probs_w_all, axis=-1)
                    pa = NEUTRAL_ACTION_INDEX
                    for i_w in range(len(ws)):
                        a_w = int(argmax_w[i_w, pa])
                        tw = float(ACTION_WEIGHTS[a_w])
                        pw = float(ACTION_WEIGHTS[pa])
                        net_r = tw * float(wr[i_w]) - float(wc[i_w]) * abs(tw - pw)
                        hist_ret[(r_id, k)].append(net_r)
                        pa = a_w
                    prev_exp_a[(r_id, k)] = pa

        expert_probs_all: Dict[Tuple[int, int], np.ndarray] = {}
        expert_argmax_all: Dict[Tuple[int, int], np.ndarray] = {}
        for r_id in (REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS):
            for k in range(m):
                p_all = self.pools[r_id][k].predict_probs_all_prev(states)
                expert_probs_all[(r_id, k)] = p_all
                expert_argmax_all[(r_id, k)] = np.argmax(p_all, axis=-1)

        n = len(states)
        ens_probs = np.zeros((n, 5), dtype=np.float32)
        chosen_actions = np.zeros(n, dtype=np.int64)
        target_weights = np.zeros(n, dtype=np.float64)
        exp_weights_log = np.zeros((n, m), dtype=np.float32)
        prev_port_weights = np.zeros(n, dtype=np.float32)

        cur_port_a = int(initial_action)
        for t in range(n):
            prev_port_weights[t] = float(ACTION_WEIGHTS[cur_port_a])
            r_t = int(active_regimes[t])
            srs = np.zeros(m, dtype=np.float64)
            for k in range(m):
                buf = hist_ret[(r_t, k)]
                if len(buf) >= 24:
                    arr = np.asarray(buf, dtype=np.float64)
                    sd = float(np.std(arr, ddof=1))
                    srs[k] = (float(np.mean(arr)) / sd) * sqrt_ann if sd > 1e-10 else 0.0
                else:
                    srs[k] = 0.0

            w_k = self.compute_boltzmann_weights(srs, temperature=temperature)
            exp_weights_log[t] = w_k.astype(np.float32)
            p_bar = np.zeros(5, dtype=np.float64)
            for k in range(m):
                p_bar += w_k[k] * expert_probs_all[(r_t, k)][t, cur_port_a]
            p_bar /= max(np.sum(p_bar), 1e-12)

            ens_probs[t] = p_bar.astype(np.float32)
            a_star = int(np.argmax(p_bar))
            chosen_actions[t] = a_star
            target_weights[t] = ACTION_WEIGHTS[a_star]
            cur_port_a = a_star

            mkt_r = float(bar_asset_returns[t])
            c_rate = float(one_way_cost_rates[t])
            for r_id in (REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS):
                for k in range(m):
                    pa_k = prev_exp_a[(r_id, k)]
                    a_k = int(expert_argmax_all[(r_id, k)][t, pa_k])
                    tw_k = float(ACTION_WEIGHTS[a_k])
                    pw_k = float(ACTION_WEIGHTS[pa_k])
                    net_rk = tw_k * mkt_r - c_rate * abs(tw_k - pw_k)
                    hist_ret[(r_id, k)].append(net_rk)
                    prev_exp_a[(r_id, k)] = a_k

        self.last_rollout_expert_weights_ = exp_weights_log
        self.last_rollout_prev_weights_ = prev_port_weights
        self.last_final_action_ = int(cur_port_a)
        return ens_probs, chosen_actions, target_weights
