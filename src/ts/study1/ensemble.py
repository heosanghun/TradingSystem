"""
Dynamic Regime-Specialized PPO Ensemble (Ch. III Sec 4, Eqs. III.23–III.24).

Manages `3 regimes × K experts` (trains up to `K_max = 7` experts per regime once
so `K ∈ {3, 5, 7}` in Table 6.13 and `T ∈ {1, 5, 10, 20, 'static_1_over_N'}` in
Table 6.12 can all be evaluated deterministically without post-hoc tampering!).
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ts.study1.ppo_expert import ACTION_WEIGHTS, RegimePPOExpert
from ts.study1.regime_labeling import REGIME_BEAR, REGIME_BULL, REGIME_SIDEWAYS


class DynamicRegimeEnsemble:
    """
    Implements Eqs. (III.23)–(III.24):
      w_{i,t} = exp(SR_{i,K} / T) / Σ_{j=1}^M exp(SR_{j,K} / T)
      π_ensemble(a | s_t) = Σ_{i=1}^M w_{i,t} π_i(a | s_t)
      a_t^* = argmax_a π_ensemble(a | s_t)
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
        ppo_kwargs = ppo_kwargs or {}

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
    ) -> Dict[int, List[float]]:
        summary: Dict[int, List[float]] = {}
        for r_id, experts in self.pools.items():
            mask = train_regimes == r_id
            r_Trajectory: List[float] = []
            for exp in experts:
                hist = exp.train_on_fold(
                    states=train_states,
                    next_bar_returns=next_bar_returns,
                    cost_rates=cost_rates,
                    regime_mask=mask,
                    total_rollouts=total_rollouts,
                )
                r_Trajectory.append(hist[-1] if hist else 0.0)
            summary[r_id] = r_Trajectory
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

        T = float(temperature)
        sr_clipped = np.clip(np.asarray(rolling_sharpes, dtype=np.float64), -10.0, 10.0)
        scaled = sr_clipped / T
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Causally evaluate the ensemble bar-by-bar on out-of-sample `states`, updating each
        expert's trailing 720-bar counterfactual Sharpe ONLY AFTER bar `t`'s return is realized!

        Supports:
        - `temperature`: `1.0, 5.0, 10.0, 20.0, 'static_1_over_N'` (Table 6.12 & Variant B)
        - `num_experts_subset`: `1` (Variant C), `3, 5, 7` (Table 6.13)

        Returns `(ensemble_probs_n5, chosen_actions_n, target_weights_n)`.
        """
        m = min(num_experts_subset or self.experts_per_regime, self.experts_per_regime)
        sqrt_ann = float(np.sqrt(self.bars_per_year))

        # Precompute each expert's action probabilities across warmup + test for speed
        hist_ret: Dict[Tuple[int, int], deque[float]] = {}
        prev_exp_w: Dict[Tuple[int, int], float] = {}
        for r_id in (REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS):
            for k in range(m):
                hist_ret[(r_id, k)] = deque(maxlen=self.rolling_sharpe_window_bars)
                prev_exp_w[(r_id, k)] = 0.0

        if warmup_states is not None and warmup_returns is not None and len(warmup_states) > 0:
            w_tail = slice(max(0, len(warmup_states) - self.rolling_sharpe_window_bars), len(warmup_states))
            ws = warmup_states[w_tail]
            wr = warmup_returns[w_tail]
            wc = warmup_costs[w_tail] if warmup_costs is not None else np.full(len(ws), 0.0005)
            for r_id in (REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS):
                for k in range(m):
                    probs_w = self.pools[r_id][k].predict_probs(ws)
                    acts_w = np.argmax(probs_w, axis=-1)
                    pw = 0.0
                    for i_w in range(len(ws)):
                        tw = float(ACTION_WEIGHTS[acts_w[i_w]])
                        net_r = tw * float(wr[i_w]) - float(wc[i_w]) * abs(tw - pw)
                        hist_ret[(r_id, k)].append(net_r)
                        pw = tw
                    prev_exp_w[(r_id, k)] = pw

        # Batch-predict probabilities on test `states` for all experts
        expert_probs: Dict[Tuple[int, int], np.ndarray] = {}
        for r_id in (REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS):
            for k in range(m):
                expert_probs[(r_id, k)] = self.pools[r_id][k].predict_probs(states)

        n = len(states)
        ens_probs = np.zeros((n, 5), dtype=np.float32)
        chosen_actions = np.zeros(n, dtype=np.int64)
        target_weights = np.zeros(n, dtype=np.float64)

        for t in range(n):
            r_t = int(active_regimes[t])
            # 1. Compute trailing Sharpe strictly BEFORE seeing `bar_asset_returns[t]`
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
            p_bar = np.zeros(5, dtype=np.float64)
            for k in range(m):
                p_bar += w_k[k] * expert_probs[(r_t, k)][t]
            p_bar /= max(np.sum(p_bar), 1e-12)

            ens_probs[t] = p_bar.astype(np.float32)
            a_star = int(np.argmax(p_bar))
            chosen_actions[t] = a_star
            target_weights[t] = ACTION_WEIGHTS[a_star]

            # 2. After bar `t` executes, update counterfactual returns for all pools
            mkt_r = float(bar_asset_returns[t])
            c_rate = float(one_way_cost_rates[t])
            for r_id in (REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS):
                for k in range(m):
                    a_k = int(np.argmax(expert_probs[(r_id, k)][t]))
                    tw_k = float(ACTION_WEIGHTS[a_k])
                    pw_k = prev_exp_w[(r_id, k)]
                    net_rk = tw_k * mkt_r - c_rate * abs(tw_k - pw_k)
                    hist_ret[(r_id, k)].append(net_rk)
                    prev_exp_w[(r_id, k)] = tw_k

        return ens_probs, chosen_actions, target_weights
