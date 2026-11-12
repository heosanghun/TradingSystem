"""
Controlled Baseline Models (Chapter VI Sec 4, Table 6.5, Eqs. VI.9–VI.11).

All baselines strictly use the same 5 discrete target weights
`[-1.0, -0.5, 0.0, +0.5, +1.0]`, the same causal test windows, and the
same Phase 1 / Phase 2 transaction cost model:
1. `BuyAndHoldBaseline` — Constant `w_t = +1.0`
2. `SinglePPOBaseline` — Schulman et al. [18] single-seed PPO on price/tech state
3. `StaticEnsembleBaseline` — Liu & Kang [49] fixed `1/N` equal-weighted PPO ensemble (Eq. VI.9)
4. `MAMLMetaPPOBaseline` — Finn et al. [61] / Noor & Fatima [50] First-Order MAML / Reptile
   regime-task meta-policy adapted across Bull, Bear, and Sideways tasks (Eqs. VI.10–VI.11)
"""

from __future__ import annotations

import copy
from typing import List, Optional, Sequence

import numpy as np
import torch

from ts.study1.ppo_expert import ACTION_WEIGHTS, RegimePPOExpert
from ts.study1.regime_labeling import REGIME_BEAR, REGIME_BULL, REGIME_SIDEWAYS


class BuyAndHoldBaseline:
    """Passive benchmark holding `+1.0` BTC exposure throughout the evaluation window."""

    def predict_weights(self, num_bars: int) -> np.ndarray:
        return np.ones(num_bars, dtype=np.float64)


class SinglePPOBaseline:
    """Single-policy PPO baseline trained on price/volume technical features only (Table 6.5)."""

    def __init__(
        self,
        state_dim: int,
        seed: int = 42,
        hidden_dims: Sequence[int] = (128, 64),
        device: Optional[torch.device] = None,
    ) -> None:
        self.expert = RegimePPOExpert(
            regime_id=REGIME_BULL,
            seed=seed,
            state_dim=state_dim,
            hidden_dims=hidden_dims,
            device=device,
        )

    def fit(
        self, states: np.ndarray, next_bar_returns: np.ndarray, cost_rates: np.ndarray, rollouts: int = 8
    ) -> "SinglePPOBaseline":
        self.expert.train_on_fold(
            states=states,
            next_bar_returns=next_bar_returns,
            cost_rates=cost_rates,
            regime_mask=None,
            total_rollouts=rollouts,
        )
        return self

    def predict_weights(self, states: np.ndarray) -> np.ndarray:
        probs = self.expert.predict_probs(states)
        actions = np.argmax(probs, axis=-1)
        return ACTION_WEIGHTS[actions]


class StaticEnsembleBaseline:
    """
    Static `1/N` Equal-Weighted PPO Ensemble (Liu & Kang [49], Eq. VI.9):
      `π_static(a | s_t) = (1 / N) * Σ_{i=1}^N π_i(a | s_t)`
    """

    def __init__(
        self,
        state_dim: int,
        n_agents: int = 5,
        base_seed: int = 200,
        hidden_dims: Sequence[int] = (128, 64),
        device: Optional[torch.device] = None,
    ) -> None:
        self.agents: List[RegimePPOExpert] = [
            RegimePPOExpert(
                regime_id=REGIME_BULL,
                seed=base_seed + i,
                state_dim=state_dim,
                hidden_dims=hidden_dims,
                device=device,
            )
            for i in range(n_agents)
        ]

    def fit(
        self, states: np.ndarray, next_bar_returns: np.ndarray, cost_rates: np.ndarray, rollouts: int = 8
    ) -> "StaticEnsembleBaseline":
        for ag in self.agents:
            ag.train_on_fold(
                states=states,
                next_bar_returns=next_bar_returns,
                cost_rates=cost_rates,
                regime_mask=None,
                total_rollouts=rollouts,
            )
        return self

    def predict_weights(self, states: np.ndarray) -> np.ndarray:
        prob_sum = np.zeros((len(states), 5), dtype=np.float64)
        for ag in self.agents:
            prob_sum += ag.predict_probs(states)
        avg_probs = prob_sum / len(self.agents)
        actions = np.argmax(avg_probs, axis=-1)
        return ACTION_WEIGHTS[actions]


class MAMLMetaPPOBaseline:
    """Reptile-style parameter interpolation with inner PPO adaptation.

    The historical class name is retained for API compatibility. This is not
    MAML: the outer update interpolates task-adapted parameters instead of
    differentiating a held-out task loss. The manuscript/model label requires
    an explicit decision before this can be reported as a MAML comparison.
    """

    def __init__(
        self,
        state_dim: int,
        seed: int = 777,
        inner_lr: float = 3e-4,
        outer_step: float = 0.3,
        hidden_dims: Sequence[int] = (128, 64),
        device: Optional[torch.device] = None,
    ) -> None:
        self.state_dim = state_dim
        self.seed = seed
        self.inner_lr = inner_lr
        self.outer_step = outer_step
        self.hidden_dims = hidden_dims
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.meta_agent = RegimePPOExpert(
            regime_id=REGIME_BULL,
            seed=seed,
            state_dim=state_dim,
            hidden_dims=hidden_dims,
            lr_actor=inner_lr,
            device=self.device,
        )

    def fit(
        self,
        states: np.ndarray,
        regime_labels: np.ndarray,
        next_bar_returns: np.ndarray,
        cost_rates: np.ndarray,
        meta_iterations: int = 6,
        inner_rollouts: int = 2,
    ) -> "MAMLMetaPPOBaseline":
        for _ in range(meta_iterations):
            base_actor_sd = copy.deepcopy(self.meta_agent.actor.state_dict())
            base_critic_sd = copy.deepcopy(self.meta_agent.critic.state_dict())

            adapted_actor_sds = []
            adapted_critic_sds = []

            for r_id in (REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS):
                task_agent = RegimePPOExpert(
                    regime_id=r_id,
                    seed=self.seed + r_id,
                    state_dim=self.state_dim,
                    hidden_dims=self.hidden_dims,
                    lr_actor=self.inner_lr,
                    device=self.device,
                )
                task_agent.actor.load_state_dict(base_actor_sd)
                task_agent.critic.load_state_dict(base_critic_sd)

                mask = regime_labels == r_id
                task_agent.train_on_fold(
                    states=states,
                    next_bar_returns=next_bar_returns,
                    cost_rates=cost_rates,
                    regime_mask=mask,
                    total_rollouts=inner_rollouts,
                )
                adapted_actor_sds.append(task_agent.actor.state_dict())
                adapted_critic_sds.append(task_agent.critic.state_dict())

            # Outer meta-update across the 3 regime tasks (Eq. VI.11)
            new_actor_sd = {}
            for k, v in base_actor_sd.items():
                if v.is_floating_point():
                    mean_task = torch.stack([sd[k] for sd in adapted_actor_sds], dim=0).mean(dim=0)
                    new_actor_sd[k] = v + self.outer_step * (mean_task - v)
                else:
                    new_actor_sd[k] = v
            self.meta_agent.actor.load_state_dict(new_actor_sd)

            new_critic_sd = {}
            for k, v in base_critic_sd.items():
                if v.is_floating_point():
                    mean_task = torch.stack([sd[k] for sd in adapted_critic_sds], dim=0).mean(dim=0)
                    new_critic_sd[k] = v + self.outer_step * (mean_task - v)
                else:
                    new_critic_sd[k] = v
            self.meta_agent.critic.load_state_dict(new_critic_sd)

        return self

    def predict_weights_causal(
        self,
        test_states: np.ndarray,
        test_returns: np.ndarray,
        test_costs: np.ndarray,
        adapt_interval_bars: int = 168,
        adapt_window_bars: int = 168,
    ) -> np.ndarray:
        """
        Evaluate out-of-sample with strictly causal K-shot inner adaptation on trailing
        realized bars `[t - adapt_window .. t - 1]` (never accessing bar `t`'s return!).
        """
        n = len(test_states)
        weights = np.zeros(n, dtype=np.float64)

        live_agent = RegimePPOExpert(
            regime_id=REGIME_BULL,
            seed=self.seed,
            state_dim=self.state_dim,
            hidden_dims=self.hidden_dims,
            lr_actor=self.inner_lr,
            device=self.device,
        )
        live_agent.actor.load_state_dict(copy.deepcopy(self.meta_agent.actor.state_dict()))
        live_agent.critic.load_state_dict(copy.deepcopy(self.meta_agent.critic.state_dict()))

        for start in range(0, n, adapt_interval_bars):
            end = min(n, start + adapt_interval_bars)
            if start >= 32:
                # Adapt from meta initialization on strictly past bars `[max(0, start-W) : start]`
                live_agent.actor.load_state_dict(copy.deepcopy(self.meta_agent.actor.state_dict()))
                live_agent.critic.load_state_dict(copy.deepcopy(self.meta_agent.critic.state_dict()))
                w_s = max(0, start - adapt_window_bars)
                live_agent.train_on_fold(
                    states=test_states[w_s:start],
                    next_bar_returns=test_returns[w_s:start],
                    cost_rates=test_costs[w_s:start],
                    regime_mask=None,
                    total_rollouts=1,
                )

            probs = live_agent.predict_probs(test_states[start:end])
            acts = np.argmax(probs, axis=-1)
            weights[start:end] = ACTION_WEIGHTS[acts]

        return weights
