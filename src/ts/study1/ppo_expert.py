"""
Regime-Specialized PPO+PER Expert (Ch. III Sec 3.1–3.3, Table A.1).

Implements:
- 5 Discrete Target-Weight Actions: `w(a) ∈ {-1.0, -0.5, 0.0, +0.5, +1.0}`
- Separate Actor (`lr=3e-4` + CosineAnnealingLR) and Critic (`lr=1e-3`) networks
- Regime-specific reward shaping (Eq. III.17–III.19):
  - Bull (`0`): `ln(V_t / V_{t-1})`
  - Bear (`1`): 24-bar rolling Sortino (`ε_dd = 1e-4`) − `λ_cost * Cost_t` (`λ_cost = 1.0`)
  - Sideways (`2`): `(V_t - V_{t-1})/V_{t-1} - C * |w_t - w_{t-1}|`
- On-policy rollout Segment-Tree PER (`α=0.6`, `β: 0.4 -> 1.0`)
"""

from __future__ import annotations

from collections import deque
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ts.study1.per_buffer import SegmentTreePERBuffer
from ts.study1.regime_labeling import REGIME_BEAR, REGIME_BULL, REGIME_SIDEWAYS


ACTION_WEIGHTS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float64)


def compute_regime_rewards(
    regime_id: int,
    bar_asset_returns: np.ndarray,
    chosen_weights: np.ndarray,
    one_way_cost_rates: np.ndarray,
    bear_window: int = 24,
    bear_dd_floor: float = 1e-4,
    bear_cost_penalty: float = 1.0,
    sideways_turnover_penalty: float = 0.002,
    hourly_rf: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized computation of exact step portfolio returns & regime-specific rewards
    (Eqs. III.17–III.19) for a rollout trajectory.
    """
    n = len(bar_asset_returns)
    net_returns = np.zeros(n, dtype=np.float64)
    rewards = np.zeros(n, dtype=np.float64)
    prev_w = 0.0
    ret_win: deque[float] = deque(maxlen=bear_window)

    for t in range(n):
        w_t = float(chosen_weights[t])
        turnover = abs(w_t - prev_w)
        cost_t = float(one_way_cost_rates[t]) * turnover
        # Position w_t set at bar t earns next-step return r_t minus switching cost
        r_port = w_t * float(bar_asset_returns[t]) - cost_t
        net_returns[t] = r_port
        ret_win.append(r_port - hourly_rf)

        if regime_id == REGIME_BULL:
            # Eq. III.17: ln(V_t / V_{t-1}) = ln(1 + r_port)
            rewards[t] = float(np.log(max(1.0 + r_port, 1e-6)))
        elif regime_id == REGIME_BEAR:
            # Eq. III.18: E_{24}[R_p - R_f] / max(sqrt(E_{24}[min(0, R_p - R_f)^2]), ε_dd) - λ_cost * Cost_t
            arr = np.asarray(ret_win, dtype=np.float64)
            downside = np.minimum(0.0, arr)
            dd = float(np.sqrt(np.mean(downside * downside)))
            sortino_step = float(np.mean(arr)) / max(dd, bear_dd_floor)
            # Clip extreme dimensionless ratio for numerical stability while preserving Eq. III.18
            rewards[t] = float(np.clip(sortino_step, -10.0, 10.0)) - bear_cost_penalty * cost_t
        else:
            # Eq. III.19: Sideways return minus explicit turnover penalty C * |w_t - w_{t-1}|
            rewards[t] = r_port - sideways_turnover_penalty * turnover

        prev_w = w_t

    return rewards.astype(np.float32), net_returns.astype(np.float32)


class _MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Sequence[int], out_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        cur = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(cur, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.LayerNorm(h))
            cur = h
        layers.append(nn.Linear(cur, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RegimePPOExpert:
    """Single regime-specialized PPO+PER expert seeded deterministically."""

    def __init__(
        self,
        regime_id: int,
        seed: int,
        state_dim: int = 128,
        action_dim: int = 5,
        hidden_dims: Sequence[int] = (128, 64),
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.20,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_end: float = 1.0,
        minibatch_size: int = 1024,
        ppo_epochs: int = 4,
        device: Optional[torch.device] = None,
    ) -> None:
        self.regime_id = regime_id
        self.seed = seed
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.per_alpha = per_alpha
        self.per_beta_start = per_beta_start
        self.per_beta_end = per_beta_end
        self.minibatch_size = minibatch_size
        self.ppo_epochs = ppo_epochs
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        self.actor = _MLPHead(state_dim, hidden_dims, action_dim).to(self.device)
        self.critic = _MLPHead(state_dim, hidden_dims, 1).to(self.device)
        self.opt_actor = torch.optim.AdamW(self.actor.parameters(), lr=lr_actor)
        self.opt_critic = torch.optim.AdamW(self.critic.parameters(), lr=lr_critic)

    @torch.no_grad()
    def predict_probs(self, states_x: np.ndarray) -> np.ndarray:
        self.actor.eval()
        x = torch.as_tensor(states_x, dtype=torch.float32, device=self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        logits = self.actor(x)
        probs = F.softmax(logits, dim=-1)
        return probs.cpu().numpy()

    @torch.no_grad()
    def predict_logits(self, states_x: np.ndarray) -> np.ndarray:
        self.actor.eval()
        x = torch.as_tensor(states_x, dtype=torch.float32, device=self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.actor(x).cpu().numpy()

    def train_on_fold(
        self,
        states: np.ndarray,
        next_bar_returns: np.ndarray,
        cost_rates: np.ndarray,
        regime_mask: Optional[np.ndarray] = None,
        total_rollouts: int = 8,
        bear_window: int = 24,
        bear_dd_floor: float = 1e-4,
        bear_cost_penalty: float = 1.0,
        sideways_turnover_penalty: float = 0.002,
    ) -> list[float]:
        """
        Train this expert on bars matching its regime inside the training fold
        (falls back to full training fold if fewer than 32 bars match its regime).
        """
        if regime_mask is not None and int(np.sum(regime_mask)) >= 32:
            idx = np.where(regime_mask)[0]
            s_arr = states[idx]
            r_arr = next_bar_returns[idx]
            c_arr = cost_rates[idx]
        else:
            s_arr = states
            r_arr = next_bar_returns
            c_arr = cost_rates

        n = len(s_arr)
        if n < 4:
            return []

        sched_actor = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_actor, T_max=max(1, total_rollouts)
        )
        mean_rewards: list[float] = []

        s_tensor = torch.as_tensor(s_arr, dtype=torch.float32, device=self.device)

        for r_step in range(total_rollouts):
            beta = self.per_beta_start + (self.per_beta_end - self.per_beta_start) * (
                r_step / max(1, total_rollouts - 1)
            )
            self.actor.eval()
            self.critic.eval()
            with torch.no_grad():
                logits = self.actor(s_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample()
                old_log_probs = dist.log_prob(actions)
                values = self.critic(s_tensor).squeeze(-1)

            act_np = actions.cpu().numpy()
            weights_np = ACTION_WEIGHTS[act_np]
            rewards_np, _ = compute_regime_rewards(
                regime_id=self.regime_id,
                bar_asset_returns=r_arr,
                chosen_weights=weights_np,
                one_way_cost_rates=c_arr,
                bear_window=bear_window,
                bear_dd_floor=bear_dd_floor,
                bear_cost_penalty=bear_cost_penalty,
                sideways_turnover_penalty=sideways_turnover_penalty,
            )
            mean_rewards.append(float(np.mean(rewards_np)))

            val_np = values.cpu().numpy()
            deltas = np.zeros(n, dtype=np.float32)
            advantages = np.zeros(n, dtype=np.float32)
            gae = 0.0
            for t in reversed(range(n)):
                next_v = val_np[t + 1] if t + 1 < n else 0.0
                delta = rewards_np[t] + self.gamma * next_v - val_np[t]
                deltas[t] = delta
                gae = delta + self.gamma * self.gae_lambda * gae
                advantages[t] = gae

            returns_np = advantages + val_np
            adv_std = float(np.std(advantages))
            norm_adv = (advantages - float(np.mean(advantages))) / max(adv_std, 1e-6)

            # Build fresh On-Policy Segment-Tree PER for this rollout buffer ONLY
            per = SegmentTreePERBuffer(alpha=self.per_alpha)
            per.build_from_td_errors(deltas)

            adv_t = torch.as_tensor(norm_adv, dtype=torch.float32, device=self.device)
            ret_t = torch.as_tensor(returns_np, dtype=torch.float32, device=self.device)

            self.actor.train()
            self.critic.train()
            batches_per_epoch = max(1, (n + self.minibatch_size - 1) // self.minibatch_size)

            for _ in range(self.ppo_epochs):
                for _ in range(batches_per_epoch):
                    b_idx, is_w = per.sample(self.minibatch_size, beta=beta, rng=self.rng)
                    b_idx_t = torch.as_tensor(b_idx, dtype=torch.long, device=self.device)
                    w_t = torch.as_tensor(is_w, dtype=torch.float32, device=self.device)

                    cur_logits = self.actor(s_tensor[b_idx_t])
                    cur_dist = torch.distributions.Categorical(logits=cur_logits)
                    new_lp = cur_dist.log_prob(actions[b_idx_t])
                    entropy = cur_dist.entropy()

                    ratio = torch.exp(new_lp - old_log_probs[b_idx_t])
                    surr1 = ratio * adv_t[b_idx_t]
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_t[b_idx_t]
                    actor_loss = -(w_t * torch.min(surr1, surr2)).mean() - self.entropy_coef * entropy.mean()

                    pred_v = self.critic(s_tensor[b_idx_t]).squeeze(-1)
                    td_new = (ret_t[b_idx_t] - pred_v).detach().cpu().numpy()
                    critic_loss = self.value_coef * (w_t * (pred_v - ret_t[b_idx_t]).pow(2)).mean()

                    self.opt_actor.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                    self.opt_actor.step()

                    self.opt_critic.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                    self.opt_critic.step()

                    per.update_priorities(b_idx, td_new)

            sched_actor.step()

        return mean_rewards
