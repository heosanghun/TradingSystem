"""
Regime-Specialized PPO+PER Expert (Chapter III Sec 3.1-3.3, Table A.1).

Implements:
- 5 discrete target-weight actions: `w(a) in {-1.0, -0.5, 0.0, +0.5, +1.0}`
- Separate Actor (`lr=3e-4` with CosineAnnealingLR) and Critic (`lr=1e-3`) networks
- Regime-specific reward functions (Eqs. III.17-III.19):
  - Bull (`0`): `ln(V_t / V_{t-1})`
  - Bear (`1`): 24-bar rolling Sortino (`eps_dd = 1e-4`) minus `lambda_cost * Cost_t` (`lambda_cost = 1.0`)
  - Sideways (`2`): `(V_t - V_{t-1})/V_{t-1} - C * |w_t - w_{t-1}|`
- On-policy rollout Segment-Tree PER (`alpha=0.6`, `beta: 0.4 -> 1.0`)
- Per-rollout convergence telemetry (`mean_reward`, `policy_entropy`, `mean_turnover`, `value_loss`,
  `actor_loss`, and `greedy_action_dist`).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import lfilter

from ts.study1.per_buffer import SegmentTreePERBuffer
from ts.study1.regime_labeling import REGIME_BEAR, REGIME_BULL


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
    Vectorized computation of step portfolio returns and regime-specific rewards
    (Eqs. III.17-III.19) for a rollout trajectory.
    """
    n = len(bar_asset_returns)
    if n == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    w = np.asarray(chosen_weights, dtype=np.float64)
    r_mkt = np.asarray(bar_asset_returns, dtype=np.float64)
    c_rate = np.asarray(one_way_cost_rates, dtype=np.float64)

    prev_w = np.empty(n, dtype=np.float64)
    prev_w[0] = 0.0
    if n > 1:
        prev_w[1:] = w[:-1]

    turnover = np.abs(w - prev_w)
    cost = c_rate * turnover
    net_returns = w * r_mkt - cost

    if regime_id == REGIME_BULL:
        rewards = np.log(np.maximum(1.0 + net_returns, 1e-6))
    elif regime_id == REGIME_BEAR:
        excess = net_returns - hourly_rf
        downside_sq = np.minimum(0.0, excess) ** 2
        csum_ex = np.concatenate(([0.0], np.cumsum(excess)))
        csum_dn = np.concatenate(([0.0], np.cumsum(downside_sq)))
        idx_end = np.arange(1, n + 1)
        idx_start = np.maximum(0, idx_end - int(bear_window))
        cnt = (idx_end - idx_start).astype(np.float64)
        mean_ex = (csum_ex[idx_end] - csum_ex[idx_start]) / cnt
        dd = np.sqrt(np.maximum((csum_dn[idx_end] - csum_dn[idx_start]) / cnt, 0.0))
        sortino_step = mean_ex / np.maximum(dd, bear_dd_floor)
        rewards = np.clip(sortino_step, -10.0, 10.0) - bear_cost_penalty * cost
    else:
        rewards = net_returns - sideways_turnover_penalty * turnover

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
        self.lr_actor = lr_actor
        self.opt_actor = torch.optim.AdamW(self.actor.parameters(), lr=lr_actor)
        self.opt_critic = torch.optim.AdamW(self.critic.parameters(), lr=lr_critic)
        self.last_training_logs_: List[Dict[str, object]] = []

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
        Populates `self.last_training_logs_` with per-rollout convergence diagnostics.
        """
        self.last_training_logs_ = []
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

        for pg in self.opt_actor.param_groups:
            pg["lr"] = self.lr_actor
            pg.pop("initial_lr", None)
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
                rollout_entropy = float(dist.entropy().mean().item())
                actions = dist.sample()
                greedy_actions = torch.argmax(logits, dim=-1)
                old_log_probs = dist.log_prob(actions)
                values = self.critic(s_tensor).squeeze(-1)

            act_np = actions.cpu().numpy()
            greedy_np = greedy_actions.cpu().numpy()
            weights_np = ACTION_WEIGHTS[act_np]
            greedy_weights_np = ACTION_WEIGHTS[greedy_np]

            sampled_turnover = float(np.mean(np.abs(np.diff(weights_np, prepend=0.0))))
            greedy_turnover = float(np.mean(np.abs(np.diff(greedy_weights_np, prepend=0.0))))

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
            m_rew = float(np.mean(rewards_np))
            mean_rewards.append(m_rew)

            val_np = values.cpu().numpy()
            deltas = np.empty(n, dtype=np.float32)
            deltas[:-1] = rewards_np[:-1] + self.gamma * val_np[1:] - val_np[:-1]
            deltas[-1] = rewards_np[-1] - val_np[-1]

            gl = float(self.gamma * self.gae_lambda)
            advantages = lfilter(
                [1.0], [1.0, -gl], deltas[::-1].astype(np.float64)
            )[::-1].astype(np.float32)

            returns_np = advantages + val_np
            adv_std = float(np.std(advantages))
            norm_adv = (advantages - float(np.mean(advantages))) / max(adv_std, 1e-6)

            per = SegmentTreePERBuffer(alpha=self.per_alpha)
            per.build_from_td_errors(deltas)

            adv_t = torch.as_tensor(norm_adv, dtype=torch.float32, device=self.device)
            ret_t = torch.as_tensor(returns_np, dtype=torch.float32, device=self.device)

            self.actor.train()
            self.critic.train()
            batches_per_epoch = max(1, (n + self.minibatch_size - 1) // self.minibatch_size)

            sum_v_loss = 0.0
            sum_a_loss = 0.0
            n_updates = 0

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
                    actor_loss = -(w_t * (torch.min(surr1, surr2) + self.entropy_coef * entropy)).mean()

                    pred_v = self.critic(s_tensor[b_idx_t]).squeeze(-1)
                    td_new = (ret_t[b_idx_t] - pred_v).detach().cpu().numpy()
                    raw_v_mse = (w_t * (pred_v - ret_t[b_idx_t]).pow(2)).mean()
                    critic_loss = self.value_coef * raw_v_mse

                    self.opt_actor.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                    self.opt_actor.step()

                    self.opt_critic.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                    self.opt_critic.step()

                    per.update_priorities(b_idx, td_new)

                    sum_v_loss += float(raw_v_mse.item())
                    sum_a_loss += float(actor_loss.item())
                    n_updates += 1

            sched_actor.step()

            greedy_counts = np.bincount(greedy_np, minlength=self.action_dim)
            self.last_training_logs_.append(
                {
                    "rollout": int(r_step + 1),
                    "mean_reward": round(m_rew, 6),
                    "policy_entropy": round(rollout_entropy, 5),
                    "mean_turnover": round(sampled_turnover, 5),
                    "greedy_mean_turnover": round(greedy_turnover, 5),
                    "value_loss": round(sum_v_loss / max(n_updates, 1), 6),
                    "actor_loss": round(sum_a_loss / max(n_updates, 1), 6),
                    "greedy_action_dist": [round(float(c) / n, 4) for c in greedy_counts],
                }
            )

        return mean_rewards
