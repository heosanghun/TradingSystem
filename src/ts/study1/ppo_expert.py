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
NEUTRAL_ACTION_INDEX = 2


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
    segment_starts: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized computation of step portfolio returns and regime-specific rewards
    (Eqs. III.17-III.19) for a rollout trajectory, optionally resetting position
    and rolling Bear Sortino windows at non-contiguous `segment_starts`.
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
    if segment_starts is not None:
        seg_mask = np.asarray(segment_starts, dtype=bool)
        prev_w[seg_mask] = 0.0
    else:
        seg_mask = None

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
        if seg_mask is not None:
            seg_first = np.maximum.accumulate(np.where(seg_mask, np.arange(n), 0))
            idx_start = np.maximum(seg_first, idx_start)
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
        self.expected_in_dim = int(in_dim)
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
        if x.shape[-1] == self.expected_in_dim - 1:
            x = F.pad(x, (0, 1), value=0.0)
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
        include_prev_weight: bool = False,
        respect_segment_boundaries: Optional[bool] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.regime_id = regime_id
        self.seed = seed
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.include_prev_weight = bool(include_prev_weight)
        self.respect_segment_boundaries = (
            bool(include_prev_weight)
            if respect_segment_boundaries is None
            else bool(respect_segment_boundaries)
        )
        self.obs_dim = state_dim + 1 if self.include_prev_weight else state_dim
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

        self.actor = _MLPHead(self.obs_dim, hidden_dims, action_dim).to(self.device)
        self.critic = _MLPHead(self.obs_dim, hidden_dims, 1).to(self.device)
        self.lr_actor = lr_actor
        self.opt_actor = torch.optim.AdamW(self.actor.parameters(), lr=lr_actor)
        self.opt_critic = torch.optim.AdamW(self.critic.parameters(), lr=lr_critic)
        self.last_training_logs_: List[Dict[str, object]] = []

    def _prepare_obs_tensor(
        self, states_x: np.ndarray, prev_weights: Optional[np.ndarray] = None
    ) -> torch.Tensor:
        x = torch.as_tensor(states_x, dtype=torch.float32, device=self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if not self.include_prev_weight:
            return x
        if prev_weights is None:
            pw = torch.zeros((x.shape[0], 1), dtype=torch.float32, device=self.device)
        else:
            pw = torch.as_tensor(prev_weights, dtype=torch.float32, device=self.device).reshape(-1, 1)
        return torch.cat([x, pw], dim=-1)

    @torch.no_grad()
    def predict_probs(
        self, states_x: np.ndarray, prev_weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        self.actor.eval()
        obs = self._prepare_obs_tensor(states_x, prev_weights=prev_weights)
        logits = self.actor(obs)
        probs = F.softmax(logits, dim=-1)
        return probs.cpu().numpy()

    @torch.no_grad()
    def predict_logits(
        self, states_x: np.ndarray, prev_weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        self.actor.eval()
        obs = self._prepare_obs_tensor(states_x, prev_weights=prev_weights)
        return self.actor(obs).cpu().numpy()

    @torch.no_grad()
    def predict_probs_all_prev(self, states_x: np.ndarray) -> np.ndarray:
        """
        Return action probability table of shape `(N, 5, 5)` where axis 1 indexes the
        previous portfolio action `prev_a in {0..4}` and axis 2 indexes `a_t in {0..4}`.
        """
        n = len(states_x)
        if n == 0:
            return np.zeros((0, self.action_dim, self.action_dim), dtype=np.float32)
        if not self.include_prev_weight:
            p = self.predict_probs(states_x).astype(np.float32)
            return np.broadcast_to(p[:, None, :], (n, self.action_dim, self.action_dim)).copy()

        self.actor.eval()
        x = torch.as_tensor(states_x, dtype=torch.float32, device=self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        k_act = self.action_dim
        w_grid = torch.tensor(ACTION_WEIGHTS, dtype=torch.float32, device=self.device).view(1, k_act, 1)
        out = np.empty((n, k_act, k_act), dtype=np.float32)
        chunk = 4096
        for s0 in range(0, n, chunk):
            s1 = min(n, s0 + chunk)
            c_len = s1 - s0
            s_rep = x[s0:s1].unsqueeze(1).expand(c_len, k_act, self.state_dim)
            w_rep = w_grid.expand(c_len, k_act, 1)
            obs_c = torch.cat([s_rep, w_rep], dim=-1).reshape(c_len * k_act, self.obs_dim)
            probs_c = F.softmax(self.actor(obs_c), dim=-1).reshape(c_len, k_act, k_act)
            out[s0:s1] = probs_c.cpu().numpy()
        return out

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
            seg_starts = np.zeros(len(idx), dtype=bool)
            seg_starts[0] = True
            if len(idx) > 1:
                seg_starts[1:] = idx[1:] != (idx[:-1] + 1)
        else:
            s_arr = states
            r_arr = next_bar_returns
            c_arr = cost_rates
            seg_starts = np.zeros(len(s_arr), dtype=bool)
            if len(s_arr) > 0:
                seg_starts[0] = True

        n = len(s_arr)
        if n < 4:
            return []

        use_seg = bool(self.respect_segment_boundaries)
        seg_arg = seg_starts if use_seg else None
        seg_bounds = list(np.where(seg_starts)[0]) + [n] if use_seg else [0, n]
        seg_list = seg_starts.tolist()
        ar_n = np.arange(n)

        for pg in self.opt_actor.param_groups:
            pg["lr"] = self.lr_actor
            pg.pop("initial_lr", None)
        sched_actor = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_actor, T_max=max(1, total_rollouts)
        )
        mean_rewards: list[float] = []

        s_tensor = torch.as_tensor(s_arr, dtype=torch.float32, device=self.device)
        w_grid_dev = torch.tensor(ACTION_WEIGHTS, dtype=torch.float32, device=self.device).view(
            1, self.action_dim, 1
        )

        for r_step in range(total_rollouts):
            beta = self.per_beta_start + (self.per_beta_end - self.per_beta_start) * (
                r_step / max(1, total_rollouts - 1)
            )
            self.actor.eval()
            self.critic.eval()

            if self.include_prev_weight:
                with torch.no_grad():
                    s_rep = s_tensor.unsqueeze(1).expand(n, self.action_dim, self.state_dim)
                    w_rep = w_grid_dev.expand(n, self.action_dim, 1)
                    obs_all = torch.cat([s_rep, w_rep], dim=-1).reshape(
                        n * self.action_dim, self.obs_dim
                    )
                    logits_all = self.actor(obs_all).reshape(n, self.action_dim, self.action_dim)
                    logp_all = F.log_softmax(logits_all, dim=-1)
                    probs_all = torch.exp(logp_all)
                    ent_all = -(probs_all * logp_all).sum(dim=-1)
                    vals_all = self.critic(obs_all).reshape(n, self.action_dim)

                    probs_np = probs_all.cpu().numpy()
                    logp_np = logp_all.cpu().numpy()
                    ent_np = ent_all.cpu().numpy()
                    vals_np = vals_all.cpu().numpy()

                cdf_np = np.cumsum(probs_np, axis=-1)
                cdf_np[..., -1] = 1.0
                cdf_list = cdf_np.tolist()
                greedy_list = np.argmax(probs_np, axis=-1).tolist()
                u_list = self.rng.random(n).tolist()

                prev_a_arr = np.empty(n, dtype=np.int64)
                act_np = np.empty(n, dtype=np.int64)
                greedy_prev_a_arr = np.empty(n, dtype=np.int64)
                greedy_np = np.empty(n, dtype=np.int64)

                pa = NEUTRAL_ACTION_INDEX
                gpa = NEUTRAL_ACTION_INDEX
                for i in range(n):
                    if use_seg and seg_list[i]:
                        pa = NEUTRAL_ACTION_INDEX
                        gpa = NEUTRAL_ACTION_INDEX
                    prev_a_arr[i] = pa
                    greedy_prev_a_arr[i] = gpa
                    u = u_list[i]
                    cdf_row = cdf_list[i][pa]
                    if u < cdf_row[0]:
                        a = 0
                    elif u < cdf_row[1]:
                        a = 1
                    elif u < cdf_row[2]:
                        a = 2
                    elif u < cdf_row[3]:
                        a = 3
                    else:
                        a = 4
                    act_np[i] = a
                    pa = a

                    ga = greedy_list[i][gpa]
                    greedy_np[i] = ga
                    gpa = ga

                prev_w_np = ACTION_WEIGHTS[prev_a_arr]
                weights_np = ACTION_WEIGHTS[act_np]
                greedy_prev_w_np = ACTION_WEIGHTS[greedy_prev_a_arr]
                greedy_weights_np = ACTION_WEIGHTS[greedy_np]

                sampled_turnover = float(np.mean(np.abs(weights_np - prev_w_np)))
                greedy_turnover = float(np.mean(np.abs(greedy_weights_np - greedy_prev_w_np)))
                rollout_entropy = float(np.mean(ent_np[ar_n, prev_a_arr]))
                val_np = vals_np[ar_n, prev_a_arr].astype(np.float32)

                actions = torch.as_tensor(act_np, dtype=torch.long, device=self.device)
                old_log_probs = torch.as_tensor(
                    logp_np[ar_n, prev_a_arr, act_np], dtype=torch.float32, device=self.device
                )
                prev_w_t = torch.as_tensor(prev_w_np, dtype=torch.float32, device=self.device).unsqueeze(-1)
                obs_rollout = torch.cat([s_tensor, prev_w_t], dim=-1)
            else:
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

                if use_seg:
                    pw = np.empty(n, dtype=np.float64)
                    pw[0] = 0.0
                    if n > 1:
                        pw[1:] = weights_np[:-1]
                    pw[seg_starts] = 0.0
                    gpw = np.empty(n, dtype=np.float64)
                    gpw[0] = 0.0
                    if n > 1:
                        gpw[1:] = greedy_weights_np[:-1]
                    gpw[seg_starts] = 0.0
                    sampled_turnover = float(np.mean(np.abs(weights_np - pw)))
                    greedy_turnover = float(np.mean(np.abs(greedy_weights_np - gpw)))
                else:
                    sampled_turnover = float(np.mean(np.abs(np.diff(weights_np, prepend=0.0))))
                    greedy_turnover = float(np.mean(np.abs(np.diff(greedy_weights_np, prepend=0.0))))

                val_np = values.cpu().numpy()
                obs_rollout = s_tensor

            rewards_np, _ = compute_regime_rewards(
                regime_id=self.regime_id,
                bar_asset_returns=r_arr,
                chosen_weights=weights_np,
                one_way_cost_rates=c_arr,
                bear_window=bear_window,
                bear_dd_floor=bear_dd_floor,
                bear_cost_penalty=bear_cost_penalty,
                sideways_turnover_penalty=sideways_turnover_penalty,
                segment_starts=seg_arg,
            )
            m_rew = float(np.mean(rewards_np))
            mean_rewards.append(m_rew)

            deltas = np.empty(n, dtype=np.float32)
            advantages = np.empty(n, dtype=np.float32)
            gl = float(self.gamma * self.gae_lambda)
            for idx_b in range(len(seg_bounds) - 1):
                s0 = seg_bounds[idx_b]
                s1 = seg_bounds[idx_b + 1]
                if s1 - s0 > 1:
                    deltas[s0 : s1 - 1] = (
                        rewards_np[s0 : s1 - 1] + self.gamma * val_np[s0 + 1 : s1] - val_np[s0 : s1 - 1]
                    )
                deltas[s1 - 1] = rewards_np[s1 - 1] - val_np[s1 - 1]
                advantages[s0:s1] = lfilter(
                    [1.0], [1.0, -gl], deltas[s0:s1][::-1].astype(np.float64)
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

                    obs_mb = obs_rollout[b_idx_t]
                    cur_logits = self.actor(obs_mb)
                    cur_dist = torch.distributions.Categorical(logits=cur_logits)
                    new_lp = cur_dist.log_prob(actions[b_idx_t])
                    entropy = cur_dist.entropy()

                    ratio = torch.exp(new_lp - old_log_probs[b_idx_t])
                    surr1 = ratio * adv_t[b_idx_t]
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_t[b_idx_t]
                    actor_loss = -(w_t * (torch.min(surr1, surr2) + self.entropy_coef * entropy)).mean()

                    pred_v = self.critic(obs_mb).squeeze(-1)
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
