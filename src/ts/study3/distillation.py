"""
Dual-Process Tri-Objective Reasoning Distillation & Property 5.1 Verifier (Ch. V Sec 4).

Implements Eqs. (V.7)–(V.11) & Appendix C.2–C.3 with 100% intact PyTorch autograd:
  `L_total = λ_1 * L_task + λ_2 * L_policy + λ_3 * L_reasoning`
  Default `(λ_1, λ_2, λ_3) = (0.4, 0.4, 0.2)`, global grad clip `0.5`, AdamW `50` epochs.

Supports `λ_3 ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}` sensitivity sweeps (setting
`λ_1 = λ_2 = (1 - λ_3) / 2` so `λ_1 + λ_2 + λ_3 == 1.0`) for Table 6.11!
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ts.study1.ppo_expert import ACTION_WEIGHTS
from ts.study3.student_model import Student1DCNN
from ts.study3.teacher_signal import TeacherSignalPack


class ReasoningDistillationLoss(nn.Module):
    """Exact tri-objective loss from Eq. (V.7)–(V.11) & Appendix C.2."""

    def __init__(
        self,
        lambda_task: float = 0.4,
        lambda_policy: float = 0.4,
        lambda_reason: float = 0.2,
        tau: float = 2.0,
        c_v: float = 0.5,
        c_e: float = 0.01,
    ) -> None:
        super().__init__()
        if abs((lambda_task + lambda_policy + lambda_reason) - 1.0) > 1e-5:
            raise ValueError("Distillation weights (lambda_1, lambda_2, lambda_3) must sum to 1.0")
        self.lambda_task = lambda_task
        self.lambda_policy = lambda_policy
        self.lambda_reason = lambda_reason
        self.tau = tau
        self.c_v = c_v
        self.c_e = c_e
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.cosine_sim = nn.CosineSimilarity(dim=-1, eps=1e-8)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_projected: torch.Tensor,
        teacher_reasoning: torch.Tensor,
        ppo_actor_loss: torch.Tensor,
        ppo_critic_loss: torch.Tensor,
        policy_entropy: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Eq. (V.7): Task Loss (PPO clip + c_v * value_mse - c_e * entropy)
        loss_task = ppo_actor_loss + self.c_v * ppo_critic_loss - self.c_e * policy_entropy

        # 2. Eq. (V.8): Policy Distillation Loss (τ^2 * KL(P_teacher^τ || P_student^τ))
        student_log_probs = F.log_softmax(student_logits / self.tau, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.tau, dim=-1)
        loss_policy = (self.tau ** 2) * self.kl_div(student_log_probs, teacher_probs)

        # 3. Eq. (V.9): Reasoning Cosine Distance Loss (1 - cos(E_reasoning, H'_student))
        cosine_score = self.cosine_sim(student_projected, teacher_reasoning)
        loss_reasoning = 1.0 - cosine_score.mean()

        total_loss = (
            self.lambda_task * loss_task
            + self.lambda_policy * loss_policy
            + self.lambda_reason * loss_reasoning
        )
        return total_loss, loss_task, loss_policy, loss_reasoning


def verify_property_5_1_orthogonality(
    h_projected: torch.Tensor, e_reasoning: torch.Tensor
) -> Dict[str, float]:
    """
    Empirically & analytically verifies Revised Thesis Property 5.1 (p. 58, Eq. V.10):
      ∂L_reasoning / ∂H' = -(1 / (B * ||H'||_2)) * (E_hat - (E_hat · H_hat) H_hat)
      => H_hat · (∂L_reasoning / ∂H') == 0 (orthogonal rotation without radial norm explosion)
      => ||∂L_reasoning / ∂H'_b||_2 * B <= 1 / ||H'_b||_2
    """
    h = h_projected.detach().clone().requires_grad_(True)
    e = e_reasoning.detach()
    cos = F.cosine_similarity(h, e, dim=-1, eps=1e-12)
    loss = 1.0 - cos.mean()
    loss.backward()
    assert h.grad is not None

    b = h.shape[0]
    per_sample_grad = h.grad * b  # undo batch 1/B factor to match single-sample Eq. V.10
    h_norm = torch.norm(h.detach(), dim=-1, keepdim=True).clamp(min=1e-12)
    h_hat = h.detach() / h_norm

    inner_prod = torch.sum(h_hat * per_sample_grad, dim=-1).abs()
    grad_norm = torch.norm(per_sample_grad, dim=-1)
    upper_bound = (1.0 / h_norm.squeeze(-1)) + 1e-6

    return {
        "max_abs_dot_product": float(inner_prod.max().item()),
        "max_grad_norm_minus_bound": float((grad_norm - upper_bound).max().item()),
        "mean_cosine_similarity": float(cos.mean().item()),
    }


class StudentDistillationTrainer:
    """
    Implements the 8-step Offline Distillation Training Procedure (Ch. V Sec 4.4, p. 59).
    """

    def __init__(
        self,
        lambda_reasoning: float = 0.2,
        tau: float = 2.0,
        ppo_clip: float = 0.2,
        c_v: float = 0.5,
        c_e: float = 0.01,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        lr: float = 3e-4,
        batch_size: int = 256,
        epochs: int = 50,
        max_grad_norm: float = 0.5,
        seed: int = 42,
        device: Optional[torch.device] = None,
    ) -> None:
        self.lambda_reasoning = float(lambda_reasoning)
        self.lambda_task = 0.5 * (1.0 - self.lambda_reasoning)
        self.lambda_policy = 0.5 * (1.0 - self.lambda_reasoning)
        self.tau = tau
        self.ppo_clip = ppo_clip
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_grad_norm = max_grad_norm
        self.seed = seed
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.loss_fn = ReasoningDistillationLoss(
            lambda_task=self.lambda_task,
            lambda_policy=self.lambda_policy,
            lambda_reason=self.lambda_reasoning,
            tau=tau,
            c_v=c_v,
            c_e=c_e,
        )

    def fit(self, pack: TeacherSignalPack) -> Tuple[Student1DCNN, List[Dict[str, float]]]:
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        model = Student1DCNN().to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)

        n = len(pack.lob_windows)
        lob_t = torch.as_tensor(pack.lob_windows, dtype=torch.float32, device=self.device)
        t_logits_t = torch.as_tensor(pack.teacher_logits, dtype=torch.float32, device=self.device)
        t_reason_t = torch.as_tensor(pack.teacher_reasoning, dtype=torch.float32, device=self.device)

        action_weights_np = ACTION_WEIGHTS
        history: List[Dict[str, float]] = []

        for epoch in range(self.epochs):
            # Collect fresh on-policy Student rollout rewards & GAE targets on the LOB dataset
            model.eval()
            with torch.no_grad():
                all_logits_list = []
                all_vals_list = []
                for s in range(0, n, self.batch_size):
                    e = min(n, s + self.batch_size)
                    lg, vl, _ = model(lob_t[s:e])
                    all_logits_list.append(lg)
                    all_vals_list.append(vl.squeeze(-1))
                all_logits = torch.cat(all_logits_list, dim=0)
                all_vals = torch.cat(all_vals_list, dim=0)
                dist = torch.distributions.Categorical(logits=all_logits)
                sampled_actions = dist.sample()
                old_log_probs = dist.log_prob(sampled_actions)

            act_np = sampled_actions.cpu().numpy()
            val_np = all_vals.cpu().numpy()
            w_chosen = action_weights_np[act_np]
            prev_w = np.roll(w_chosen, 1)
            prev_w[0] = 0.0
            step_rewards = (
                w_chosen * pack.bar_returns
                - pack.cost_rates * np.abs(w_chosen - prev_w)
            ).astype(np.float32)

            advantages = np.zeros(n, dtype=np.float32)
            gae = 0.0
            for i in reversed(range(n)):
                next_v = val_np[i + 1] if (i + 1 < n and pack.bar_indices[i + 1] >= pack.bar_indices[i]) else 0.0
                delta = step_rewards[i] + self.gamma * next_v - val_np[i]
                gae = delta + self.gamma * self.gae_lambda * gae
                advantages[i] = gae

            returns_np = advantages + val_np
            adv_std = float(np.std(advantages))
            norm_adv = (advantages - float(np.mean(advantages))) / max(adv_std, 1e-6)

            adv_t = torch.as_tensor(norm_adv, dtype=torch.float32, device=self.device)
            ret_t = torch.as_tensor(returns_np, dtype=torch.float32, device=self.device)

            # Mini-batch SGD over all 3 losses
            model.train()
            perm = rng.permutation(n)
            ep_tot, ep_tsk, ep_pol, ep_rsn = 0.0, 0.0, 0.0, 0.0
            n_batches = 0

            for s in range(0, n, self.batch_size):
                b_idx = torch.as_tensor(perm[s : s + self.batch_size], dtype=torch.long, device=self.device)
                optimizer.zero_grad()

                s_logits, s_val, h_proj = model(lob_t[b_idx])
                cur_dist = torch.distributions.Categorical(logits=s_logits)
                new_lp = cur_dist.log_prob(sampled_actions[b_idx])
                ratio = torch.exp(new_lp - old_log_probs[b_idx])

                surr1 = ratio * adv_t[b_idx]
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * adv_t[b_idx]
                ppo_actor_loss = -torch.min(surr1, surr2).mean()
                ppo_critic_loss = F.mse_loss(s_val.squeeze(-1), ret_t[b_idx])
                entropy = cur_dist.entropy().mean()

                tot_l, tsk_l, pol_l, rsn_l = self.loss_fn(
                    student_logits=s_logits,
                    teacher_logits=t_logits_t[b_idx],
                    student_projected=h_proj,
                    teacher_reasoning=t_reason_t[b_idx],
                    ppo_actor_loss=ppo_actor_loss,
                    ppo_critic_loss=ppo_critic_loss,
                    policy_entropy=entropy,
                )
                tot_l.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.max_grad_norm)
                optimizer.step()

                ep_tot += float(tot_l.item())
                ep_tsk += float(tsk_l.item())
                ep_pol += float(pol_l.item())
                ep_rsn += float(rsn_l.item())
                n_batches += 1

            denom = max(1, n_batches)
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "total_loss": ep_tot / denom,
                    "task_loss": ep_tsk / denom,
                    "policy_loss": ep_pol / denom,
                    "reasoning_loss": ep_rsn / denom,
                }
            )

        model.eval()
        return model, history
