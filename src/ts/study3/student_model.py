"""
System 1 Lightweight Student Model (`Student1DCNN`) matching Appendix C.

Implements Chapter V Sec 3, Sec 5, and Appendix C (p. 97):
- Input: `(B, 40, 100)` (40 microstructure features x 100 time steps)
- `Conv1d(40, 64, k=3, pad=1) -> ReLU -> MaxPool1d(2, 2)`
- `Conv1d(64, 128, k=3, pad=1) -> ReLU -> MaxPool1d(2, 2)`
- `AdaptiveAvgPool1d(1) -> LayerNorm(128)` -> `H_student in R^{B x 128}`
- Heads:
  - `actor`: `Linear(128, 5)` (only head executed during online deployment)
  - `critic`: `Linear(128, 1)` (used only during offline training for `L_task`)
  - `projection_head`: `Linear(128, 256) -> GELU -> Linear(256, 768)` with no output
    activation (used only during offline training for `L_reasoning` cosine alignment)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Student1DCNN(nn.Module):
    """Appendix C Student 1D-CNN with standalone online execution path."""

    def __init__(
        self,
        input_dim: int = 40,
        hidden_dim: int = 128,
        action_dim: int = 5,
        llm_dim: int = 768,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(hidden_dim)  # Keeps ||H|| away from zero (Property 5.1)

        # Projection head: Linear-GELU-Linear, NO output activation
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, llm_dim),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def encode_hidden(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 40, 100)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        return self.norm(self.gap(x).squeeze(-1))  # H: (B, 128)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full training forward pass returning `(action_logits, state_value, h_proj)`."""
        h = self.encode_hidden(x)                  # H:  (B, 128)
        h_proj = self.projection_head(h)           # H': (B, 768)
        return self.actor(h), self.critic(h), h_proj

    @torch.no_grad()
    def act_online(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Online deployment path (Ch. V Sec 5.2 & Fig. 5.2):
        Executes only the 1D-CNN backbone and `actor` head without running
        Teacher, LLM, FinBERT, `critic`, or `projection_head`.
        Returns `(greedy_actions, action_probs)`.
        """
        h = self.encode_hidden(x)
        logits = self.actor(h)
        probs = F.softmax(logits, dim=-1)
        actions = torch.argmax(probs, dim=-1)
        return actions, probs


def apply_action_confidence_gate(probs, threshold: float, initial_action: int = 2):
    """
    Action-level counterpart of the regime rule in Eq. (III.16):
      a_t = argmax_a p_t(a)   if max_a p_t(a) >= threshold
      a_t = a_{t-1}           otherwise
    `probs` is `(N, 5)` in time order. `threshold <= 0` returns the plain argmax, which is
    the behaviour without a gate. `initial_action` is the position held before the first
    bar (index 2 is cash).
    """
    import numpy as np

    p = np.asarray(probs, dtype=np.float64)
    best = np.argmax(p, axis=1).astype(np.int64)
    if threshold <= 0.0 or len(p) == 0:
        return best
    top = p.max(axis=1)
    out = np.empty(len(p), dtype=np.int64)
    cur = int(initial_action)
    for t in range(len(p)):
        if top[t] >= threshold:
            cur = int(best[t])
        out[t] = cur
    return out


def smooth_action_probabilities(probs, span: int):
    """
    Causal exponentially weighted average of the per-bar action probabilities:
      q_t = alpha * p_t + (1 - alpha) * q_{t-1},  alpha = 2 / (span + 1),  q_0 = p_0
    `span <= 1` returns the input unchanged. Each row of the result still sums to one.
    The average uses only the current and earlier bars.
    """
    import numpy as np

    p = np.asarray(probs, dtype=np.float64)
    if span <= 1 or len(p) == 0:
        return p
    alpha = 2.0 / (float(span) + 1.0)
    out = np.empty_like(p)
    out[0] = p[0]
    for t in range(1, len(p)):
        out[t] = alpha * p[t] + (1.0 - alpha) * out[t - 1]
    return out


def execute_student_actions(probs, threshold: float, smooth_span: int = 1, initial_action: int = 2):
    """Execution rule of the student: smooth the probabilities, then apply the confidence gate."""
    return apply_action_confidence_gate(smooth_action_probabilities(probs, smooth_span), threshold, initial_action)


def execute_student_weights(probs, action_weights, mode: str = "gate", threshold: float = 0.0,
                            smooth_span: int = 1, band: float = 0.25, initial_weight: float = 0.0):
    """
    Target weight per bar from the student's action probabilities, in time order.

    mode "gate"          : the confidence-gated action of `execute_student_actions`, as a weight.
    mode "expected"      : the expected weight under the action distribution,
                           pi_t = sum_a p_t(a) * w(a). This is the quantity the net-of-cost
                           training objective optimises.
    mode "expected_band" : a position-dependent no-trade band around the held weight. The held
                           weight changes only when |pi_t - held| > band, and then moves to the
                           action level closest to pi_t. With band = half the spacing of the
                           action grid, the position stays on the five thesis action levels.
    """
    import numpy as np

    p = np.asarray(probs, dtype=np.float64)
    aw = np.asarray(action_weights, dtype=np.float64)
    if mode == "gate":
        return aw[execute_student_actions(p, threshold, smooth_span)]
    pi = p @ aw
    if mode == "expected":
        return pi
    if mode == "expected_band":
        out = np.empty(len(pi), dtype=np.float64)
        cur = float(initial_weight)
        for t in range(len(pi)):
            if abs(pi[t] - cur) > band:
                cur = float(aw[int(np.argmin(np.abs(aw - pi[t])))])
            out[t] = cur
        return out
    raise ValueError(f"unknown execution mode: {mode}")
