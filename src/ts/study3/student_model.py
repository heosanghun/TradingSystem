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
