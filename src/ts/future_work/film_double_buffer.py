"""
Appendix D Reference Design ONLY (Unimplemented Future Work in Thesis Ch. VII & App. D).

NOTE (Revised Thesis p. 98 Disclaimer):
This module is an isolated reference design for future research (FiLM Eq. VII.2 + lock-free
double-buffered `mmap` + 1-hour dead-man switch `(gamma=0, beta=-10)`).
It is NOT used anywhere in the Chapter VI experiments, Student1DCNN training, or online backtest.
"""

from __future__ import annotations

import mmap
import struct
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ts.study3.student_model import Student1DCNN


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation `FiLM(h) = gamma * h + beta` (Eq. VII.2)."""

    def forward(self, h: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        return gamma * h + beta


class LockFreeFiLMDoubleBuffer:
    """
    2-slot ping-pong shared-memory buffer storing `(gamma[128], beta[128], timestamp_ms)`
    with a 1-hour (`3,600,000 ms`) stale-signal dead-man switch (`gamma=0, beta=-10`).
    """

    def __init__(self, hidden_dim: int = 128, max_staleness_sec: float = 3600.0) -> None:
        self.hidden_dim = int(hidden_dim)
        self.stale_timeout_ms = int(max_staleness_sec * 1000.0)
        self._slot_fmt = f"<Q{self.hidden_dim}f{self.hidden_dim}f"
        self._slot_bytes = struct.calcsize(self._slot_fmt)
        self._header_bytes = 8
        self._total_bytes = self._header_bytes + 2 * self._slot_bytes
        self._buf = mmap.mmap(-1, self._total_bytes)
        struct.pack_into("<i", self._buf, 0, 0)

    def teacher_write(
        self, gamma: np.ndarray, beta: np.ndarray, now_ts: float | None = None
    ) -> int:
        ts_ms = int((now_ts if now_ts is not None else time.time()) * 1000.0)
        cur_active = struct.unpack_from("<i", self._buf, 0)[0]
        inactive_slot = 1 - cur_active
        offset = self._header_bytes + inactive_slot * self._slot_bytes
        g = np.asarray(gamma, dtype=np.float32).reshape(self.hidden_dim)
        b = np.asarray(beta, dtype=np.float32).reshape(self.hidden_dim)
        struct.pack_into(self._slot_fmt, self._buf, offset, ts_ms, *g.tolist(), *b.tolist())
        # Atomic index flip after inactive slot payload is completely written
        struct.pack_into("<i", self._buf, 0, inactive_slot)
        return inactive_slot

    def student_read(self, now_ts: float | None = None) -> Tuple[np.ndarray, np.ndarray, bool]:
        cur_ms = int((now_ts if now_ts is not None else time.time()) * 1000.0)
        active_slot = struct.unpack_from("<i", self._buf, 0)[0]
        offset = self._header_bytes + active_slot * self._slot_bytes
        unpacked = struct.unpack_from(self._slot_fmt, self._buf, offset)
        last_ts = int(unpacked[0])

        if last_ts == 0 or (cur_ms - last_ts) > self.stale_timeout_ms:
            # Dead-man's switch safety fallback: gamma = 0, beta = -10
            return (
                np.zeros(self.hidden_dim, dtype=np.float32),
                np.full(self.hidden_dim, -10.0, dtype=np.float32),
                True,
            )

        gamma = np.array(unpacked[1 : 1 + self.hidden_dim], dtype=np.float32)
        beta = np.array(unpacked[1 + self.hidden_dim :], dtype=np.float32)
        return gamma, beta, False

    def close(self) -> None:
        self._buf.close()


# Alias for external callers
DoubleBufferedContextStore = LockFreeFiLMDoubleBuffer


class FiLMConditionedStudent1DCNN(nn.Module):
    """Appendix D Future Reference Student wrapper modulating `H_student` via shared-memory FiLM."""

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.student = Student1DCNN(hidden_dim=hidden_dim)
        self.film = FiLMLayer()

    @torch.no_grad()
    def act_with_context(
        self,
        x: torch.Tensor,
        store: LockFreeFiLMDoubleBuffer,
        now_ts: float | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        gamma_np, beta_np, is_stale = store.student_read(now_ts=now_ts)
        h = self.student.encode_hidden(x)
        g_t = torch.as_tensor(gamma_np, dtype=h.dtype, device=h.device).unsqueeze(0)
        b_t = torch.as_tensor(beta_np, dtype=h.dtype, device=h.device).unsqueeze(0)
        h_mod = self.film(h, g_t, b_t)
        logits = self.student.actor(h_mod)
        probs = F.softmax(logits, dim=-1)
        return torch.argmax(probs, dim=-1), probs, is_stale
