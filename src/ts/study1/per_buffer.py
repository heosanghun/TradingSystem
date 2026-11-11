"""
On-Policy Rollout Segment-Tree Prioritized Experience Replay (Ch. III Sec 3.3, Eq. III.20–III.22).

Strictly operates ONLY within the current on-policy rollout buffer `B` collected under
`π_{θ_old}` (buffer is discarded after each PPO policy update so stale off-policy
importance ratios never corrupt PPO's clipped objective!).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


class _SumMinSegmentTree:
    """Binary segment tree supporting O(log N) prefix-sum lookup and global min."""

    def __init__(self, capacity: int) -> None:
        size = 1
        while size < capacity:
            size *= 2
        self.size = size
        self.sum_tree = np.zeros(2 * size, dtype=np.float64)
        self.min_tree = np.full(2 * size, np.inf, dtype=np.float64)

    def set(self, idx: int, value: float) -> None:
        pos = idx + self.size
        self.sum_tree[pos] = value
        self.min_tree[pos] = value if value > 0.0 else np.inf
        pos //= 2
        while pos >= 1:
            left, right = 2 * pos, 2 * pos + 1
            self.sum_tree[pos] = self.sum_tree[left] + self.sum_tree[right]
            self.min_tree[pos] = min(self.min_tree[left], self.min_tree[right])
            pos //= 2

    def total_sum(self) -> float:
        return float(self.sum_tree[1])

    def min_positive(self) -> float:
        return float(self.min_tree[1])

    def find_prefixsum_idx(self, mass: float) -> int:
        pos = 1
        while pos < self.size:
            left = 2 * pos
            if self.sum_tree[left] >= mass:
                pos = left
            else:
                mass -= self.sum_tree[left]
                pos = left + 1
        return pos - self.size


class SegmentTreePERBuffer:
    """
    Implements Eqs. (III.20)–(III.22):
      p_i = |δ_i| + ε
      P(i) = p_i^α / Σ_{k ∈ B} p_k^α
      ω_i = (1 / (N_B * P(i)))^β / max_j ω_j
    """

    def __init__(self, alpha: float = 0.6, eps: float = 1e-6) -> None:
        self.alpha = alpha
        self.eps = eps
        self.n_items = 0
        self.tree: _SumMinSegmentTree | None = None

    def build_from_td_errors(self, td_errors: np.ndarray) -> None:
        td = np.asarray(td_errors, dtype=np.float64)
        self.n_items = len(td)
        self.tree = _SumMinSegmentTree(max(1, self.n_items))
        priorities_alpha = np.power(np.abs(td) + self.eps, self.alpha)
        for i, val in enumerate(priorities_alpha):
            self.tree.set(i, float(val))

    def update_priorities(self, indices: np.ndarray, new_td_errors: np.ndarray) -> None:
        if self.tree is None:
            return
        vals = np.power(np.abs(new_td_errors) + self.eps, self.alpha)
        for idx, val in zip(indices, vals):
            if 0 <= int(idx) < self.n_items:
                self.tree.set(int(idx), float(val))

    def sample(
        self, batch_size: int, beta: float, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return `(sampled_indices, normalized_is_weights)`."""
        if self.tree is None or self.n_items == 0:
            raise RuntimeError("SegmentTreePERBuffer must be initialized with `build_from_td_errors` first.")

        eff_bs = min(batch_size, self.n_items)
        total = max(self.tree.total_sum(), 1e-12)
        segment = total / eff_bs

        indices = np.empty(eff_bs, dtype=np.int64)
        probs = np.empty(eff_bs, dtype=np.float64)

        for k in range(eff_bs):
            mass = rng.uniform(k * segment, (k + 1) * segment)
            idx = min(self.tree.find_prefixsum_idx(mass), self.n_items - 1)
            indices[k] = idx
            p_alpha = max(self.tree.sum_tree[idx + self.tree.size], 1e-12)
            probs[k] = p_alpha / total

        min_prob = max(self.tree.min_positive() / total, 1e-12)
        max_weight = (self.n_items * min_prob) ** (-beta)
        weights = ((self.n_items * probs) ** (-beta)) / max(max_weight, 1e-12)
        return indices, weights.astype(np.float32)
