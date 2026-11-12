"""
On-Policy Rollout Segment-Tree Prioritized Experience Replay (Chapter III Sec 3.3, Eqs. III.20-III.22).

Operates strictly within the current on-policy rollout buffer `B` collected under `pi_{theta_old}`;
the buffer is discarded after each PPO policy update so off-policy stale importance ratios do not
enter the clipped surrogate objective.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


class _SumMinSegmentTree:
    """Binary segment tree supporting O(log N) prefix-sum lookup and global min."""

    def __init__(self, capacity: int) -> None:
        size = 1
        depth = 0
        while size < capacity:
            size *= 2
            depth += 1
        self.size = size
        self.depth = depth
        self.sum_tree = np.zeros(2 * size, dtype=np.float64)
        self.min_tree = np.full(2 * size, np.inf, dtype=np.float64)

    def build_bulk(self, values: np.ndarray) -> None:
        n = len(values)
        self.sum_tree.fill(0.0)
        self.min_tree.fill(np.inf)
        if n > 0:
            leaves = np.asarray(values, dtype=np.float64)
            self.sum_tree[self.size : self.size + n] = leaves
            pos_leaves = np.where(leaves > 0.0, leaves, np.inf)
            self.min_tree[self.size : self.size + n] = pos_leaves

        # Vectorized level-by-level reduction (depth <= 16 -> <= 16 array ops)
        width = self.size // 2
        while width >= 1:
            parents = np.arange(width, 2 * width)
            left = 2 * parents
            right = left + 1
            self.sum_tree[parents] = self.sum_tree[left] + self.sum_tree[right]
            self.min_tree[parents] = np.minimum(self.min_tree[left], self.min_tree[right])
            width //= 2

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

    def find_prefixsum_idx_batch(self, masses: np.ndarray) -> np.ndarray:
        pos = np.ones(len(masses), dtype=np.int64)
        rem = np.asarray(masses, dtype=np.float64).copy()
        for _ in range(self.depth):
            left = 2 * pos
            left_sum = self.sum_tree[left]
            go_right = rem > left_sum
            rem = np.where(go_right, rem - left_sum, rem)
            pos = np.where(go_right, left + 1, left)
        return pos - self.size

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
    Implements Eqs. (III.20)-(III.22):
      p_i = |delta_i| + eps
      P(i) = p_i^alpha / sum_{k in B} p_k^alpha
      omega_i = (1 / (N_B * P(i)))^beta  (capped at max_is_weight and batch-mean normalized)
    """

    def __init__(self, alpha: float = 0.6, eps: float = 1e-6, max_is_weight: float = 10.0) -> None:
        self.alpha = alpha
        self.eps = eps
        self.max_is_weight = max_is_weight
        self.n_items = 0
        self.tree: _SumMinSegmentTree | None = None

    def build_from_td_errors(self, td_errors: np.ndarray) -> None:
        td = np.asarray(td_errors, dtype=np.float64)
        self.n_items = len(td)
        self.tree = _SumMinSegmentTree(max(1, self.n_items))
        priorities_alpha = np.power(np.abs(td) + self.eps, self.alpha)
        self.tree.build_bulk(priorities_alpha)

    def update_priorities(self, indices: np.ndarray, new_td_errors: np.ndarray) -> None:
        if self.tree is None or self.n_items == 0:
            return
        idx_arr = np.asarray(indices, dtype=np.int64)
        valid = (idx_arr >= 0) & (idx_arr < self.n_items)
        if not np.any(valid):
            return
        idx_v = idx_arr[valid]
        vals = np.power(np.abs(np.asarray(new_td_errors, dtype=np.float64)[valid]) + self.eps, self.alpha)
        rev_uniq, rev_first = np.unique(idx_v[::-1], return_index=True)
        last_pos = (len(idx_v) - 1) - rev_first
        leaf_pos = rev_uniq + self.tree.size
        vals_u = vals[last_pos]
        self.tree.sum_tree[leaf_pos] = vals_u
        self.tree.min_tree[leaf_pos] = np.where(vals_u > 0.0, vals_u, np.inf)

        parents = np.unique(leaf_pos // 2)
        while len(parents) > 0 and parents[0] >= 1:
            left = 2 * parents
            right = left + 1
            self.tree.sum_tree[parents] = self.tree.sum_tree[left] + self.tree.sum_tree[right]
            self.tree.min_tree[parents] = np.minimum(self.tree.min_tree[left], self.tree.min_tree[right])
            parents = np.unique(parents // 2)
            if len(parents) > 0 and parents[0] == 0:
                parents = parents[1:]

    def sample(
        self, batch_size: int, beta: float, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return `(sampled_indices, normalized_is_weights)`."""
        if self.tree is None or self.n_items == 0:
            raise RuntimeError("SegmentTreePERBuffer must be initialized with build_from_td_errors first.")

        eff_bs = min(batch_size, self.n_items)
        total = max(self.tree.total_sum(), 1e-12)
        segment = total / eff_bs

        k_arr = np.arange(eff_bs, dtype=np.float64)
        masses = rng.uniform(k_arr * segment, (k_arr + 1.0) * segment)
        indices = np.minimum(self.tree.find_prefixsum_idx_batch(masses), self.n_items - 1)
        p_alpha = np.maximum(self.tree.sum_tree[indices + self.tree.size], 1e-12)
        probs = p_alpha / total

        raw_weights = np.power(np.maximum(self.n_items * probs, 1e-12), -beta)
        clipped = np.minimum(raw_weights, self.max_is_weight)
        weights = clipped / max(float(np.mean(clipped)), 1e-12)
        return indices, weights.astype(np.float32)
