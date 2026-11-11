"""
Walk-Forward (Phase 2) & K-Fold (Phase 1) Splitters (Chapter VI Sec 3.1, Fig. 6.1).

Phase 2 Primary Evaluation Protocol:
- 9 Rolling Folds covering `2021-10-01 00:00 UTC` through `2023-12-31 23:00 UTC` (27 months):
  - Fold 1: Train `2020-10-01` .. `2021-09-30` (12m) -> Test `2021-10-01` .. `2021-12-31` (3m)
  - Fold 2: Train `2021-01-01` .. `2021-12-31` (12m) -> Test `2022-01-01` .. `2022-03-31` (3m)
  - ...
  - Fold 9: Train `2022-10-01` .. `2023-09-30` (12m) -> Test `2023-10-01` .. `2023-12-31` (3m)
- Strictly enforces `train_end < test_start` on every fold via `temporal_guard.assert_strictly_before`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from ts.data.temporal_guard import assert_strictly_before


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_indices: np.ndarray
    test_indices: np.ndarray


def generate_phase2_walk_forward_folds(
    timestamps: pd.DatetimeIndex,
    first_train_start: str = "2020-10-01T00:00:00Z",
    train_months: int = 12,
    test_months: int = 3,
    step_months: int = 3,
    num_folds: int = 9,
) -> List[WalkForwardFold]:
    """
    Generate the exact 9 rolling walk-forward folds from Figure 6.1.
    The union of the 9 disjoint test windows spans `2021-10-01` to `2023-12-31` without overlap.
    """
    idx = pd.DatetimeIndex(timestamps)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")

    base_start = pd.Timestamp(first_train_start)
    base_start = base_start.tz_localize("UTC") if base_start.tzinfo is None else base_start.tz_convert("UTC")

    folds: List[WalkForwardFold] = []
    for k in range(num_folds):
        t_start = base_start + pd.DateOffset(months=k * step_months)
        test_s = t_start + pd.DateOffset(months=train_months)
        t_end = test_s - pd.Timedelta(seconds=1)
        test_e = test_s + pd.DateOffset(months=test_months) - pd.Timedelta(seconds=1)

        assert_strictly_before(t_end, test_s, context=f"WalkForward Fold {k + 1}")

        tr_idx = np.where((idx >= t_start) & (idx <= t_end))[0]
        te_idx = np.where((idx >= test_s) & (idx <= test_e))[0]

        if len(tr_idx) > 0 and len(te_idx) > 0:
            assert_strictly_before(
                idx[tr_idx[-1]], idx[te_idx[0]], context=f"WalkForward Fold {k + 1} bar boundary"
            )
            folds.append(
                WalkForwardFold(
                    fold_id=k + 1,
                    train_start=t_start,
                    train_end=t_end,
                    test_start=test_s,
                    test_end=test_e,
                    train_indices=tr_idx,
                    test_indices=te_idx,
                )
            )
    return folds


def generate_phase1_kfold_splits(
    eval_indices: np.ndarray,
    n_splits: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Phase 1 Conventional K-Fold CV Splitter (Table 6.4):
    Splits the evaluation period into `K` contiguous folds where each fold is held out once
    while training on the remaining `K - 1` folds (including future folds — intentionally
    reproducing the optimistic conventional benchmark contrasted against Phase 2 in Ch. VI Sec 3.1).
    """
    blocks = np.array_split(np.asarray(eval_indices, dtype=np.int64), n_splits)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_splits):
        test_idx = blocks[k]
        train_idx = np.concatenate([blocks[j] for j in range(n_splits) if j != k])
        splits.append((train_idx, test_idx))
    return splits
