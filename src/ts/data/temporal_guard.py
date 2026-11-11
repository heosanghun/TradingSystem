"""
Point-in-time temporal integrity assertions (Principle #4).

Verifies at runtime and in unit tests that features, news items, regime labels,
and Teacher distillation signals never reference information after decision time `t`.
"""

from __future__ import annotations

import pandas as pd


class TemporalLeakageError(AssertionError):
    """Raised whenever any look-ahead or train/test temporal boundary violation is detected."""


def assert_monotonic_utc_index(df: pd.DataFrame, name: str = "dataframe") -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"{name} must have a DatetimeIndex, got {type(df.index)}")
    if df.index.tz is None:
        raise ValueError(f"{name} DatetimeIndex must be timezone-aware (UTC).")
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{name} index is not strictly monotonic increasing.")
    if df.index.has_duplicates:
        raise ValueError(f"{name} index contains duplicate timestamps.")


def assert_strictly_before(
    earlier: pd.Timestamp,
    later: pd.Timestamp,
    context: str = "temporal_boundary",
) -> None:
    """Assert `earlier < later` in UTC."""
    t_early = earlier.tz_localize("UTC") if earlier.tzinfo is None else earlier.tz_convert("UTC")
    t_late = later.tz_localize("UTC") if later.tzinfo is None else later.tz_convert("UTC")
    if t_early >= t_late:
        raise TemporalLeakageError(
            f"Temporal boundary violation in {context}: {t_early} >= {t_late}"
        )


def assert_no_future_leakage(
    item_timestamps: pd.Series | pd.DatetimeIndex,
    decision_timestamp: pd.Timestamp,
    strict_less: bool = True,
    context: str = "feature",
) -> None:
    """Ensure all input records occurred strictly before (or at) `decision_timestamp`."""
    if len(item_timestamps) == 0:
        return
    ts_idx = pd.DatetimeIndex(item_timestamps)
    if ts_idx.tz is None:
        ts_idx = ts_idx.tz_localize("UTC")
    else:
        ts_idx = ts_idx.tz_convert("UTC")

    dec_ts = (
        decision_timestamp.tz_localize("UTC")
        if decision_timestamp.tzinfo is None
        else decision_timestamp.tz_convert("UTC")
    )
    max_ts = ts_idx.max()
    if strict_less and max_ts >= dec_ts:
        raise TemporalLeakageError(
            f"Future leakage in {context}: max item timestamp {max_ts} >= decision time {dec_ts}"
        )
    if not strict_less and max_ts > dec_ts:
        raise TemporalLeakageError(
            f"Future leakage in {context}: max item timestamp {max_ts} > decision time {dec_ts}"
        )


def assert_purged_train_horizon(
    last_labeled_bar_idx: int,
    forward_horizon_bars: int,
    train_fold_end_idx_exclusive: int,
) -> None:
    """
    Ensure forward-looking labels (e.g., Phase 2 Trend Scanning over [t, t+L])
    never read prices at or beyond `train_fold_end_idx_exclusive` (test fold start).
    """
    furthest_idx = last_labeled_bar_idx + forward_horizon_bars
    if furthest_idx >= train_fold_end_idx_exclusive:
        raise TemporalLeakageError(
            f"Trend-scanning label at {last_labeled_bar_idx} with horizon {forward_horizon_bars} "
            f"reaches index {furthest_idx} >= test fold boundary {train_fold_end_idx_exclusive}."
        )
