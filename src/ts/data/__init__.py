"""Data collection, causal point-in-time alignment, and LOB microstructure processing."""

from ts.data.binance_fetcher import BinanceDataFetcher
from ts.data.lob_processor import LOBWindowProcessor
from ts.data.macro_fetcher import DEFAULT_60_ASSET_UNIVERSE, MacroAssetFetcher
from ts.data.news_fetcher import PointInTimeNewsStore
from ts.data.temporal_guard import (
    TemporalLeakageError,
    assert_monotonic_utc_index,
    assert_no_future_leakage,
    assert_purged_train_horizon,
    assert_strictly_before,
)

__all__ = [
    "BinanceDataFetcher",
    "LOBWindowProcessor",
    "DEFAULT_60_ASSET_UNIVERSE",
    "MacroAssetFetcher",
    "PointInTimeNewsStore",
    "TemporalLeakageError",
    "assert_monotonic_utc_index",
    "assert_no_future_leakage",
    "assert_purged_train_horizon",
    "assert_strictly_before",
]
