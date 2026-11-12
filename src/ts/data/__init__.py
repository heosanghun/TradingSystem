"""Data interfaces with lazy imports."""
from importlib import import_module

_EXPORTS = {
    'BinanceDataFetcher': 'ts.data.binance_fetcher',
    'LOBWindowProcessor': 'ts.data.lob_processor',
    'DEFAULT_60_ASSET_UNIVERSE': 'ts.data.macro_fetcher',
    'MacroAssetFetcher': 'ts.data.macro_fetcher',
    'PointInTimeNewsStore': 'ts.data.news_fetcher',
    'TemporalLeakageError': 'ts.data.temporal_guard',
    'assert_monotonic_utc_index': 'ts.data.temporal_guard',
    'assert_no_future_leakage': 'ts.data.temporal_guard',
    'assert_purged_train_horizon': 'ts.data.temporal_guard',
    'assert_strictly_before': 'ts.data.temporal_guard',
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value
