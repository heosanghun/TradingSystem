"""Hourly market archive loading, causal alignment and universe diagnostics."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import logging

LOGGER = logging.getLogger(__name__)

from ts.data.binance_fetcher import BinanceDataFetcher
from ts.data.temporal_guard import assert_monotonic_utc_index


CRYPTO_20_SYMBOLS: Tuple[str, ...] = (
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "DOTUSDT", "LTCUSDT", "LINKUSDT",
    "UNIUSDT", "AVAXUSDT", "ATOMUSDT", "TRXUSDT", "ETCUSDT",
    "XLMUSDT", "FILUSDT", "AAVEUSDT", "EOSUSDT", "BCHUSDT",
)

MACRO_ETF_FX_40_SYMBOLS: Tuple[str, ...] = (
    # US & Global Equity Indices (6)
    "SPY", "QQQ", "DIA", "IWM", "EFA", "EEM",
    # US Sector & Thematic ETFs (14)
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI",
    "XLB", "XLU", "XLRE", "XLC", "SMH", "KRE", "ARKK",
    # Treasury Yields & Credit Spreads via Liquid ETFs + Yield Indices (8)
    "TLT", "IEF", "SHY", "TIP", "HYG", "LQD", "^TNX", "^IRX",
    # Commodities (5)
    "GLD", "SLV", "USO", "UNG", "DBC",
    # FX & Dollar Index (5)
    "UUP", "EURUSD=X", "JPY=X", "GBPUSD=X", "AUDUSD=X",
    # Volatility Indices / Proxies (2)
    "^VIX", "VIXY",
)

DEFAULT_60_ASSET_UNIVERSE: List[str] = list(CRYPTO_20_SYMBOLS) + list(MACRO_ETF_FX_40_SYMBOLS)


class MacroAssetFetcher:
    """Align actual hourly archives on their information-availability timestamps.

    Missing symbols are reported explicitly. Daily closes are never substituted
    for hourly macro observations. Local archives declare resolution and whether
    their clock labels bar opens or closes.
    """

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.binance = BinanceDataFetcher(cache_dir=self.cache_dir)
        self.last_universe_report = {}

    @staticmethod
    def _hourly_archive(path: Path) -> pd.DataFrame:
        frame = pd.read_parquet(path)
        assert_monotonic_utc_index(frame, name=str(path))
        if frame.attrs.get("resolution") != "1h":
            raise ValueError(f"{path}: resolution='1h' metadata is required")
        semantics = frame.attrs.get("timestamp_semantics")
        if semantics not in ("bar_open", "bar_close"):
            raise ValueError(f"{path}: timestamp_semantics must be bar_open or bar_close")
        if len(frame) < 2 or "close" not in frame:
            raise ValueError(f"{path}: at least two hourly close observations are required")
        deltas = frame.index.to_series().diff().dropna()
        if (deltas < pd.Timedelta(hours=1)).any() or not (deltas == pd.Timedelta(hours=1)).any():
            raise ValueError(f"{path}: timestamps do not describe an hourly series")
        close = frame["close"].to_numpy(dtype=float)
        if not np.isfinite(close).all() or (close <= 0).any():
            raise ValueError(f"{path}: closes must be positive and finite")
        frame = frame.copy()
        if semantics == "bar_open":
            frame.index = frame.index + pd.Timedelta(hours=1)
        frame.index = frame.index.tz_convert("UTC")
        frame.attrs["timestamp_semantics"] = "bar_close"
        return frame

    def fetch_us_control_asset(self, ticker: str,
                               start_utc: str = "2021-10-01T00:00:00Z",
                               end_utc: str = "2023-12-31T23:00:00Z",
                               prefer_hourly: bool = True) -> pd.DataFrame:
        """Read an hourly archive, or daily data only when explicitly requested.

        Hourly mode raises if the archive is absent. An explicitly daily control
        comparison is a different experiment and must be labelled as such.
        """
        local = self.cache_dir / f"{ticker.lower()}_1h.parquet" if self.cache_dir else None
        if prefer_hourly:
            if local is None or not local.is_file():
                raise FileNotFoundError(f"Hourly {ticker} archive missing; daily fallback is disabled")
            return self._slice_utc(self._hourly_archive(local), start_utc, end_utc)
        LOGGER.warning("Explicit daily control requested: %s; do not mix with hourly statistics", ticker)
        start_dt = pd.Timestamp(start_utc).strftime("%Y-%m-%d")
        end_dt = (pd.Timestamp(end_utc) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        frame = yf.download(ticker, start=start_dt, end=end_dt, interval="1d", progress=False, auto_adjust=True)
        if frame is None or frame.empty:
            raise RuntimeError(f"No daily observations returned for {ticker}")
        frame.columns = [str(c[0] if isinstance(c, tuple) else c).lower() for c in frame.columns]
        # 16:00 New York civil time, including DST, is an availability bound for
        # regular-session US ETF daily closes; this path is not used for hourly TE.
        dates = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
        frame.index = (dates + pd.Timedelta(hours=16)).tz_localize("America/New_York").tz_convert("UTC")
        frame.attrs.update(resolution="1d", timestamp_semantics="bar_close")
        return self._slice_utc(frame, start_utc, end_utc)

    def build_aligned_60_asset_returns(self, btc_hourly_index: pd.DatetimeIndex,
                                      crypto_symbols: Sequence[str] = CRYPTO_20_SYMBOLS,
                                      macro_symbols: Sequence[str] = MACRO_ETF_FX_40_SYMBOLS) -> pd.DataFrame:
        """Return hourly log returns, with a transparent included/excluded universe.

        Index t denotes information available by t. Cryptocurrency kline opens
        are shifted one hour before alignment. Nontrading gaps carry the last
        observed close forward; no backfill or daily-to-hourly interpolation.
        Leading missing observations exclude the symbol instead of inventing data.
        This does not distinguish every vendor outage from a scheduled closure;
        source coverage must still be audited against a trading calendar.
        """
        assert_monotonic_utc_index(pd.DataFrame(index=btc_hourly_index), name="btc_hourly_index")
        if btc_hourly_index.empty:
            raise ValueError("The alignment clock is empty")
        report = {"requested": list(crypto_symbols) + list(macro_symbols), "included": [], "excluded": []}
        self.last_universe_report = report
        columns = {}
        start = (btc_hourly_index.min() - pd.Timedelta(days=7)).isoformat()
        end = btc_hourly_index.max().isoformat()
        for symbol in report["requested"]:
            try:
                if symbol in crypto_symbols:
                    frame, gaps = self.binance.fetch_ohlcv(symbol, "1h", start, end, realign_grid=False)
                    prices = frame["close"].copy()
                    prices.index = prices.index + pd.Timedelta(hours=1)
                    source = "Binance spot 1h klines"
                else:
                    archive = self.cache_dir / f"{symbol.lower()}_1h.parquet" if self.cache_dir else None
                    if archive is None or not archive.is_file():
                        raise FileNotFoundError("An actual hourly local archive is required")
                    frame = self._hourly_archive(archive)
                    prices = frame["close"]
                    gaps = None
                    source = frame.attrs.get("source", "local hourly archive; provider unverified")
                observed_on_clock = prices.reindex(btc_hourly_index)
                union = prices.index.union(btc_hourly_index).sort_values()
                aligned = prices.reindex(union).ffill().reindex(btc_hourly_index)
                if aligned.isna().any():
                    raise ValueError("Archive lacks a known price at the start of the requested clock")
                age = btc_hourly_index[-1] - prices.index[prices.index <= btc_hourly_index[-1]][-1]
                if age > pd.Timedelta(days=7):
                    raise ValueError("Archive ends more than seven days before the requested end")
                columns[symbol] = aligned
                report["included"].append(dict(symbol=symbol, source=source, resolution="1h",
                    source_rows=len(prices), source_start=prices.index.min().isoformat(),
                    source_end=prices.index.max().isoformat(), gap_report=gaps,
                    missing_exact_clock_fraction=float(observed_on_clock.isna().mean())))
            except (RuntimeError, ValueError, OSError, KeyError) as exc:
                LOGGER.warning("Excluding hourly asset %s: %s", symbol, exc)
                report["excluded"].append(dict(symbol=symbol, error=type(exc).__name__, reason=str(exc)))
        if len(columns) < 2:
            raise RuntimeError(f"Only {len(columns)} valid hourly assets; inspect last_universe_report")
        prices = pd.DataFrame(columns, index=btc_hourly_index)
        result = np.log(prices / prices.shift(1)).fillna(0.)
        result.attrs["universe"] = report
        return result

    @staticmethod
    def _slice_utc(df: pd.DataFrame, start_utc: str, end_utc: str) -> pd.DataFrame:
        start = pd.Timestamp(start_utc)
        end = pd.Timestamp(end_utc)
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        sliced = df.loc[(df.index >= start) & (df.index <= end)].copy()
        sliced.attrs.update(df.attrs)
        if sliced.empty:
            raise ValueError("No observations in the requested period")
        return sliced
