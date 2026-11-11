"""
Binance Spot OHLCV & AggTrades Historical Fetcher (`data-api.binance.vision` / `api.binance.com`).

Downloads public historical 1-hour (and optional 1-minute) klines from
`2020-08-01` through `2023-12-31` (covering the 60-day hypergraph warmup,
Walk-Forward Fold 1 training window `2020.10~2021.09`, and the 27-month
evaluation period `2021.10~2023.12`), recording exact exchange gap counts.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from ts.data.temporal_guard import assert_monotonic_utc_index


BINANCE_ENDPOINTS = (
    "https://data-api.binance.vision/api/v3/klines",
    "https://api.binance.com/api/v3/klines",
    "https://api1.binance.com/api/v3/klines",
)


class BinanceDataFetcher:
    """Public historical kline downloader with causal gap tracking."""

    def __init__(self, cache_dir: Optional[Path] = None, timeout_sec: int = 20) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_sec = timeout_sec
        self.session = requests.Session()

    def _request_klines(
        self, symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = 1000
    ) -> List[List]:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": limit,
        }
        last_err: Optional[Exception] = None
        for url in BINANCE_ENDPOINTS:
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout_sec)
                if resp.status_code == 200:
                    return resp.json()
            except Exception as exc:
                last_err = exc
        raise RuntimeError(f"Failed to fetch Binance klines for {symbol}: {last_err}")

    def fetch_ohlcv(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        start_utc: str = "2020-08-01T00:00:00Z",
        end_utc: str = "2023-12-31T23:00:00Z",
        realign_grid: bool = False,
        use_cache: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Fetch spot OHLCV bars and return `(df, gap_report)`.

        Note: By default `realign_grid=False` preserves exact observed exchange bars
        (e.g., ~19,704 bars in 2021.10–2023.12 due to Binance maintenance outages),
        while `gap_report` explicitly records `expected_grid_bars`, `observed_bars`,
        and `missing_bars` to resolve Thesis Issue #5 (`822*24=19,728` vs observed).
        If `realign_grid=True`, gaps are causally forward-filled with `is_gap_filled=1`.
        """
        start_ts = pd.Timestamp(start_utc).tz_convert("UTC") if pd.Timestamp(start_utc).tzinfo else pd.Timestamp(start_utc, tz="UTC")
        end_ts = pd.Timestamp(end_utc).tz_convert("UTC") if pd.Timestamp(end_utc).tzinfo else pd.Timestamp(end_utc, tz="UTC")

        cache_file: Optional[Path] = None
        if self.cache_dir is not None:
            safe_start = start_ts.strftime("%Y%m%d")
            safe_end = end_ts.strftime("%Y%m%d")
            cache_file = self.cache_dir / f"{symbol.lower()}_{interval}_{safe_start}_{safe_end}.parquet"
            if use_cache and cache_file.exists():
                df_cached = pd.read_parquet(cache_file)
                return self._finalize_dataframe(df_cached, start_ts, end_ts, interval, realign_grid)

        cur_ms = int(start_ts.timestamp() * 1000)
        end_ms = int(end_ts.timestamp() * 1000)
        step_ms = self._interval_to_ms(interval)
        rows: List[List] = []

        while cur_ms <= end_ms:
            batch = self._request_klines(symbol, interval, cur_ms, end_ms, limit=1000)
            if not batch:
                break
            rows.extend(batch)
            last_open_ms = int(batch[-1][0])
            next_ms = last_open_ms + step_ms
            if next_ms <= cur_ms:
                break
            cur_ms = next_ms
            if len(batch) < 1000:
                break
            time.sleep(0.05)

        if not rows:
            raise RuntimeError(f"No bars returned from Binance for {symbol} [{start_ts}..{end_ts}]")

        cols = [
            "open_time_ms",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time_ms",
            "quote_volume",
            "num_trades",
            "taker_buy_base_vol",
            "taker_buy_quote_vol",
            "ignore",
        ]
        raw_df = pd.DataFrame(rows, columns=cols)
        raw_df["timestamp"] = pd.to_datetime(raw_df["open_time_ms"].astype("int64"), unit="ms", utc=True)
        for c in ("open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base_vol", "taker_buy_quote_vol"):
            raw_df[c] = raw_df[c].astype(np.float64)
        raw_df["num_trades"] = raw_df["num_trades"].astype(np.int64)

        df = (
            raw_df.drop_duplicates(subset=["timestamp"], keep="last")
            .set_index("timestamp")
            .sort_index()[
                [
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "quote_volume",
                    "num_trades",
                    "taker_buy_base_vol",
                    "taker_buy_quote_vol",
                ]
            ]
        )
        df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)].copy()

        if cache_file is not None:
            df.to_parquet(cache_file)

        return self._finalize_dataframe(df, start_ts, end_ts, interval, realign_grid)

    @staticmethod
    def _interval_to_ms(interval: str) -> int:
        if interval.endswith("m"):
            return int(interval[:-1]) * 60_000
        if interval.endswith("h"):
            return int(interval[:-1]) * 3_600_000
        if interval.endswith("d"):
            return int(interval[:-1]) * 86_400_000
        raise ValueError(f"Unsupported interval: {interval}")

    @staticmethod
    def _interval_to_pandas_freq(interval: str) -> str:
        if interval.endswith("m"):
            return f"{int(interval[:-1])}min"
        if interval.endswith("h"):
            return f"{int(interval[:-1])}h"
        if interval.endswith("d"):
            return f"{int(interval[:-1])}D"
        raise ValueError(f"Unsupported interval: {interval}")

    def _finalize_dataframe(
        self,
        df: pd.DataFrame,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        interval: str,
        realign_grid: bool,
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        freq = self._interval_to_pandas_freq(interval)
        expected_idx = pd.date_range(start=start_ts, end=end_ts, freq=freq, tz="UTC")
        observed_bars = int(len(df))
        expected_bars = int(len(expected_idx))
        missing_bars = int(len(expected_idx.difference(df.index)))

        if realign_grid:
            reindexed = df.reindex(expected_idx)
            is_gap = reindexed["close"].isna().astype(np.int8)
            reindexed["close"] = reindexed["close"].ffill()
            for col in ("open", "high", "low"):
                reindexed[col] = reindexed[col].fillna(reindexed["close"])
            for col in ("volume", "quote_volume", "taker_buy_base_vol", "taker_buy_quote_vol"):
                if col in reindexed.columns:
                    reindexed[col] = reindexed[col].fillna(0.0)
            if "num_trades" in reindexed.columns:
                reindexed["num_trades"] = reindexed["num_trades"].fillna(0).astype(np.int64)
            reindexed["is_gap_filled"] = is_gap
            out = reindexed.dropna(subset=["close"]).copy()
        else:
            out = df.copy()
            out["is_gap_filled"] = np.int8(0)

        assert_monotonic_utc_index(out, name="binance_ohlcv")
        report = {
            "expected_grid_bars": expected_bars,
            "observed_bars": observed_bars,
            "missing_bars": missing_bars,
        }
        return out, report
