"""
Macro & Multi-Asset Universe Fetcher (~60 Assets + SPY/GLD Controls).

Implements Chapter IV Sec 2.1 & Chapter VI Sec 2:
- ~20 Major Cryptocurrencies from Binance 1h Spot
- ~40 Equity Indices, Sector ETFs, Treasury/Credit Bond ETFs, FX, Commodities,
  and Volatility proxies via Yahoo Finance (`yfinance`)
- Strict point-in-time causal alignment onto the 24/7 BTC 1h clock:
  Non-24/7 traditional assets only reveal bar closes at/before UTC hour `t`
  (between US market close and next open, log return is `0.0` — zero future leak).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

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
    """Fetches and causally aligns ~60 macro/crypto return series + SPY/GLD controls."""

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.binance = BinanceDataFetcher(cache_dir=self.cache_dir)

    def fetch_us_control_asset(
        self,
        ticker: str,
        start_utc: str = "2021-10-01T00:00:00Z",
        end_utc: str = "2023-12-31T23:00:00Z",
        prefer_hourly: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch SPY or GLD OHLCV for Table 6.2 comparison.
        Note: Yahoo Finance restricts `1h` history to the trailing 730 days; if the
        requested historical window (`2021–2023`) is older than 730 days and no local
        hourly CSV/Parquet archive exists in `cache_dir`, falls back to daily bars and
        tags `resolution='1d'` explicitly so `make_tables.py` & `ledger` record the exact bar count.
        """
        if self.cache_dir:
            local_1h = self.cache_dir / f"{ticker.lower()}_1h.parquet"
            if local_1h.exists():
                df = pd.read_parquet(local_1h)
                return self._slice_utc(df, start_utc, end_utc)

        start_dt = pd.Timestamp(start_utc).strftime("%Y-%m-%d")
        end_dt = (pd.Timestamp(end_utc) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        df_dl = pd.DataFrame()
        resolution = "1h"
        if prefer_hourly:
            try:
                df_dl = yf.download(ticker, start=start_dt, end=end_dt, interval="1h", progress=False, auto_adjust=True)
            except Exception:
                df_dl = pd.DataFrame()

        if df_dl is None or df_dl.empty:
            resolution = "1d"
            df_dl = yf.download(ticker, start=start_dt, end=end_dt, interval="1d", progress=False, auto_adjust=True)

        if df_dl is None or df_dl.empty:
            raise RuntimeError(f"Could not download {ticker} from Yahoo Finance.")

        if isinstance(df_dl.columns, pd.MultiIndex):
            df_dl.columns = [str(c[0]).lower() for c in df_dl.columns]
        else:
            df_dl.columns = [str(c).lower() for c in df_dl.columns]

        idx = pd.DatetimeIndex(df_dl.index)
        if idx.tz is None:
            # Daily US close occurs at 16:00 ET (21:00 UTC standard)
            idx = idx.tz_localize("UTC") + (pd.Timedelta(hours=21) if resolution == "1d" else pd.Timedelta(0))
        else:
            idx = idx.tz_convert("UTC")

        out = df_dl.copy()
        out.index = idx
        out = out[~out.index.duplicated(keep="last")].sort_index()
        out = out[[c for c in ("open", "high", "low", "close", "volume") if c in out.columns]].dropna(subset=["close"])
        out.attrs["resolution"] = resolution
        assert_monotonic_utc_index(out, name=f"control_{ticker}")
        return self._slice_utc(out, start_utc, end_utc)

    def build_aligned_60_asset_returns(
        self,
        btc_hourly_index: pd.DatetimeIndex,
        crypto_symbols: Sequence[str] = CRYPTO_20_SYMBOLS,
        macro_symbols: Sequence[str] = MACRO_ETF_FX_40_SYMBOLS,
    ) -> pd.DataFrame:
        """
        Constructs a `(T, |V|)` hourly log-return matrix causally aligned to `btc_hourly_index`.

        Causal alignment rule:
        - 24/7 Crypto: matched at exact UTC hour `t`.
        - Session-traded ETF/FX/Macro: forward-fill last known close strictly `<= t`,
          then compute `log(P_t / P_{t-1})`. Outside market hours `P_t == P_{t-1}`, so
          return is `0.0` without leaking next-day opens.
        """
        assert_monotonic_utc_index(pd.DataFrame(index=btc_hourly_index), name="btc_hourly_index")
        start_str = btc_hourly_index.min().isoformat()
        end_str = btc_hourly_index.max().isoformat()

        price_cols: Dict[str, pd.Series] = {}

        for sym in crypto_symbols:
            try:
                df_c, _ = self.binance.fetch_ohlcv(sym, "1h", start_str, end_str, realign_grid=False)
                price_cols[sym] = df_c["close"].reindex(btc_hourly_index).ffill()
            except Exception:
                continue

        start_yf = (btc_hourly_index.min() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        end_yf = (btc_hourly_index.max() + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        if macro_symbols:
            try:
                yf_df = yf.download(
                    list(macro_symbols),
                    start=start_yf,
                    end=end_yf,
                    interval="1d",
                    progress=False,
                    auto_adjust=True,
                )
                if yf_df is not None and not yf_df.empty:
                    closes = yf_df["Close"] if "Close" in yf_df else yf_df
                    if isinstance(closes, pd.Series):
                        closes = closes.to_frame(name=macro_symbols[0])
                    # Assign daily close to 21:00 UTC (after 16:00 ET close) to prevent intra-day look-ahead
                    c_idx = pd.DatetimeIndex(closes.index)
                    c_idx = (c_idx.tz_localize("UTC") if c_idx.tz is None else c_idx.tz_convert("UTC")) + pd.Timedelta(hours=21)
                    closes.index = c_idx
                    closes = closes[~closes.index.duplicated(keep="last")].sort_index()
                    union_idx = btc_hourly_index.union(closes.index).sort_values()
                    aligned_closes = closes.reindex(union_idx).ffill().reindex(btc_hourly_index)
                    for col in aligned_closes.columns:
                        s = aligned_closes[col].astype(np.float64)
                        if s.notna().sum() > len(btc_hourly_index) // 2:
                            price_cols[str(col)] = s
            except Exception:
                pass

        prices_df = pd.DataFrame(price_cols, index=btc_hourly_index).ffill()
        log_ret = np.log(prices_df / prices_df.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return log_ret

    @staticmethod
    def _slice_utc(df: pd.DataFrame, start_utc: str, end_utc: str) -> pd.DataFrame:
        s = pd.Timestamp(start_utc).tz_convert("UTC") if pd.Timestamp(start_utc).tzinfo else pd.Timestamp(start_utc, tz="UTC")
        e = pd.Timestamp(end_utc).tz_convert("UTC") if pd.Timestamp(end_utc).tzinfo else pd.Timestamp(end_utc, tz="UTC")
        sliced = df.loc[(df.index >= s) & (df.index <= e)].copy()
        sliced.attrs.update(df.attrs)
        return sliced
