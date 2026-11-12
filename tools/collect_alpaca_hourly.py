"""
Download 1-hour bars for the US-listed ETFs of the Study 2 asset universe from Alpaca.

Output: one parquet per symbol in `data/cache/<symbol>_1h.parquet`, in the format
`MacroAssetFetcher` reads (`resolution="1h"`, `timestamp_semantics="bar_open"`), and a
summary in `data/macro_hourly_summary.json`.

Credentials are read from the environment or from a `.env` file in the project root
(which is git-ignored). Either naming is accepted:
    APCA_API_KEY_ID / APCA_API_SECRET_KEY
    ALPACA_API_KEY  / ALPACA_SECRET_KEY
The key is sent only to `data.alpaca.markets` and is never printed or written anywhere.

What the data is:
- Bars are requested with `adjustment=all` (splits and dividends), so hour-to-hour returns
  are not distorted on ex-dates.
- Alpaca labels each bar by its opening time and includes pre-market and after-hours bars
  when trades occurred. Bars are kept as delivered. Hours without trades have no bar.
- Yahoo-style tickers that are not exchange-traded products (`^VIX`, `^TNX`, `^IRX`,
  `EURUSD=X`, ...) are not available from this source and are listed as not requested.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

from ts.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, SystemConfig
from ts.data.macro_fetcher import MACRO_ETF_FX_40_SYMBOLS

BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
KEY_NAMES = (("APCA_API_KEY_ID", "APCA_API_SECRET_KEY"), ("ALPACA_API_KEY", "ALPACA_SECRET_KEY"))


def read_credentials(env_file: Path) -> Optional[Tuple[str, str]]:
    values: Dict[str, str] = dict(os.environ)
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            values.setdefault(name.strip().removeprefix("export ").strip(), value.strip().strip('"').strip("'"))
    for key_name, secret_name in KEY_NAMES:
        if values.get(key_name) and values.get(secret_name):
            return values[key_name], values[secret_name]
    return None


def fetch_symbol(session: requests.Session, symbol: str, start: str, end: str, feed: str) -> pd.DataFrame:
    rows: List[dict] = []
    token: Optional[str] = None
    while True:
        params = {"symbols": symbol, "timeframe": "1Hour", "start": start, "end": end,
                  "adjustment": "all", "feed": feed, "limit": 10000, "sort": "asc"}
        if token:
            params["page_token"] = token
        for attempt in range(5):
            resp = session.get(BARS_URL, params=params, timeout=60)
            if resp.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            break
        if resp.status_code != 200:
            # The response body may describe the problem; it never contains the credentials.
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        body = resp.json()
        rows.extend(body.get("bars", {}).get(symbol, []) or [])
        token = body.get("next_page_token")
        if not token:
            break
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low",
                                               "c": "close", "v": "volume", "n": "trade_count", "vw": "vwap"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    return frame[["open", "high", "low", "close", "volume", "trade_count", "vwap"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--env-file", type=str, default=str(PROJECT_ROOT / ".env"))
    parser.add_argument("--feed", type=str, default="sip", choices=["sip", "iex"])
    parser.add_argument("--symbols", type=str, default="", help="Comma-separated subset (default: all ETFs).")
    parser.add_argument("--force", action="store_true", help="Download again even if the archive exists.")
    args = parser.parse_args()

    creds = read_credentials(Path(args.env_file))
    if creds is None:
        raise SystemExit("No Alpaca credentials found. Put APCA_API_KEY_ID and APCA_API_SECRET_KEY "
                         f"in {args.env_file} or in the environment.")

    cfg = SystemConfig.from_json(str(DEFAULT_CONFIG_PATH))
    # One week before the warm-up start, matching the window MacroAssetFetcher asks for.
    start = (pd.Timestamp(cfg.data.warmup_start) - pd.Timedelta(days=75)).strftime("%Y-%m-%dT%H:%M:%SZ")
    end = pd.Timestamp(cfg.data.eval_end).strftime("%Y-%m-%dT%H:%M:%SZ")

    listed = [s for s in MACRO_ETF_FX_40_SYMBOLS if not s.startswith("^") and "=" not in s]
    not_requested = [s for s in MACRO_ETF_FX_40_SYMBOLS if s not in listed]
    if args.symbols:
        listed = [s for s in listed if s in set(args.symbols.split(","))]

    cache_dir = PROJECT_ROOT / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"APCA-API-KEY-ID": creds[0], "APCA-API-SECRET-KEY": creds[1]})

    summary = {"source": "Alpaca Market Data API v2, stock bars", "timeframe": "1Hour", "feed": args.feed,
               "adjustment": "all", "requested_start": start, "requested_end": end,
               "not_available_from_this_source": not_requested, "symbols": {}, "failed": {}}
    for symbol in listed:
        out = cache_dir / f"{symbol.lower()}_1h.parquet"
        if out.is_file() and not args.force:
            frame = pd.read_parquet(out)
        else:
            try:
                frame = fetch_symbol(session, symbol, start, end, args.feed)
            except Exception as exc:  # reported per symbol; nothing is substituted
                summary["failed"][symbol] = f"{type(exc).__name__}: {exc}"
                print(f"{symbol}: FAILED {type(exc).__name__}: {exc}")
                continue
            if frame.empty:
                summary["failed"][symbol] = "no bars returned"
                print(f"{symbol}: no bars returned")
                continue
            frame.attrs.update(resolution="1h", timestamp_semantics="bar_open",
                               source=f"Alpaca v2 stock bars, feed={args.feed}, adjustment=all")
            frame.to_parquet(out)
        per_day = frame.groupby(frame.index.tz_convert("America/New_York").date).size()
        summary["symbols"][symbol] = {
            "rows": int(len(frame)), "first": frame.index.min().isoformat(), "last": frame.index.max().isoformat(),
            "trading_days": int(len(per_day)), "median_bars_per_day": float(per_day.median()),
            "min_close": float(frame["close"].min()), "max_close": float(frame["close"].max()),
        }
        print(f"{symbol}: {len(frame)} bars, {frame.index.min():%Y-%m-%d} to {frame.index.max():%Y-%m-%d}, "
              f"{len(per_day)} days, median {per_day.median():.0f} bars per day")

    (PROJECT_ROOT / "data" / "macro_hourly_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"done: {len(summary['symbols'])} symbols saved, {len(summary['failed'])} failed, "
          f"{len(not_requested)} not available from this source")


if __name__ == "__main__":
    main()
