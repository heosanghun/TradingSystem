"""Four synchronized Binance spot closes, dated by the final hourly decision candle."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ts.config import PROJECT_ROOT

SYMBOLS = ("ETH", "BNB", "XRP", "ADA")


def read_closes(spec, manifest, cache):
    path = cache / spec["name"]
    if path.is_file():
        raw = path.read_bytes()
    else:
        url = f"https://raw.githubusercontent.com/{manifest['repository']}/{manifest['ref']}/{spec['path']}"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        raw = response.content
    blob = b"blob " + str(len(raw)).encode() + b"\0" + raw
    if hashlib.sha1(blob).hexdigest() != spec["sha"]:
        raise ValueError(f"Market source differs: {path.name}")
    if not path.is_file():
        cache.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    frame = pd.DataFrame(json.loads(raw))
    times = pd.DatetimeIndex(pd.to_datetime(frame[0], unit="ms", utc=True))
    if not times.equals(pd.date_range(times[0].normalize(), times[-1], freq="D")):
        raise ValueError("Missing or unordered daily market data")
    return pd.Series(frame[4].astype(float).to_numpy(), index=times + pd.Timedelta(hours=23))


def market_closes(clock):
    cache = PROJECT_ROOT / "data/cache"
    manifest = json.loads(Path(__file__).with_name("p07_market_sources.json").read_text())
    market, checks = {}, {}
    for symbol in SYMBOLS:
        path = cache / f"{symbol.lower()}usdt_1h_20200725_20231231.parquet"
        h = pd.read_parquet(path, columns=["close"]).close
        local = h[h.index.hour == 23]
        mirrored = pd.concat([read_closes(spec, manifest, cache / "p07_sources") for spec in manifest["files"]
                              if spec["name"].startswith(symbol + "_")])
        overlap = local.index.intersection(mirrored.index)
        if len(overlap) != 730 or not mirrored.index.is_unique:
            raise ValueError(f"Incomplete two-year overlap for {symbol}")
        np.testing.assert_array_equal(local.loc[overlap].values, mirrored.loc[overlap].values)
        market[symbol] = pd.concat([local, mirrored[mirrored.index > local.index[-1]]]).reindex(clock)
        checks[symbol] = len(overlap)
    frame = pd.DataFrame(market, index=clock)
    if not np.isfinite(frame).all().all() or (frame <= 0).any().any():
        raise ValueError("Missing decision-time market close")
    return frame, checks
