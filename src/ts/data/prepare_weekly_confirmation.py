"""Build the fixed 34-month Binance spot OHLCV input and verify the joins.

If cached mirrors are missing, retrieve the exact blob recorded in the manifest.
Run from any directory with the project virtual environment.
"""
import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ts.config import PROJECT_ROOT

ROOT = PROJECT_ROOT / "data"
CACHE = ROOT / "cache/p01_sources"
FIELDS = ["open", "high", "low", "close", "volume"]


def load_file(item):
    path = CACHE / item["file"]
    if path.is_file():
        raw = path.read_bytes()
    else:
        url = f"https://api.github.com/repos/{item['repository']}/git/blobs/{item['blob_sha']}"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        raw = base64.b64decode(response.json()["content"])
    blob = b"blob " + str(len(raw)).encode() + b"\x00" + raw
    if hashlib.sha1(blob).hexdigest() != item["blob_sha"]:
        raise ValueError(f"Source hash mismatch: {path}")
    if not path.exists():
        path.write_bytes(raw)
    if item["file"] in {"middle.csv", "late.csv"}:
        frame = pd.read_csv(path)
        frame.index = pd.to_datetime(frame.pop("timestamp"), utc=True)
    else:
        frame = pd.read_csv(path, header=None, usecols=range(6), names=["timestamp", *FIELDS])
        frame.index = pd.to_datetime(frame.pop("timestamp"), unit="ms", utc=True)
    frame = frame[FIELDS].astype(float)
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError(f"Invalid clock: {path}")
    return frame


def compare(left, right):
    index = left.index.intersection(right.index)
    if not len(index):
        raise ValueError("No timestamps in common between adjacent data sources")
    np.testing.assert_allclose(left.loc[index, FIELDS], right.loc[index, FIELDS], rtol=1e-10, atol=1e-8)
    return len(index)


def main():
    CACHE.mkdir(exist_ok=True, parents=True)
    specs = json.loads(Path(__file__).with_name("p01_hourly_sources.json").read_text(encoding="utf-8"))
    data = {item["file"]: load_file(item) for item in specs}
    previous = pd.read_parquet(ROOT / "cache/btcusdt_1h_20200801_20231231.parquet")
    early = pd.concat([data[key] for key in sorted(data) if key.startswith("2024")])
    middle, late = data["middle.csv"], data["late.csv"]
    # The final candle of a live API download can still be forming.
    middle_closed = middle.iloc[:-1]
    overlap = {"historical_december_2023": compare(previous, data["2023-12.csv"]),
               "early_middle": compare(early, middle_closed), "middle_late": compare(middle_closed, late)}
    bars = pd.concat([early, middle.loc["2025-01-01":"2026-04-30"], late.loc["2026-05-01":"2026-10-31"]])
    expected = pd.date_range("2024-01-01", "2026-10-31 23:00", freq="h", tz="UTC")
    if not bars.index.equals(expected) or not np.isfinite(bars).all().all():
        raise ValueError("Incomplete or invalid confirmation window")
    if ((bars[FIELDS[:4]] <= 0).any().any() or (bars.volume < 0).any()
        or (bars.high < bars[["open","close","low"]].max(axis=1)).any()
        or (bars.low > bars[["open","close","high"]].min(axis=1)).any()):
        raise ValueError("Invalid OHLCV")
    path = CACHE / "confirmed_ohlcv.parquet"
    bars.to_parquet(path)
    print(json.dumps({"overlap_bars_matching": overlap, "bars": len(bars), "first": str(bars.index[0]), "last": str(bars.index[-1])}))


if __name__ == "__main__":
    main()
