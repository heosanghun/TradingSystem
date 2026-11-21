"""Frozen P61 entry rule on earlier Bitstamp BTC/USD, without a future-trained student."""
import hashlib
import io
import json

import numpy as np
import pandas as pd
import requests

from ts.baselines.audit_momentum_condition import statistics
from ts.baselines.early_breakout_audit import policy_inputs
from ts.baselines.trend_entry_student import OUT, day_volume, entry_signals
from ts.baselines.weekly_student import decisions_at
from ts.config import PROJECT_ROOT
from ts.eval.backtester import CausalPortfolioBacktester

BLOB = "1f8b736982dda4d0f4e3edffd5fc405355d538e3"
REPO = "Julian-Andres-Thomas/BTC-USD"
SOURCE_PATH = "btc_1h_cleaned.csv"
SUBSET_SHA = "d920d0a6392af73db6b9abe750134a8c6a3bdd37a29d8f369d4bed47724f89db"


def load_subset():
    path = PROJECT_ROOT / "data/cache/p61_sources/bitstamp_201501_201707.csv"
    if path.exists():
        raw = path.read_bytes().replace(b"\r\n", b"\n")
    else:
        response = requests.get(f"https://api.github.com/repos/{REPO}/git/blobs/{BLOB}",
                                headers={"Accept": "application/vnd.github.raw+json"}, timeout=60)
        response.raise_for_status()
        blob = response.content
        if hashlib.sha1(b"blob " + str(len(blob)).encode() + b"\0" + blob).hexdigest() != BLOB:
            raise ValueError("Historical data source differs")
        lines = blob.splitlines()
        raw = b"\n".join([lines[0], *(row for row in lines[1:] if b"2015-01-01" <= row[:10] <= b"2017-07-31")]) + b"\n"
    if hashlib.sha256(raw).hexdigest() != SUBSET_SHA:
        raise ValueError("Unexpected Bitstamp hourly subset")
    return pd.read_csv(io.BytesIO(raw), index_col=0, parse_dates=True).rename(columns=str.lower)


def main():
    bars = load_subset()
    if not bars.index.equals(pd.date_range("2015-01-01", "2017-07-31 23:00", tz="UTC", freq="h")):
        raise ValueError("Bitstamp hourly clock differs")
    if (not np.isfinite(bars).all().all() or bars.volume.lt(0).any() or bars.low.le(0).any()
            or bars.high.lt(bars[["open", "close", "low"]].max(axis=1)).any()
            or bars.low.gt(bars[["open", "close", "high"]].min(axis=1)).any()):
        raise ValueError("Invalid historical market data")
    close = bars.close[bars.index.hour == 23]
    gate = entry_signals(close, day_volume(bars)).P61
    sample = bars.loc["2015-08":]
    if not sample.volume.gt(0).all():
        raise ValueError("Untraded hour in evaluation")
    clock = sample.index[sample.index.hour == 23]
    budget = policy_inputs(bars).budget.loc[clock]
    if budget.isna().any():
        raise ValueError("Missing risk budget")
    rule = decisions_at(sample.index, clock, budget.where(gate.loc[clock], 0.))
    report = {"source_repo": REPO, "path": SOURCE_PATH, "git_blob": BLOB,
              "subset_sha256": SUBSET_SHA, "start": str(sample.index[0]), "end": str(sample.index[-1]),
              "hours": len(sample), "warmup_zero_volume_hours": int(bars.loc[:"2015-07"].volume.eq(0).sum())}
    for name, weights in (("P61_rule", rule), ("buy_hold", np.ones(len(sample)))):
        sim = CausalPortfolioBacktester().run_decisions(sample, weights, phase=2)
        report[name] = statistics(sim.bar_net_returns)
    print(json.dumps(report, indent=2))
    path = OUT / "P61_seed42" / "bitstamp_rule.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
