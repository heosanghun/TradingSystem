"""Gate a proposed spot teacher on an earlier non-overlapping 27-month history."""
import hashlib
import io
import json

import numpy as np
import pandas as pd
import requests

from ts.baselines.breakout_student import TARGET, daily_inputs
from ts.baselines.weekly_student import decisions_at
from ts.config import PROJECT_ROOT
from ts.eval.backtester import CausalPortfolioBacktester

SHA = "71d752e1a5fbac004c089576b5b9608f3d508aac"
REF = "58b54c115ee12efc48376831b5def3e272057baf"


def historical_prices():
    cache = PROJECT_ROOT / "data/cache"
    file = cache / "p05_sources/BTCUSDT_1h.csv"
    if file.exists():
        raw = file.read_bytes()
    else:
        url = f"https://raw.githubusercontent.com/Seggov/seasonal-market-dashboard/{REF}/data/BTCUSDT_1h.csv"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        raw = response.content
    blob = b"blob " + str(len(raw)).encode() + b"\0" + raw
    if hashlib.sha1(blob).hexdigest() != SHA:
        raise ValueError("Unexpected historical source hash")
    if not file.exists():
        file.parent.mkdir(exist_ok=True, parents=True)
        file.write_bytes(raw)
    bars = pd.read_csv(io.BytesIO(raw), index_col="timestamp_utc", parse_dates=True).loc[:"2020-08-31"]
    original = pd.read_parquet(cache / "btcusdt_1h_20200801_20231231.parquet").loc["2020-08"]
    overlap = bars.loc["2020-08"]
    if not bars.index.is_unique or not bars.index.is_monotonic_increasing:
        raise ValueError("Unordered historical prices")
    if len(overlap) != 744 or not original.index.equals(overlap.index):
        raise ValueError("Incomplete intersection")
    np.testing.assert_array_equal(original[bars.columns].values, overlap.values)
    if (not np.isfinite(bars).all().all() or (bars.volume < 0).any() or (bars.low <= 0).any()
            or (bars.high < bars[["open", "close", "low"]].max(axis=1)).any()
            or (bars.low > bars[["open", "close", "high"]].min(axis=1)).any()):
        raise ValueError("Invalid historical OHLCV")
    return bars.loc[:"2020-07-31"]


def policy_inputs(bars):
    close = bars.close[bars.index.hour == 23].asfreq("D")
    votes = []
    for entry_days, exit_days in ((20, 10), (55, 20)):
        previous = close.shift(1)
        state = pd.Series(np.nan, index=close.index)
        state.loc[close > previous.rolling(entry_days).max()] = 1
        state.loc[close < previous.rolling(exit_days).min()] = 0
        votes.append(state.ffill().fillna(0))
    return pd.DataFrame({"fast": votes[0], "slow": votes[1], "budget": daily_inputs(bars)[0].budget})


def risk_policy(bars):
    features = policy_inputs(bars)
    return (features.fast + features.slow) / 2 * features.budget


def main():
    prices = historical_prices()
    paper = prices.loc["2018-05-01":"2020-07-31"]
    expected = pd.date_range("2018-05-01", "2020-07-31 23:00", freq="h", tz="UTC")
    if len(paper.index.difference(expected)) or paper.index[-1] != expected[-1]:
        raise ValueError("Unexpected historical clock")
    daily = risk_policy(prices).reindex(paper.index[paper.index.hour == 23]).dropna()
    policies = {"P05": decisions_at(paper.index, daily.index, daily), "always_long": np.ones(len(paper))}
    metrics, returns = {}, {}
    for name, weights in policies.items():
        simulation = CausalPortfolioBacktester().run_decisions(paper, weights, phase=2)
        metrics[name] = simulation.summary.to_dict()
        returns[name] = simulation.bar_net_returns
    out = PROJECT_ROOT / "runs/baselines/early_breakout"
    out.mkdir(parents=True, exist_ok=True)
    miss = expected.difference(paper.index)
    result = dict(source=REF, blob_sha=SHA, overlap=744, missing_hours=len(miss),
                  daily_signals=len(daily), first=str(paper.index[0]), last=str(paper.index[-1]),
                  pass_all=all(metrics["P05"][k] >= v for k, v in TARGET.items()), results=metrics)
    (out / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(returns, index=paper.index).to_parquet(out / "returns.parquet")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
