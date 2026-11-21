"""Confirm P-01 on the complete fixed period with the frozen 2023 model."""
import json

import numpy as np
import pandas as pd
import torch

from ts.baselines.weekly_student import OUT, decision_windows, decisions_at, digest, weekly_inputs
from ts.config import PROJECT_ROOT
from ts.eval.backtester import CausalPortfolioBacktester
from ts.study1.ppo_expert import ACTION_WEIGHTS
from ts.study3.student_model import Student1DCNN


def paired_sharpe_interval(returns, benchmark):
    """Percentile CI of annualized Sharpe difference, paired circular 7-day blocks."""
    r = np.column_stack((returns, benchmark))
    n = len(r)
    block = 168
    steps, tail = divmod(n, block)
    moments = np.column_stack((r, r*r))
    circular = np.vstack((moments, moments[:block]))
    sums = np.vstack((np.zeros(4), np.cumsum(circular, axis=0)))
    idx = np.arange(n)
    blocks = sums[idx + block] - sums[idx]
    tails = sums[idx + tail] - sums[idx]
    rng = np.random.default_rng(42)
    differences = []
    for _ in range(50):
        drawn = rng.integers(n, size=(100, steps))
        total = blocks[drawn].sum(axis=1) + tails[rng.integers(n, size=100)]
        mean = total[:, :2] / n
        sd = np.sqrt(np.maximum((total[:, 2:] - n*mean**2) / (n-1), 0))
        sr = np.divide(mean, sd, out=np.zeros_like(mean), where=sd > 1e-12) * np.sqrt(8760)
        differences.extend(sr[:, 0] - sr[:, 1])
    return np.percentile(differences, [2.5, 97.5]).tolist()


def main():
    torch.set_num_threads(2)
    saved = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
    weights_file = OUT / "frozen_student.pt"
    model_hash = digest(weights_file)
    if not saved["eligible_for_confirmation"] or model_hash != saved["frozen_model"]["sha256"]:
        raise ValueError("Development gate or frozen model identity failed")
    base = PROJECT_ROOT / "data/cache/btcusdt_1h_20200801_20231231.parquet"
    input_file = PROJECT_ROOT / "data/cache/p01_sources/confirmed_ohlcv.parquet"
    live = pd.read_parquet(input_file)
    clock = pd.date_range("2024-01-01", "2026-10-31 23:00", freq="h", tz="UTC")
    if not clock.equals(live.index):
        raise ValueError("Expected the complete 34-month confirmation window")
    past = pd.read_parquet(base)[live.columns]
    if digest(base) != saved["sha256"][base.name] or past.index[-1] >= clock[0]:
        raise ValueError("Invalid development archive")
    prices = pd.concat([past, live])
    features, _ = weekly_inputs(prices)
    days, x = decision_windows(features)
    times = prices.index[days]
    pos = np.flatnonzero((times >= clock[0]) & (times.day_of_week == 6))
    times = times[pos]
    model = Student1DCNN(input_dim=x.shape[1])
    model.load_state_dict(torch.load(weights_file, map_location="cpu", weights_only=True))
    with torch.inference_mode():
        acts, _ = model.eval().act_online(torch.from_numpy(x[pos]))
    sundays = clock[(clock.day_of_week == 6) & (clock.hour == 23)]
    if not times.equals(sundays):
        raise ValueError("Missing confirmation decisions")
    lag = prices.close.reindex(sundays - pd.Timedelta(days=30)).to_numpy()
    trend = (prices.close.loc[sundays].to_numpy() > lag).astype(float)
    policies = {"weekly_student": decisions_at(clock, times, ACTION_WEIGHTS[acts.numpy()]),
                "always_long": np.ones(len(clock)),
                "weekly_30d_trend": decisions_at(clock, sundays, trend)}
    bt = CausalPortfolioBacktester()
    metrics, net = {}, {}
    for name, policy in policies.items():
        result = bt.run_decisions(live, policy, phase=2)
        metrics[name] = {**result.summary.to_dict(), "cash_fraction": float(np.mean(result.weights == 0)),
                         "position_changes": int(np.count_nonzero(np.diff(result.weights, prepend=0)))}
        net[name] = result.bar_net_returns
    dev = pd.read_parquet(OUT / "hourly_returns.parquet")
    intervals = {phase: {name: paired_sharpe_interval(frame["weekly_student"], frame[name])
                          for name in frame if name != "weekly_student"}
                 for phase, frame in (("development", dev), ("confirmation", net))}
    payload = {"stage": "fixed_model_confirmation", "frozen_weights_sha256": model_hash,
               "data_sha256": digest(input_file), "start": str(clock[0]), "end": str(clock[-1]),
               "strategies": metrics, "paired_sharpe_difference_95pct_weekly_blocks": intervals}
    target = OUT / "confirmation.json"
    if target.exists() and json.loads(target.read_text()) != payload:
        raise RuntimeError("A different confirmation is already recorded")
    pd.DataFrame(net, index=clock).to_parquet(OUT / "confirmation_hourly_returns.parquet")
    pd.DataFrame(policies, index=clock).to_parquet(OUT / "confirmation_decisions.parquet")
    target.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
