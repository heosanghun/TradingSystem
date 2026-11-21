"""Verify the first 20 complete November days, continuing the frozen portfolios."""
import hashlib
import json

import numpy as np
import pandas as pd
import requests
import torch

from ts.baselines.breakout_student import OUT as BREAKOUT, daily_inputs, predict, windows
from ts.baselines.early_breakout_audit import policy_inputs
from ts.baselines.panel_student import OUT as PANEL
from ts.baselines.risk_student import OUT as RISK, RiskStudent
from ts.baselines.weekly_student import OUT as WEEKLY, decision_windows, decisions_at, digest, weekly_inputs
from ts.config import PROJECT_ROOT
from ts.eval.backtester import CausalPortfolioBacktester
from ts.eval.metrics import compute_performance_summary
from ts.study1.ppo_expert import ACTION_WEIGHTS
from ts.study3.student_model import Student1DCNN

REF = "6ba2f47c88055567183b2abdf2fb8ba6bb502869"
BLOBS = {"10": "3c8587aafa2160b0da9deb39b6c32355ad5663f8",
         "11": "e6e0d45162f6d2644219cfb413252bc81ee8c6fa"}
FIELDS = ["open", "high", "low", "close", "volume"]


def mirror_month(month):
    cache = PROJECT_ROOT / "data/cache/p02_sources"
    cached = cache / f"{month}.json"
    if cached.is_file():
        raw = cached.read_bytes()
    else:
        url = f"https://raw.githubusercontent.com/finom/static-klines/{REF}/.klines-cache/BTCUSDT/1h/2026-{month}-01.json"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        raw = response.content
    blob = b"blob " + str(len(raw)).encode() + b"\x00" + raw
    if hashlib.sha1(blob).hexdigest() != BLOBS[month]:
        raise ValueError("Unexpected mirror contents")
    if not cached.exists():
        cache.mkdir(exist_ok=True, parents=True)
        cached.write_bytes(raw)
    rows = pd.DataFrame(json.loads(raw))
    data = rows.loc[:, 1:5].astype(float)
    data.columns = FIELDS
    data.index = pd.to_datetime(rows[0], unit="ms", utc=True)
    data.index.name = "timestamp"
    if not data.index.is_unique or not data.index.is_monotonic_increasing:
        raise ValueError("Incorrect mirror timestamp order")
    return data


def load_student(path, sha, channels, model_class=Student1DCNN):
    if digest(path) != sha:
        raise ValueError(f"Frozen student weights differ at {path}")
    model = model_class(input_dim=channels)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()


def main():
    torch.set_num_threads(2)
    cutoff = pd.Timestamp("2026-11-01", tz="UTC")
    end = pd.Timestamp("2026-11-20 23:00", tz="UTC")
    follow = pd.read_parquet(PROJECT_ROOT / "data/cache/p01_sources/confirmed_ohlcv.parquet")
    overlap = follow.loc["2026-10"]
    oct_bars = mirror_month("10")
    if len(overlap) != 744 or not overlap.index.equals(oct_bars.index):
        raise ValueError("Incomplete October overlap")
    np.testing.assert_array_equal(overlap[FIELDS].values, oct_bars.values)
    nov = mirror_month("11").loc[cutoff:end]
    if not nov.index.equals(pd.date_range(cutoff, end, freq="h")):
        raise ValueError("Expected 480 consecutive hourly candles")
    if (not np.isfinite(nov).all().all() or (nov[FIELDS[:4]] <= 0).any().any() or (nov.volume < 0).any()
            or (nov.high < nov[["open", "close", "low"]].max(axis=1)).any()
            or (nov.low > nov[["open", "close", "high"]].min(axis=1)).any()):
        raise ValueError("Invalid OHLCV")
    bars = pd.concat([follow, nov])
    hist = pd.read_parquet(PROJECT_ROOT / "data/cache/btcusdt_1h_20200801_20231231.parquet")[FIELDS]
    prices = pd.concat([hist, bars])
    d, x1 = decision_windows(weekly_inputs(prices)[0])
    times1 = prices.index[d]
    w1 = (times1 >= bars.index[0]) & (times1.day_of_week == 6)
    metadata = json.loads((WEEKLY / "metrics.json").read_text())
    m1 = load_student(WEEKLY / "frozen_student.pt", metadata["frozen_model"]["sha256"], x1.shape[1])
    with torch.inference_mode():
        actions, _ = m1.act_online(torch.from_numpy(x1[w1]))
    policies = {"P01": decisions_at(bars.index, times1[w1], ACTION_WEIGHTS[actions.numpy()])}
    f, rules = daily_inputs(prices)
    times, x = windows(f)
    b = json.loads((BREAKOUT / "metrics.json").read_text())
    p = json.loads((PANEL / "metrics.json").read_text())
    models = {"P02": (BREAKOUT / "frozen_student.pt", b["frozen_model_sha256"]),
              **{name: (PANEL / f"{name}.pt", sha) for name, sha in p["frozen_model_sha256"].items()}}
    daily = times >= bars.index[0]
    for name, (path, sha) in models.items():
        keep = daily if name == "P02" else daily & (times.day_of_week == 6)
        model = load_student(path, sha, x.shape[1])
        policies[name] = decisions_at(bars.index, times[keep], predict(model, x[keep]))
    selected = b["folds"][-1]["selected"]
    policies["P03"] = decisions_at(bars.index, times[daily], rules.loc[times[daily], selected])
    risk_info = json.loads((RISK / "metrics.json").read_text())
    risk_model = load_student(RISK / "frozen_student.pt", risk_info["frozen_model_sha256"], x.shape[1], RiskStudent)
    votes = policy_inputs(prices).loc[times[daily]]
    p05 = (votes.fast + votes.slow) / 2 * votes.budget
    with torch.inference_mode():
        # Keep the historical batch shape identical for the float32 prefix replay.
        segments = []
        for period in (daily & (times < cutoff), daily & (times >= cutoff)):
            today = torch.tensor(votes.loc[times[period]].to_numpy(), dtype=torch.float32)
            segments.append(risk_model.weight(risk_model.encode_hidden(x[period]), today).numpy())
        p06 = np.concatenate(segments)
    policies["P05"] = decisions_at(bars.index, times[daily], p05)
    policies["P06"] = decisions_at(bars.index, times[daily], p06)
    policies["always_long"] = np.ones(len(bars))
    prior = {"P01": pd.read_parquet(WEEKLY / "confirmation_hourly_returns.parquet")["weekly_student"],
             **pd.read_parquet(BREAKOUT / "reused_returns.parquet")[["P02", "P03"]].to_dict("series"),
             **pd.read_parquet(PANEL / "reused_returns.parquet").to_dict("series"),
             **pd.read_parquet(RISK / "reused_returns.parquet").to_dict("series")}
    metrics, raw = {}, {}
    for name, decisions in policies.items():
        sim = CausalPortfolioBacktester().run_decisions(bars, decisions, phase=2)
        if name in prior:
            np.testing.assert_allclose(sim.bar_net_returns[:len(follow)], prior[name], atol=1e-12, rtol=0.)
        returns = sim.bar_net_returns[len(follow):]
        held = sim.weights[len(follow):]
        summary = compute_performance_summary(np.r_[1., (1. + returns).cumprod()], returns, held)
        metrics[name] = {"return_pct": summary.cum_return_pct, "mdd_pct": summary.mdd_pct,
                         "position_changes": int(np.count_nonzero(np.diff(sim.weights[len(follow)-1:]))),
                         "mean_exposure": float(np.mean(held))}
        raw[name] = returns
    output = dict(start=str(cutoff), end=str(end), overlap=744, hours=len(nov), mirror_ref=REF,
                  blobs=BLOBS, continued_since_2024=True, results=metrics,
                  designed_after_short_window_was_inspected=["P05", "P06"])
    (PANEL / "november.json").write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(raw, index=nov.index).to_parquet(PANEL / "november_returns.parquet")
    print(pd.DataFrame(metrics).T.round(3))


if __name__ == "__main__":
    main()
