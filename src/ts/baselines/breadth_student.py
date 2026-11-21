"""P-07 to P-09 fixed hypotheses for the continuous BTC spot student."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ts.baselines.breakout_student import TARGET, daily_inputs, teacher_examples, windows
from ts.baselines.early_breakout_audit import policy_inputs
from ts.baselines.risk_student import OUT as PREVIOUS, fit
from ts.baselines.weekly_confirmation import paired_sharpe_interval
from ts.baselines.weekly_student import OUT as PRIOR, decisions_at, digest
from ts.config import PROJECT_ROOT
from ts.data.market_breadth import SYMBOLS, market_closes
from ts.eval.backtester import CausalPortfolioBacktester
from ts.eval.walk_forward import generate_phase2_walk_forward_folds

OUT = PROJECT_ROOT / "runs/baselines/breadth_student"


def breadth_inputs(close, market):
    market = market.reindex(close.index)
    features, signals = {}, {}
    for n in (7, 30):
        moves = market.pct_change(n, fill_method=None)
        btc = close.pct_change(n, fill_method=None)
        share = moves.gt(0).mean(axis=1).where(moves.notna().all(axis=1))
        features[f"breadth_{n}"] = share
        signals[n] = ((btc > 0) & (share >= .5)).astype(float).where(btc.notna() & share.notna())
        if n == 30:
            features["relative_30"] = btc - moves.median(axis=1, skipna=False)
    return pd.DataFrame(features), pd.DataFrame({"fast": signals[7], "slow": signals[30]})


def six_hour_inputs(bars):
    # A six-hour candle ends when its final observable hourly candle closes.
    close = bars.close.resample("6h").last()
    close.index += pd.Timedelta(hours=5)
    ret = np.log(close).diff()
    features = pd.DataFrame({"return_6h": ret, "return_24h": np.log(close).diff(4),
                             "positive_share": ret.gt(0).where(ret.notna()).rolling(4).mean()})
    signals = {}
    for name, high, low in (("fast", 20, 10), ("slow", 55, 20)):
        prev = close.shift(1)
        state = pd.Series(np.nan, index=close.index)
        state.loc[close > prev.rolling(high * 4).max()] = 1.
        state.loc[close < prev.rolling(low * 4).min()] = 0.
        signals[name] = state.ffill().fillna(0.)
    return features.loc[features.index.hour == 23], pd.DataFrame(signals).loc[lambda f: f.index.hour == 23]


def train_policy(name, features, states, teacher_times, logits, reasons, spans):
    times, x = windows(features)
    states = states.loc[times]
    inputs = torch.tensor(states[["fast", "slow", "budget"]].to_numpy(), dtype=torch.float32)
    targets = torch.tensor(((states.fast + states.slow) * states.budget / 2).to_numpy(), dtype=torch.float32)
    if not torch.isfinite(inputs).all():
        raise ValueError("Missing decision state")
    predictions, training = [], []
    for start, cutoff, end in spans:
        train = np.flatnonzero((times >= start) & (times < cutoff))
        test = np.flatnonzero((times >= cutoff) & (times <= end))
        idx = teacher_times.get_indexer(times[train])
        if (idx < 0).any() or not len(test):
            raise ValueError("Missing teacher alignment or test decisions")
        model = fit(x[train], inputs[train], targets[train], logits[idx], reasons[idx])
        with torch.inference_mode():
            weights = model.weight(model.encode_hidden(x[test]), inputs[test]).numpy()
        predictions.append(pd.Series(weights, index=times[test]))
        training.append(dict(first=str(times[train[0]]), last=str(times[train[-1]]),
                             n=len(train), test_from=str(cutoff)))
    path = OUT / f"{name}.pt"
    torch.save(model.state_dict(), path)
    return predictions, {"folds": training, "frozen_sha256": digest(path)}


def main():
    torch.set_num_threads(2)
    cache = PROJECT_ROOT / "data/cache"
    hist_file = cache / "btcusdt_1h_20200801_20231231.parquet"
    later_file = cache / "p01_sources/confirmed_ohlcv.parquet"
    hashes = json.loads((PRIOR / "metrics.json").read_text())["sha256"]
    if digest(hist_file) != hashes[hist_file.name] or digest(later_file) != json.loads((PRIOR / "confirmation.json").read_text())["data_sha256"]:
        raise ValueError("Market inputs differ from previous experiment")
    hist, later = pd.read_parquet(hist_file), pd.read_parquet(later_file)
    bars = pd.concat([hist[later.columns], later])
    base = daily_inputs(bars)[0]
    close = bars.close.reindex(base.index)
    market, overlaps = market_closes(close.index)
    extra, broad = breadth_inputs(close, market)
    fast_features, fast = six_hour_inputs(bars)
    legacy = policy_inputs(bars)
    panel = base.join(extra)
    candidates = {"P07": (panel, legacy), "P08": (panel, broad.assign(budget=base.budget)),
                  "P09": (base.join(fast_features), fast.assign(budget=base.budget))}
    # Teacher signals are identical across candidates and indexed by actual market timestamp.
    dates = windows(base)[0]
    dates = dates[dates <= hist.index[-1]]
    _, logits, reasons = teacher_examples(dates, hist, np.zeros(len(dates)), PROJECT_ROOT / "runs/study2", hashes)
    folds = generate_phase2_walk_forward_folds(hist.index)
    if len(folds) != 9:
        raise ValueError("Expected nine thesis folds")
    spans = [(f.train_start, f.test_start, f.test_end) for f in folds]
    spans.append((later.index[0] - pd.DateOffset(years=1), later.index[0], later.index[-1]))
    OUT.mkdir(parents=True, exist_ok=True)
    predicted, training = {}, {}
    for name, (features, states) in candidates.items():
        predicted[name], training[name] = train_policy(name, features, states, dates, logits, reasons, spans)
    paper = hist.iloc[np.concatenate([f.test_indices for f in folds])]
    files = [Path(__file__), hist_file, later_file, *(cache / f"{s.lower()}usdt_1h_20200725_20231231.parquet" for s in SYMBOLS)]
    report = {"validation_overlap_days": overlaps, "training": training,
              "sha256": {p.name: digest(p) for p in files}}
    for stage, frame in (("paper", paper), ("reused", later)):
        prior = pd.read_parquet(PREVIOUS / f"{stage}_returns.parquet").P06
        decisions = pd.read_parquet(PREVIOUS / f"{stage}_decisions.parquet")[["P06"]].copy()
        if not frame.index.equals(decisions.index) or not frame.index.equals(prior.index):
            raise ValueError("P-06 reference clock differs")
        for name, pieces in predicted.items():
            daily = pd.concat(pieces[:-1]) if stage == "paper" else pieces[-1]
            if not daily.index.equals(frame.index[frame.index.hour == 23]):
                raise ValueError("Missing daily actions")
            decisions[name] = decisions_at(frame.index, daily.index, daily)
        results, net, intervals = {}, {}, {}
        for name, policy in decisions.items():
            res = CausalPortfolioBacktester().run_decisions(frame, policy.to_numpy(), phase=2)
            results[name] = res.summary.to_dict()
            net[name] = res.bar_net_returns
            if name != "P06":
                intervals[name] = paired_sharpe_interval(net[name], prior)
        np.testing.assert_allclose(net["P06"], prior.to_numpy(), rtol=0, atol=1e-12)
        report[stage] = dict(results=results, paired_sharpe_diff_ci_vs_p06=intervals)
        pd.DataFrame(net, index=frame.index).to_parquet(OUT / f"{stage}_returns.parquet")
        decisions.to_parquet(OUT / f"{stage}_decisions.parquet")
        print(stage, pd.DataFrame(results).T[list(TARGET)].round(3))
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
