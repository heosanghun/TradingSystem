"""Research-only daily spot students with a momentum condition during training."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ts.baselines.breakout_student import daily_inputs, teacher_examples, windows
from ts.baselines.cycle_student import allowed_dates, economic_labels, training_indices
from ts.baselines.early_breakout_audit import historical_prices, policy_inputs
from ts.baselines.risk_student import RiskStudent, fit
from ts.baselines.weekly_student import OUT as PRIOR, decisions_at, digest
from ts.config import PROJECT_ROOT
from ts.eval.backtester import CausalPortfolioBacktester
from ts.eval.cost_model import compute_phase2_bar_base_cost_rates
from ts.eval.walk_forward import generate_phase2_walk_forward_folds

OUT = PROJECT_ROOT / "runs/baselines/momentum_condition"
NAMES = ("P17", "P18", "P19", "P20")


def momentum(close, period):
    if period == 0:
        return close.notna()
    return close > close.shift(period)


def choose_early(bars):
    daily = bars.close[bars.index.hour == 23].asfreq("D")
    long_regime = allowed_dates(daily).eq(True)
    s = policy_inputs(bars)
    teacher = (s.fast + s.slow) * s.budget / 2
    sample = bars.loc["2018-05-01":"2020-07-31"]
    scores, eligible = {}, []
    for n in (0, 21, 30, 42):
        dates = sample.index[sample.index.hour == 23]
        w = teacher.where(long_regime & momentum(daily, n), 0.).loc[dates].fillna(0.)
        dec = decisions_at(sample.index, dates, w)
        bt = CausalPortfolioBacktester()
        overall = bt.run_decisions(sample, dec, phase=2).summary
        mid = len(sample)//2
        halves = [bt.run_decisions(sample.iloc[sl], dec[sl], phase=2).summary.sortino
                  for sl in (slice(None, mid), slice(mid, None))]
        scores[n] = {"halves": halves, "cagr_pct":overall.cagr_pct, "mdd_pct":overall.mdd_pct}
        if overall.cagr_pct >= 10.6 and overall.mdd_pct >= -24.:
            eligible.append(n)
    choice = max(eligible, key=lambda n: min(scores[n]["halves"])) if eligible else 0
    return choice, scores


def make_data():
    cache = PROJECT_ROOT/"data/cache"
    base = cache/"btcusdt_1h_20200801_20231231.parquet"
    extra = cache/"p01_sources/confirmed_ohlcv.parquet"
    hashes = json.loads((PRIOR/"metrics.json").read_text())["sha256"]
    if digest(base) != hashes[base.name] or digest(extra) != json.loads((PRIOR/"confirmation.json").read_text())["data_sha256"]:
        raise ValueError("Unexpected price archive")
    hist, later = pd.read_parquet(base), pd.read_parquet(extra)
    old = historical_prices()
    bars = pd.concat([hist[later.columns], later])
    full_close = pd.concat([old.close, bars.close])
    long_close = full_close[full_close.index.hour==23].asfreq("D")
    features = daily_inputs(bars)[0]
    times, x = windows(features)
    state = policy_inputs(bars)
    trend = (state.fast+state.slow)*state.budget/2
    clock = features.index
    fee = pd.Series(compute_phase2_bar_base_cost_rates(bars), index=bars.index).reindex(clock)
    def label(center):
        return economic_labels(bars.close.reindex(clock), center, features.volatility, fee).loc[times].to_numpy()
    known = times <= hist.index[-1]
    _, logits, reasons = teacher_examples(times[known], hist, trend.loc[times[known]].to_numpy(),
                                          PROJECT_ROOT/"runs/study2", hashes)
    gate = allowed_dates(long_close).loc[times]
    if gate.isna().any():
        raise ValueError("Missing 200-day regime")
    return dict(hist=hist, later=later, old=old, times=times, x=x, close=long_close,
                state=torch.tensor(state.loc[times].to_numpy(), dtype=torch.float32), regime=gate.to_numpy(bool),
                known=known, logits=logits, reasons=reasons, trend=trend.loc[times], budget=state.budget.loc[times],
                trend_y=label(trend), budget_y=label(state.budget))


def evaluate(name, n, seed, data):
    d = data
    t, x, s = d["times"], d["x"], d["state"]
    strength = momentum(d["close"], n).loc[t].to_numpy()
    allowed = d["regime"] if name=="P18" else d["regime"] & strength
    y = d["budget_y"] if name=="P20" else d["trend_y"]
    if name=="P18":
        y = np.where(strength, y, np.where(np.isfinite(y), 0., np.nan))
    rule = d["budget"] if name=="P20" else d["trend"]
    rule = rule.where(d["regime"] & strength, 0.)
    train_and_evaluate(d, allowed, y, rule, seed, OUT/f"{name}_{n}d_seed{seed}",
                       {"policy": name, "lookback": n, "seed": seed,
                        "source_sha256": digest(Path(__file__))})


def train_and_evaluate(d, allowed, y, rule, seed, out, metadata):
    """Refit the spot student on eligible days of each purged historical fold."""
    t, x, s = d["times"], d["x"], d["state"]
    folds = generate_phase2_walk_forward_folds(d["hist"].index)
    if len(folds)!=9:
        raise ValueError("Missing thesis fold")
    paper = d["hist"].iloc[np.concatenate([f.test_indices for f in folds])]
    later = d["later"]
    spans = [(f.train_start, f.test_start, f.test_end) for f in folds]
    spans.append((later.index[0]-pd.DateOffset(years=1), later.index[0], later.index[-1]))
    torch.manual_seed(seed)
    initial = RiskStudent(x.shape[1]).eval()
    model, pieces, records = None, [], []
    for start, cutoff, end in spans:
        rows = training_indices(t, start, cutoff, allowed & d["known"], y, True)
        test = np.flatnonzero((t>=cutoff) & (t<=end))
        active = test[allowed[test]]
        if len(rows):
            model = fit(x[rows], s[rows], torch.tensor(y[rows], dtype=torch.float32),
                        d["logits"][rows], d["reasons"][rows], seed=seed)
        predicted = pd.DataFrame(0., index=t[test], columns=["student", "untrained"])
        if len(active):
            with torch.inference_mode():
                for key, actor in (("student",model), ("untrained",initial)):
                    if actor is not None:
                        predicted.loc[t[active],key] = actor.weight(actor.encode_hidden(x[active]), s[active]).numpy()
        pieces.append(predicted)
        records.append(dict(test_from=str(cutoff), n_train=len(rows), active=len(active),
                            last_label=str(t[rows[-1]]+pd.Timedelta(days=7)) if len(rows) else None))
    if model is None:
        raise ValueError("No trained position model")
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out/"frozen.pt")
    payload = {**metadata, "folds": records, "weights_sha256": digest(out/"frozen.pt")}
    for stage, bars, steps in (("paper",paper,pd.concat(pieces[:-1])), ("reused",later,pieces[-1])):
        expected = bars.index[bars.index.hour==23]
        if not expected.equals(steps.index):
            raise ValueError("Missing daily decisions")
        steps = steps.assign(rule=rule.loc[expected])
        decisions = pd.DataFrame({key:decisions_at(bars.index,expected,series) for key,series in steps.items()},index=bars.index)
        metrics, returns = {}, {}
        for key,series in decisions.items():
            sim=CausalPortfolioBacktester().run_decisions(bars,series.to_numpy(),phase=2)
            metrics[key]=sim.summary.to_dict()
            returns[key]=sim.bar_net_returns
        payload[stage]=metrics
        decisions.to_parquet(out/f"{stage}_decisions.parquet")
        pd.DataFrame(returns,index=bars.index).to_parquet(out/f"{stage}_returns.parquet")
        print(out.name,stage,{k:round(metrics["student"][k],4) for k in ("sharpe","sortino","cum_return_pct","cagr_pct","mdd_pct")},flush=True)
    (out/"metrics.json").write_text(json.dumps(payload,indent=2,allow_nan=False),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--policies",nargs="+",default=list(NAMES),choices=NAMES)
    parser.add_argument("--seeds",nargs="+",type=int,default=[42])
    parser.add_argument("--lookbacks",nargs="+",type=int)
    args=parser.parse_args()
    torch.set_num_threads(2)
    data=make_data()
    selected,scores=choose_early(data["old"])
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"early_selection.json").write_text(json.dumps({"lookback":selected,"scores":scores},indent=2),encoding="utf-8")
    for name in args.policies:
        for n in args.lookbacks or [selected if name=="P19" else 30]:
            for seed in args.seeds:
                evaluate(name,n,seed,data)


if __name__=="__main__":
    main()
