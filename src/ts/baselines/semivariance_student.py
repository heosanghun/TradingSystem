"""BTC spot students with staggered decisions and realized downside training targets."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax

from ts.baselines.breakout_student import daily_inputs, teacher_examples, windows
from ts.baselines.cycle_student import allowed_dates, economic_labels, training_indices
from ts.baselines.early_breakout_audit import historical_prices, policy_inputs
from ts.baselines.risk_student import fit
from ts.baselines.weekly_student import OUT as PRIOR, decisions_at, digest
from ts.config import PROJECT_ROOT
from ts.eval.backtester import CausalPortfolioBacktester
from ts.eval.cost_model import compute_phase2_bar_base_cost_rates
from ts.eval.walk_forward import generate_phase2_walk_forward_folds

OUT = PROJECT_ROOT / "runs/baselines/semivariance_student"
POLICIES = {"P14": (False, (5, 11, 17, 23)), "P15": (True, (5, 11, 17, 23)),
            "P16": (True, (23,))}


def risk_inputs(hourly_close):
    squared = hourly_close.pct_change(fill_method=None).clip(upper=0).pow(2)
    return pd.DataFrame({f"down_{n}d": np.sqrt(8760 * squared.rolling(24*n).mean())
                         for n in (1, 7, 30)})


def path_labels(close, center, vol, fee, next_downside):
    risk = vol.clip(lower=1e-8) * np.sqrt(7/365)
    advantage = ((np.log(close.shift(-7)/close) - 2*fee) / risk).clip(-3, 3)
    actions = np.linspace(0., 1., 5)
    scores = -.5*((actions - center.to_numpy()[:, None])/.25)**2
    scores += advantage.to_numpy()[:, None] * actions
    scores -= (next_downside / risk.pow(2)).to_numpy()[:, None] * actions**2
    return pd.Series(softmax(scores, axis=1) @ actions, index=close.index)


def cohort(prices, warm_close, hour, with_risk):
    offset = pd.Timedelta(hours=23-hour)
    shifted = prices.copy()
    shifted.index = shifted.index + offset
    # Missing hourly bars carry only their last known close into features. No trades on missing bars.
    regular = shifted[["close", "volume"]].asfreq("h")
    regular["close"] = regular.close.ffill()
    regular["volume"] = regular.volume.fillna(0.)
    features = daily_inputs(regular)[0]
    signals = policy_inputs(regular)
    daily = features.index
    all_closes = warm_close.copy()
    all_closes.index = all_closes.index + offset
    all_closes = all_closes.asfreq("h").ffill()
    allowed = allowed_dates(all_closes[all_closes.index.hour == 23]).reindex(daily)
    close = regular.close.reindex(daily)
    observed = shifted.close.asfreq("h")
    fees = pd.Series(compute_phase2_bar_base_cost_rates(shifted), index=shifted.index).reindex(daily)
    center = (signals.fast + signals.slow) * signals.budget / 2
    if with_risk:
        features = features.join(risk_inputs(regular.close).reindex(daily))
        # Only fully observed future paths enter training. This series never enters features or decisions.
        future = observed.pct_change(fill_method=None).clip(upper=0).pow(2).rolling(168).sum().shift(-168)
        y = path_labels(close, center, features.volatility, fees, future.reindex(daily))
    else:
        y = economic_labels(close, center, features.volatility, fees)
    t, x = windows(features)
    keep = t.isin(shifted.index) & allowed.loc[t].notna().to_numpy()
    t = t[keep]
    state = torch.tensor(signals.loc[t].to_numpy(), dtype=torch.float32)
    return t-offset, x[keep], state, y.loc[t].to_numpy(), allowed.loc[t].to_numpy(dtype=bool), center.loc[t].to_numpy()


def combine(chunks):
    times = chunks[0][0].append([chunk[0] for chunk in chunks[1:]])
    order = np.argsort(times)
    if times.has_duplicates:
        raise ValueError("Overlapping observation clocks")
    return (times[order], *(torch.cat([chunk[i] for chunk in chunks])[order] for i in (1, 2)),
            *(np.concatenate([chunk[i] for chunk in chunks])[order] for i in (3, 4, 5)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", nargs="+", default=list(POLICIES), choices=POLICIES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(2)
    out = OUT / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    cache = PROJECT_ROOT / "data/cache"
    base = cache/"btcusdt_1h_20200801_20231231.parquet"
    extra = cache/"p01_sources/confirmed_ohlcv.parquet"
    hashes = json.loads((PRIOR/"metrics.json").read_text())["sha256"]
    if digest(base) != hashes[base.name] or digest(extra) != json.loads((PRIOR/"confirmation.json").read_text())["data_sha256"]:
        raise ValueError("Unexpected price archive")
    hist, later = pd.read_parquet(base), pd.read_parquet(extra)
    prices = pd.concat([hist[later.columns], later])
    warm = pd.concat([historical_prices().close, prices.close])
    folds = generate_phase2_walk_forward_folds(hist.index)
    if len(folds) != 9:
        raise ValueError("Incomplete paper calendar")
    spans = [(f.train_start, f.test_start, f.test_end) for f in folds]
    spans.append((later.index[0]-pd.DateOffset(years=1), later.index[0], later.index[-1]))
    paper = hist.iloc[np.concatenate([f.test_indices for f in folds])]
    source = PROJECT_ROOT/"runs/study2"
    with np.load(source/"teacher_signals.npz") as archive:
        available = hist.index.take(archive["bar_indices"])
    hours = sorted({h for name in args.policies for h in POLICIES[name][1]})
    tt = available[available.hour.isin(hours)]
    _, logits, reasons = teacher_examples(tt, hist, np.zeros(len(tt)), source, hashes)
    for name in args.policies:
        risk, hours = POLICIES[name]
        times, x, state, y, allowed, rule = combine([cohort(prices, warm, hour, risk) for hour in hours])
        teacher_idx = tt.get_indexer(times)
        pieces, records = [], []
        model = None
        for start, cutoff, end in spans:
            rows = training_indices(times, start, cutoff, allowed & (teacher_idx >= 0), y, True)
            test = np.flatnonzero((times >= cutoff) & (times <= end))
            active = test[allowed[test]]
            if len(rows):
                ids = teacher_idx[rows]
                model = fit(x[rows], state[rows], torch.tensor(y[rows], dtype=torch.float32),
                            logits[ids], reasons[ids], seed=args.seed)
            weights = pd.Series(0., index=times[test])
            if model is not None and len(active):
                with torch.inference_mode():
                    weights.loc[times[active]] = model.weight(model.encode_hidden(x[active]), state[active]).numpy()
            pieces.append(weights)
            records.append(dict(test_start=str(cutoff), n_train=len(rows), n_active=len(active),
                                last_label=str(times[rows[-1]]+pd.Timedelta(days=7)) if len(rows) else None))
        if model is None:
            raise ValueError("Missing trained model")
        path = out / f"{name}.pt"
        torch.save(model.state_dict(), path)
        report = {"policy": name, "seed": args.seed, "weights_sha256": digest(path), "folds": records}
        rules = pd.Series(np.where(allowed, rule, 0.), index=times)
        for stage, bars, steps in (("paper", paper, pd.concat(pieces[:-1])), ("reused", later, pieces[-1])):
            dates = bars.index[bars.index.hour.isin(hours)]
            if not dates.equals(steps.index):
                raise ValueError("Missing observed decision timestamps")
            weights = pd.DataFrame({name:decisions_at(bars.index, dates, steps),
                                    "rule":decisions_at(bars.index, dates, rules.loc[dates])}, index=bars.index)
            report[stage], returns = {}, {}
            for key, dec in weights.items():
                sim = CausalPortfolioBacktester().run_decisions(bars, dec.to_numpy(), phase=2)
                report[stage][key] = sim.summary.to_dict()
                returns[key] = sim.bar_net_returns
            weights.to_parquet(out / f"{name}_{stage}_decisions.parquet")
            pd.DataFrame(returns, index=bars.index).to_parquet(out / f"{name}_{stage}_returns.parquet")
            print(name, stage, report[stage][name], flush=True)
        (out / f"{name}.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
