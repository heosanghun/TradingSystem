"""P-02 daily spot student and P-03 train-only trend comparator."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ts.baselines.weekly_confirmation import paired_sharpe_interval
from ts.baselines.weekly_student import OUT as PRIOR, decisions_at, digest, fit_student
from ts.config import PROJECT_ROOT
from ts.eval.backtester import CausalPortfolioBacktester
from ts.eval.walk_forward import generate_phase2_walk_forward_folds
from ts.study1.ppo_expert import ACTION_WEIGHTS
from ts.study3.teacher_signal import FinBERTReasoningEncoder, compute_teacher_soft_logits_and_probs

OUT = PROJECT_ROOT / "runs/baselines/breakout_student"
RULES = ("momentum", "breakout", "moving_average")
TARGET = dict(sharpe=.92, sortino=2.08, cum_return_pct=25.4, cagr_pct=10.6, mdd_pct=-24.)


def daily_inputs(bars):
    """Daily features timestamped at the last hourly candle, with no backward fill."""
    if not bars.index.is_unique or not bars.index.is_monotonic_increasing or str(bars.index.tz) != "UTC":
        raise ValueError("Expected ordered, unique UTC candles")
    close = bars.close[bars.index.hour == 23].asfreq("D")
    volume = bars.volume.resample("D").sum(min_count=1)
    volume.index += pd.Timedelta(hours=23)
    log_close = np.log(close)
    sigma = log_close.diff().rolling(30).std() * np.sqrt(365)
    budget = (.25 / sigma.clip(lower=1e-6)).clip(upper=1.)
    history = close.shift(1).rolling(50)
    upper, lower = history.max(), history.min()
    state = pd.Series(np.nan, index=close.index)
    state.loc[close > upper] = 1.
    state.loc[close < lower] = 0.
    state = state.ffill().fillna(0.)
    location = (close - lower) / (upper - lower).clip(lower=1e-8)
    inputs = pd.DataFrame({f"momentum_{n}": log_close.diff(n) for n in (1, 7, 30)})
    inputs = inputs.assign(location=location, state=state, volatility=sigma, budget=budget,
                           volume_ratio=volume / volume.rolling(30).mean().clip(lower=1e-8))
    rules = pd.DataFrame({"momentum": (close > close.shift(30)) * budget,
                          "breakout": state * budget,
                          "moving_average": (close > close.rolling(50).mean()) * budget})
    grid = ACTION_WEIGHTS[ACTION_WEIGHTS >= 0]
    rules["imitation_target"] = grid[np.abs(rules.breakout.to_numpy()[:, None] - grid).argmin(axis=1)]
    rules.loc[upper.isna() | lower.isna() | sigma.isna() | close.isna(), :] = np.nan
    return inputs.clip(-8, 8), rules


def windows(inputs):
    rolling = np.lib.stride_tricks.sliding_window_view(inputs.to_numpy(dtype=np.float32), 24, axis=0)
    valid = np.isfinite(rolling).all(axis=(1, 2))
    return inputs.index[23:][valid], torch.from_numpy(np.ascontiguousarray(rolling[valid]))


def teacher_examples(times, hist, labels, teacher, prior_hashes):
    files = [teacher / name for name in ("teacher_signals.npz", "consensus_texts.json")]
    if any(digest(path) != prior_hashes[path.name] for path in files):
        raise ValueError("Study 2 teacher inputs differ from the validated P-01 inputs")
    with np.load(files[0]) as archive:
        idx = hist.index.take(archive["bar_indices"]).get_indexer(times)
        if (idx < 0).any():
            raise ValueError("Missing Study 2 teacher timestamp")
        logits, _ = compute_teacher_soft_logits_and_probs(archive["w_finals"][idx])
    texts = json.loads(files[1].read_text(encoding="utf-8"))
    codes, vocab = pd.factorize(np.asarray([texts[i] for i in idx], dtype=object))
    reasoning = FinBERTReasoningEncoder().encode_texts(vocab.tolist())[codes]
    actions = np.abs(labels[:, None] - ACTION_WEIGHTS).argmin(axis=1)
    return [torch.from_numpy(arr) for arr in (actions, logits, reasoning)]


def align_rules(bars, rules):
    """Start in cash until the first daily decision in this evaluation segment."""
    valid = rules.loc[rules.index.intersection(bars.index)].dropna()
    return {name: decisions_at(bars.index, valid.index, valid[name]) for name in rules}


def pick_rule(bars, rules, bt):
    weights = align_rules(bars, rules[list(RULES)])
    scores = {k: bt.run_decisions(bars, v, phase=2).summary.sortino for k, v in weights.items()}
    return max(RULES, key=scores.get), scores


def predict(model, x):
    with torch.inference_mode():
        _, probs = model.act_online(x)
    grid = np.flatnonzero(ACTION_WEIGHTS >= 0)
    return ACTION_WEIGHTS[grid[probs[:, grid].argmax(dim=1).numpy()]]


def evaluate(stage, bars, rules, students, selected, prior_file, bt):
    decisions = align_rules(bars, rules)
    for name, series in (("P02", students), ("P03", selected)):
        if not series.index.equals(bars.index[bars.index.hour == 23]) or series.isna().any():
            raise ValueError(f"Incomplete daily decisions for {name}")
        decisions[name] = decisions_at(bars.index, series.index, series)
    prior = pd.read_parquet(PRIOR / prior_file)
    if not bars.index.equals(prior.index):
        raise ValueError("P-01 and new calendar differ")
    results, returns = {}, {}
    for name, weights in decisions.items():
        res = bt.run_decisions(bars, weights, phase=2)
        results[name] = dict(**res.summary.to_dict(),
                             weight_changes=int(np.count_nonzero(np.diff(res.weights, prepend=0))),
                             mean_exposure=float(np.mean(res.weights)))
        returns[name] = res.bar_net_returns
    pd.DataFrame(returns, index=bars.index).to_parquet(OUT / f"{stage}_returns.parquet")
    pd.DataFrame(decisions, index=bars.index).to_parquet(OUT / f"{stage}_decisions.parquet")
    return {"results": results,
            "paired_sr_diff_ci_vs_p01": paired_sharpe_interval(returns["P02"], prior["weekly_student"]),
            "paired_sr_diff_ci_vs_rule_target": paired_sharpe_interval(returns["P02"], returns["imitation_target"]),
            "p02_target_gap": {k: max(0., v - results["P02"][k]) for k, v in TARGET.items()
                               if stage == "paper" or k != "cum_return_pct"}}


def main():
    torch.set_num_threads(2)
    bt = CausalPortfolioBacktester()
    cache = PROJECT_ROOT / "data/cache"
    base, follow = cache / "btcusdt_1h_20200801_20231231.parquet", cache / "p01_sources/confirmed_ohlcv.parquet"
    hashes = json.loads((PRIOR / "metrics.json").read_text())["sha256"]
    if digest(base) != hashes[base.name] or digest(follow) != json.loads((PRIOR / "confirmation.json").read_text())["data_sha256"]:
        raise ValueError("Market data differ from validated P-01 inputs")
    hist, later = pd.read_parquet(base), pd.read_parquet(follow)
    features, rules = daily_inputs(pd.concat([hist[later.columns], later]))
    times, x = windows(features)
    historical = times <= hist.index[-1]
    train_times = times[historical]
    y = rules.imitation_target.loc[train_times].to_numpy()
    teacher = teacher_examples(train_times, hist, y, PROJECT_ROOT / "runs/study2", hashes)
    OUT.mkdir(parents=True, exist_ok=True)
    records, student_series, selection_series = [], [], []
    folds = generate_phase2_walk_forward_folds(hist.index)
    if len(folds) != 9:
        raise ValueError("Expected the nine thesis folds")
    spans = [(f.train_start, f.test_start, f.test_end) for f in folds]
    spans.append((later.index[0] - pd.DateOffset(years=1), later.index[0], later.index[-1]))
    for i, (start, cutoff, end) in enumerate(spans):
        rows = np.flatnonzero((train_times >= start) & (train_times < cutoff))
        test = np.flatnonzero((times >= cutoff) & (times <= end))
        model = fit_student(x[rows], *(tensor[rows] for tensor in teacher))
        choice, scores = pick_rule(hist.loc[(hist.index >= start) & (hist.index < cutoff)], rules, bt)
        dates = times[test]
        s = pd.Series(predict(model, x[test]), index=dates)
        p = rules[choice].loc[dates]
        records.append({"cutoff": str(cutoff), "last_training_observation": str(train_times[rows[-1]]),
                        "n_train": len(rows), "n_decisions": len(test), "selected": choice,
                        "training_sortino": scores})
        if i < len(folds):
            student_series.append(s)
            selection_series.append(p)
        else:
            torch.save(model.state_dict(), OUT / "frozen_student.pt")
            later_student, later_selection = s, p
    paper = hist.iloc[np.concatenate([f.test_indices for f in folds])]
    payload = {"inputs_sha256": {p.name: digest(p) for p in (base, follow, Path(__file__))},
               "frozen_model_sha256": digest(OUT / "frozen_student.pt"), "folds": records,
               "paper": evaluate("paper", paper, rules, pd.concat(student_series), pd.concat(selection_series),
                                 "hourly_returns.parquet", bt),
               "reused": evaluate("reused", later, rules, later_student, later_selection,
                                  "confirmation_hourly_returns.parquet", bt)}
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    for stage in ("paper", "reused"):
        print(stage, pd.DataFrame(payload[stage]["results"]).T[list(TARGET)].round(3))


if __name__ == "__main__":
    main()
