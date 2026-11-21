"""Daily conditional spot actor trained and evaluated using the same 200-day action space."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax

from ts.baselines.breakout_student import TARGET, daily_inputs, teacher_examples, windows
from ts.baselines.early_breakout_audit import historical_prices, policy_inputs
from ts.baselines.risk_student import OUT as PREVIOUS, fit
from ts.baselines.weekly_confirmation import paired_sharpe_interval
from ts.baselines.weekly_student import OUT as PRIOR, WEEK, decisions_at, digest
from ts.config import PROJECT_ROOT
from ts.eval.backtester import CausalPortfolioBacktester
from ts.eval.cost_model import compute_phase2_bar_base_cost_rates
from ts.eval.walk_forward import generate_phase2_walk_forward_folds

OUT = PROJECT_ROOT / "runs/baselines/cycle_student"


def allowed_dates(close):
    daily = close.asfreq("D")
    average = daily.rolling(200).mean()
    return daily.gt(average).where(average.notna())


def economic_labels(close, trend_weight, volatility, fee):
    """Noncausal training labels; last observation is exactly seven days after decision."""
    gain = np.log(close.shift(-7) / close) - 2 * fee
    risk = volatility.clip(lower=1e-8) * np.sqrt(7 / 365)
    advantage = (gain / risk).clip(-3, 3).to_numpy()[:, None]
    actions = np.linspace(0., 1., 5)
    log_prior = -.5 * ((actions - trend_weight.to_numpy()[:, None]) / .25)**2
    return pd.Series(softmax(log_prior + advantage * actions, axis=1) @ actions, index=close.index)


def training_indices(times, start, cutoff, eligible, labels, forward):
    last = times + WEEK if forward else times
    return np.flatnonzero((times >= start) & (last < cutoff) & eligible & np.isfinite(labels))


def main():
    torch.set_num_threads(2)
    cache = PROJECT_ROOT / "data/cache"
    base_path = cache / "btcusdt_1h_20200801_20231231.parquet"
    follow_path = cache / "p01_sources/confirmed_ohlcv.parquet"
    hashes = json.loads((PRIOR / "metrics.json").read_text())["sha256"]
    if digest(base_path) != hashes[base_path.name] or digest(follow_path) != json.loads((PRIOR / "confirmation.json").read_text())["data_sha256"]:
        raise ValueError("Different BTC archive")
    hist, later = pd.read_parquet(base_path), pd.read_parquet(follow_path)
    prices = pd.concat([hist[later.columns], later])
    earlier = historical_prices().close
    if earlier.index[-1] >= prices.index[0]:
        raise ValueError("Unordered warmup")
    long_close = pd.concat([earlier, prices.close])
    market_open = allowed_dates(long_close[long_close.index.hour == 23])
    features = daily_inputs(prices)[0]
    times, x = windows(features)
    permitted = market_open.loc[times]
    if permitted.isna().any():
        raise ValueError("Missing 200-day state")
    eligible = permitted.to_numpy(dtype=bool)
    state = policy_inputs(prices).loc[times]
    current = torch.tensor(state.to_numpy(), dtype=torch.float32)
    trend = (state.fast + state.slow) * state.budget / 2
    # Forward labels are constructed on a complete daily calendar, independent of feature warmup.
    clock = features.index
    hfee = pd.Series(compute_phase2_bar_base_cost_rates(prices), index=prices.index)
    label = economic_labels(prices.close.reindex(clock), trend.reindex(clock), features.volatility, hfee.reindex(clock))
    labels = {"P10": state.budget.to_numpy(), "P11": trend.to_numpy(), "P12": label.loc[times].to_numpy()}
    known = times <= hist.index[-1]
    _, logits, reasons = teacher_examples(times[known], hist, trend.to_numpy()[known],
                                          PROJECT_ROOT / "runs/study2", hashes)
    folds = generate_phase2_walk_forward_folds(hist.index)
    if len(folds) != 9:
        raise ValueError("Expected nine folds")
    spans = [(f.train_start, f.test_start, f.test_end) for f in folds]
    spans.append((later.index[0] - pd.DateOffset(years=1), later.index[0], later.index[-1]))
    OUT.mkdir(parents=True, exist_ok=True)
    predictions = {name: [] for name in labels}
    training = {name: [] for name in labels}
    last_models, last_labels = {}, {}
    fallback_dates = {name: [] for name in labels}
    frozen = {}
    for fold, (start, cutoff, end) in enumerate(spans):
        test = np.flatnonzero((times >= cutoff) & (times <= end))
        active = test[eligible[test]]
        for name, target in labels.items():
            rows = training_indices(times, start, cutoff, eligible & known, target, name == "P12")
            if len(rows):
                y = torch.tensor(target[rows], dtype=torch.float32)
                last_models[name] = fit(x[rows], current[rows], y, logits[rows], reasons[rows])
                last_labels[name] = times[rows[-1]] + (WEEK if name == "P12" else pd.Timedelta(0))
            else:
                fallback_dates[name].extend(times[test])
            model = last_models.get(name)
            weights = pd.Series(0., index=times[test])
            if model is not None and len(active):
                with torch.inference_mode():
                    weights.loc[times[active]] = model.weight(model.encode_hidden(x[active]), current[active]).numpy()
            predictions[name].append(weights)
            training[name].append({"start": str(cutoff), "last_label": str(last_labels.get(name)), "n_train": len(rows),
                                   "retained_prior_model": not len(rows) and model is not None,
                                   "active_test_days": len(active)})
            if fold == len(folds):
                if model is None:
                    raise ValueError("No prior model available to freeze")
                path = OUT / f"{name}.pt"
                torch.save(model.state_dict(), path)
                frozen[name] = digest(path)
    paper = hist.iloc[np.concatenate([f.test_indices for f in folds])]
    payload = {"training": training, "frozen_sha256": frozen,
               "sha256": {p.name: digest(p) for p in (Path(__file__), base_path, follow_path)}}
    for stage, bars in (("paper", paper), ("reused", later)):
        ref = pd.read_parquet(PREVIOUS / f"{stage}_returns.parquet").P06
        if not ref.index.equals(bars.index):
            raise ValueError("P-06 reference clock differs")
        decisions = pd.DataFrame(index=bars.index)
        dates = bars.index[bars.index.hour == 23]
        for name, parts in predictions.items():
            series = pd.concat(parts[:-1]) if stage == "paper" else parts[-1]
            if not series.index.equals(dates):
                raise ValueError("Missing daily predictions")
            decisions[name] = decisions_at(bars.index, dates, series)
            fallback = series.index.isin(fallback_dates[name])
            if fallback.any():
                cash_policy = series.mask(fallback, 0.)
                decisions[name + "_cash_fallback"] = decisions_at(bars.index, dates, cash_policy)
        for name, target in (("budget_teacher", state.budget), ("breakout_teacher", trend)):
            weights = target.loc[dates].where(market_open.loc[dates].astype(bool), 0.)
            decisions[name] = decisions_at(bars.index, dates, weights)
        metrics, returns, intervals = {}, {}, {}
        for name, weights in decisions.items():
            res = CausalPortfolioBacktester().run_decisions(bars, weights.to_numpy(), phase=2)
            metrics[name] = res.summary.to_dict()
            returns[name] = res.bar_net_returns
            if name in predictions:
                intervals[name] = paired_sharpe_interval(returns[name], ref)
        payload[stage] = {"results": metrics, "paired_sharpe_diff_ci_vs_p06": intervals}
        decisions.to_parquet(OUT / f"{stage}_decisions.parquet")
        pd.DataFrame(returns, index=bars.index).to_parquet(OUT / f"{stage}_returns.parquet")
        print(stage, pd.DataFrame(metrics).T[list(TARGET)].round(3))
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
