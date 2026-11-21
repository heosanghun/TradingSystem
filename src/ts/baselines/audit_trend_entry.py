"""Replay daily BTC entries, attribute P-20 returns, and check execution stress."""
import argparse
import json

import numpy as np
import pandas as pd
import torch

from ts.baselines.audit_momentum_condition import block_intervals, statistics
from ts.baselines.breakout_student import daily_inputs, windows
from ts.baselines.conditional_budget import NAMES as BUDGET_NAMES, risk_budgets
from ts.baselines.early_breakout_audit import historical_prices, policy_inputs
from ts.baselines.momentum_condition_student import OUT as PRIOR
from ts.baselines.november_spot_check import FIELDS, load_student, mirror_month
from ts.baselines.risk_student import RiskStudent
from ts.baselines.trend_entry_student import OUT, day_volume, entry_signals
from ts.baselines.weekly_student import decisions_at
from ts.config import PROJECT_ROOT
from ts.eval.backtester import CausalPortfolioBacktester


def attribution(net, held):
    active = pd.Series(held > 0, index=net.index)
    spell = active.ne(active.shift()).cumsum()
    observation = active.groupby(spell).cumcount()
    first_week = active & (observation < 168)
    squares = net.clip(upper=0)**2
    groups = {"first_week": first_week, "later_in_spell": active & ~first_week, "flat": ~active}
    return {key: {"hours": int(mask.sum()), "net_sum_pct": 100 * net[mask].sum(),
                  "downside_share": squares[mask].sum()/squares.sum()} for key, mask in groups.items()}


def november(hist, later, folder, name):
    # Cross-sectional source ends in October. Do not silently change P24 inputs.
    if name == "P24":
        return {"status": "four-market November closes required"}
    overlap = mirror_month("10")
    pd.testing.assert_frame_equal(later.loc["2026-10", FIELDS], overlap[FIELDS],
                                  check_names=False, check_freq=False)
    extra = mirror_month("11").loc[:"2026-11-20"]
    expected = pd.date_range("2026-11-01", periods=480, freq="h", tz="UTC")
    if not extra.index.equals(expected):
        raise ValueError("Incomplete November check")
    bars = pd.concat([hist[FIELDS], later, extra])
    full = pd.concat([historical_prices(), bars])
    close = full.close[full.index.hour == 23].asfreq("D")
    info = json.loads((folder / "metrics.json").read_text())
    signals = entry_signals(close, day_volume(full), early_choice=info.get("early_choice"))
    gate = signals["P37" if name in BUDGET_NAMES else name]
    times, x = windows(daily_inputs(bars)[0])
    state = policy_inputs(bars).loc[times]
    if name in BUDGET_NAMES:
        state["budget"] = risk_budgets(full)[name].loc[times]
    model = load_student(folder / "frozen.pt", info["weights_sha256"], x.shape[1], RiskStudent)
    dates = extra.index[extra.index.hour == 23]
    w = pd.Series(0., index=dates)
    prev = pd.read_parquet(folder / "reused_decisions.parquet").student
    for clock in (later.index[later.index.hour == 23], dates):
        predicted = pd.Series(0., index=clock)
        active = times.isin(clock) & gate.loc[times].to_numpy()
        if active.any():
            with torch.inference_mode():
                s = torch.tensor(state.iloc[active].to_numpy(), dtype=torch.float32)
                predicted.loc[times[active]] = model.weight(model.encode_hidden(x[active]), s).numpy()
        if clock.equals(dates):
            w = predicted
        else:
            np.testing.assert_allclose(predicted, prev.loc[clock], atol=1e-7, rtol=0.)
    daily = pd.concat([prev, w])
    prices = pd.concat([later, extra])
    result = CausalPortfolioBacktester().run_decisions(prices, decisions_at(prices.index, daily.index, daily), phase=2)
    r = result.bar_net_returns[-len(extra):]
    return {"n_hours": len(r), "active_hours": int(np.count_nonzero(result.weights[-len(extra):])),
            "cum_return_pct": statistics(r)["cum_return_pct"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("policy")
    args = parser.parse_args()
    torch.set_num_threads(2)
    cache = PROJECT_ROOT / "data/cache"
    hist = pd.read_parquet(cache / "btcusdt_1h_20200801_20231231.parquet")
    later = pd.read_parquet(cache / "p01_sources/confirmed_ohlcv.parquet")
    folder = OUT / f"{args.policy}_seed42"
    audit = {}
    for stage, market in (("paper", hist), ("reused", later)):
        weights = pd.read_parquet(folder / f"{stage}_decisions.parquet").student
        bars = market.loc[weights.index]
        saved = pd.read_parquet(folder / f"{stage}_returns.parquet")
        prior_dir = PRIOR / "P20_30d_seed42"
        old = pd.read_parquet(prior_dir / f"{stage}_decisions.parquet").student
        prior = pd.read_parquet(prior_dir / f"{stage}_returns.parquet").student
        if not weights.index.equals(prior.index) or not old.index.equals(prior.index):
            raise ValueError("Unmatched comparison clock")
        bt = CausalPortfolioBacktester()
        current = bt.run_decisions(bars, weights, phase=2)
        previous = bt.run_decisions(bars, old, phase=2)
        np.testing.assert_allclose(current.bar_net_returns, saved.student, atol=1e-12, rtol=0.)
        np.testing.assert_allclose(previous.bar_net_returns, prior, atol=1e-12, rtol=0.)
        stress = CausalPortfolioBacktester(phase2_fee_rate=.001, phase2_kappa=.2, phase2_eta=.1)
        annual = {year: statistics(group) for year, group in saved.student.groupby(saved.index.year)}
        quarters = saved.student.add(1).resample("QE").prod()
        best = quarters.idxmax()
        keep = (saved.index.year != best.year) | (saved.index.quarter != best.quarter)
        old_only = (previous.weights > 0) & (current.weights == 0)
        comparison = saved[["student", "rule"]].assign(P20=prior)
        if args.policy != "P28" and (OUT / "P28_seed42").is_dir():
            base = pd.read_parquet(OUT / "P28_seed42" / f"{stage}_returns.parquet").student
            if not base.index.equals(prior.index):
                raise ValueError("Unmatched P28 clock")
            comparison["P28"] = base
        audit[stage] = {
            "student": statistics(saved.student), "rule": statistics(saved.rule),
            "untrained": statistics(saved.untrained), "P20": statistics(prior),
            "P20_attribution": attribution(prior, previous.weights),
            "P20_only_exposure": {"hours": int(old_only.sum()), "net_sum_pct": float(100 * prior[old_only].sum()),
                                  "gross_sum_pct": float(100 * previous.bar_gross_returns[old_only].sum())},
            "gross": statistics(current.bar_gross_returns),
            "annual_cost_pct": float(100 * current.bar_costs.mean() * 8760),
            "P20_annual_cost_pct": float(100 * previous.bar_costs.mean() * 8760),
            "annual": annual, "sortino_without_best_quarter": statistics(saved.student[keep])["sortino"],
            "intervals": block_intervals(comparison),
            "double_cost": statistics(stress.run_decisions(bars, weights, phase=2).bar_net_returns),
            "extra_hour": statistics(bt.run_decisions(bars, weights.shift(1, fill_value=0), phase=2).bar_net_returns),
            "annual_turnover": float(current.summary.avg_turnover * 8760),
            "P20_annual_turnover": float(previous.summary.avg_turnover * 8760),
        }
        if stage == "paper":
            folds = json.loads((folder / "metrics.json").read_text())["folds"]
            alternative = weights.copy()
            for fold, following in zip(folds, folds[1:]):
                if not fold["n_train"]:
                    start = pd.Timestamp(fold["test_from"]).normalize() + pd.Timedelta(hours=23)
                    end = pd.Timestamp(following["test_from"]).normalize() + pd.Timedelta(hours=23)
                    alternative.loc[(weights.index >= start) & (weights.index < end)] = 0.
            audit[stage]["cash_in_empty_training_fold"] = statistics(
                bt.run_decisions(bars, alternative, phase=2).bar_net_returns)
    audit["november"] = november(hist, later, folder, args.policy)
    (folder / "audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
