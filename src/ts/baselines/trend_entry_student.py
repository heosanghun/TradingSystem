"""Train conditional BTC spot students with observable daily entry signals."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ts.baselines.momentum_condition_student import make_data, train_and_evaluate
from ts.baselines.conditional_budget import NAMES as BUDGET_NAMES, risk_budgets
from ts.baselines.breakout_student import daily_inputs
from ts.baselines.cycle_student import economic_labels
from ts.baselines.early_breakout_audit import policy_inputs
from ts.baselines.weekly_student import decisions_at, digest
from ts.config import PROJECT_ROOT
from ts.data.market_breadth import market_closes
from ts.eval.backtester import CausalPortfolioBacktester
from ts.eval.cost_model import compute_phase2_bar_base_cost_rates

OUT = PROJECT_ROOT / "runs/baselines/trend_entry"
NAMES = ("P21", "P22", "P23", "P24", "P25", "P26", "P27", "P28", "P29", "P30")
IMPULSE_NAMES = ("P31", "P32", "P33", "P34", "P35", "P36")
CONFIRM_NAMES = ("P37", "P38", "P39", "P40", "P41", "P42", "P43")
RECOVERY_NAMES = ("P44", "P45", "P46", "P47", "P48", "P49", "P50")
FILTER_NAMES = ("P57", "P58", "P59", "P60", "P61", "P62")


def entry_state(entry, exit_signal):
    state = pd.Series(np.nan, index=entry.index)
    state.loc[entry] = 1.
    state.loc[exit_signal] = 0.
    return state.ffill().eq(1.)


def first_recovery(event, trigger, expiry=None):
    """First confirming close after each new event, on a daily calendar."""
    episode = event.cumsum()
    age = event.groupby(episode).cumcount()
    valid = trigger & episode.gt(0) & age.gt(0)
    if expiry is not None:
        valid &= age.le(expiry)
    return valid & valid.groupby(episode).cumsum().eq(1)


def recent_event(event, days):
    return event.rolling(days, min_periods=1).max().eq(1)


def entry_signals(close, volume, market=None, early_choice=None):
    regime = close.gt(close.rolling(200).mean())
    month = np.log(close).diff(30)
    positive = month.gt(0)
    sigma = np.log(close).diff().rolling(30).std()
    previous = close.shift(1)
    new_high = close.gt(previous.rolling(20).max())
    breakout = entry_state(new_high, close.lt(previous.rolling(10).min()))
    slow = entry_state(close.gt(previous.rolling(55).max()), close.lt(previous.rolling(20).min()))
    volume = volume.reindex(close.index)
    avg_cost = (close * volume).rolling(30).sum().shift(1) / volume.rolling(30).sum().shift(1)
    week = close.rolling(7).mean()
    exit_month = ~regime | ~positive
    result = {
        "P21": entry_state(regime & month.gt(sigma), ~regime | month.lt(-sigma)),
        "P22": entry_state(regime & positive & breakout, exit_month),
        "P23": regime & positive & close.gt(avg_cost),
        "P25": ~exit_month & (~exit_month).shift(1, fill_value=False),
        "P26": regime & week.gt(week.shift(30)),
        "P27": ~exit_month & breakout,
        "P28": entry_state(~exit_month & slow, exit_month),
        "P29": entry_state(~exit_month & breakout & slow, exit_month | ~breakout),
        "P30": entry_state(~exit_month & new_high, exit_month | ~breakout),
    }
    if market is not None:
        moves = market.pct_change(30, fill_method=None).reindex(close.index)
        broad = moves.notna().all(axis=1) & moves.gt(0).sum(axis=1).ge(3)
        result["P24"] = entry_state(regime & positive & broad, exit_month)
    log_price = np.log(close)
    weekly_return = log_price.diff(7)
    weekly_risk = sigma * np.sqrt(7)
    impulse = weekly_return.gt(weekly_risk)
    base = result["P28"]
    confirmed = base & impulse
    retreat = log_price.shift(1).rolling(7).max() - log_price
    result.update({
        "P31": confirmed,
        "P32": entry_state(confirmed, ~base),
        "P33": entry_state(confirmed, ~base | weekly_return.le(0)),
        "P34": entry_state(slow & positive & impulse, ~positive | close.lt(previous.rolling(20).min())),
        "P35": entry_state(~exit_month & impulse, exit_month),
        "P36": entry_state(confirmed, ~base | retreat.gt(weekly_risk)),
    })
    confirms = {days: log_price.diff(days).gt(sigma * np.sqrt(days)) for days in (7, 14, 30)}
    votes = pd.DataFrame(confirms).sum(axis=1)
    previous_sigma = log_price.diff().rolling(30).std().shift(7)
    previous_volume = volume.rolling(30).mean().shift(7)
    confirmation = {
        "P37": weekly_return.gt(1.645 * weekly_risk),
        "P38": confirms[14],
        "P39": confirms[30],
        "P40": votes.ge(2),
        "P41": impulse & log_price.diff().rolling(7).std().gt(previous_sigma),
        "P42": impulse & volume.rolling(7).mean().gt(previous_volume),
    }
    if early_choice is not None:
        days, threshold = early_choice
        confirmation["P43"] = log_price.diff(days).gt(threshold * sigma * np.sqrt(days))
    for name, strong in confirmation.items():
        result[name] = entry_state(base & strong, ~base)
    daily_return = log_price.diff()
    shock = daily_return.lt(-2 * sigma.shift(1))
    new_low = close.lt(previous.rolling(20).min())
    rebound = first_recovery(shock, daily_return.gt(0), 7)
    bull_rebound = first_recovery(shock & regime.shift(1, fill_value=False), daily_return.gt(0), 7)
    reclaim = first_recovery(new_low, close.gt(previous.rolling(3).max()), 7)
    deep_recovery = first_recovery(new_low, close.gt(previous.rolling(10).max()))
    early_break = close.gt(previous.rolling(55).max()) & confirmation["P37"]
    pullback = ~exit_month & week.gt(close.shift(30)) & daily_return.lt(-sigma.shift(1))
    extra = {
        "P44": recent_event(shock, 3),
        "P45": recent_event(rebound, 3),
        "P46": recent_event(bull_rebound, 3),
        "P47": recent_event(reclaim, 7),
        "P48": entry_state(early_break, close.lt(close.rolling(20).mean())),
        "P49": recent_event(pullback, 3),
        "P50": entry_state(deep_recovery, close.lt(previous.rolling(5).min())),
    }
    for name, opportunity in extra.items():
        result[name] = result["P37"] | opportunity
    above_medium = close.gt(close.rolling(50).mean())
    recovered = extra["P50"]
    medium_recovery = above_medium & recovered
    result.update({
        "P57": result["P37"] | medium_recovery,
        "P58": result["P37"] | (regime & recovered),
        "P59": result["P32"] | medium_recovery,
        "P60": result["P40"] | medium_recovery,
        "P61": result["P37"] | entry_state(deep_recovery & above_medium, close.lt(previous.rolling(5).min())),
        "P62": result["P37"] | (positive & medium_recovery),
    })
    return pd.DataFrame(result)


def day_volume(bars):
    volume = bars.volume.resample("D").sum(min_count=1)
    return volume.shift(freq="23h")


def select_early(bars):
    close = bars.close[bars.index.hour == 23].asfreq("D")
    volume = day_volume(bars)
    budget = policy_inputs(bars).budget
    sample = bars.loc["2018-05-01":"2020-07-31"]
    thirds = np.array_split(np.arange(len(sample)), 3)
    scores = []
    for days in (7, 14, 30):
        for threshold in (1., 1.645, 2.):
            mask = entry_signals(close, volume, early_choice=(days, threshold)).P43
            chosen = budget.where(mask, 0.)
            ratios = []
            for rows in thirds:
                period = sample.iloc[rows]
                clock = period.index[period.index.hour == 23]
                decision = decisions_at(period.index, clock, chosen.loc[clock])
                result = CausalPortfolioBacktester().run_decisions(period, decision, phase=2)
                ratios.append(result.summary.sortino)
            scores.append(dict(days=days, threshold=threshold, thirds=ratios, median=float(np.median(ratios))))
    winner = max(scores, key=lambda row: row["median"])
    return (winner["days"], winner["threshold"]), scores


def main():
    parser = argparse.ArgumentParser()
    all_names = NAMES + IMPULSE_NAMES + CONFIRM_NAMES + RECOVERY_NAMES + BUDGET_NAMES + FILTER_NAMES
    parser.add_argument("--policies", nargs="+", choices=all_names, default=NAMES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(2)
    data = make_data()
    bars = pd.concat([data["hist"][data["later"].columns], data["later"]])
    close = data["close"]
    clock = close.loc[bars.index[0]:].index
    market = market_closes(clock)[0] if "P24" in args.policies else None
    full = pd.concat([data["old"], bars])
    volume = day_volume(full)
    choice = None
    if "P43" in args.policies:
        choice, scores = select_early(data["old"])
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "P43_early_choice.json").write_text(json.dumps(scores, indent=2, allow_nan=False))
    signals = entry_signals(close, volume, market, choice).loc[data["times"]]
    if set(args.policies).intersection(BUDGET_NAMES):
        budgets = risk_budgets(full)
        volatility = daily_inputs(bars)[0].volatility
        costs = pd.Series(compute_phase2_bar_base_cost_rates(bars), index=bars.index)
        costs = costs.reindex(volatility.index)
    for name in args.policies:
        active = signals["P37" if name in BUDGET_NAMES else name]
        inputs, y, budget = data, data["budget_y"], data["budget"]
        metadata = {"policy": name, "seed": args.seed, "source_sha256": digest(Path(__file__)),
                    "training_source_sha256": digest(Path(__file__).with_name("momentum_condition_student.py"))}
        if name == "P43":
            metadata["early_choice"] = choice
        if name in BUDGET_NAMES:
            target = budgets[name].reindex(volatility.index)
            budget = target.loc[data["times"]]
            if not np.isfinite(budget).all():
                raise ValueError("Missing risk state at a decision time")
            state = data["state"].clone()
            state[:, 2] = torch.tensor(budget.to_numpy(), dtype=state.dtype)
            inputs = dict(data, state=state)
            y = economic_labels(close.reindex(volatility.index), target, volatility, costs).loc[budget.index].to_numpy()
            metadata["budget_source_sha256"] = digest(Path(__file__).with_name("conditional_budget.py"))
        train_and_evaluate(inputs, active.to_numpy(), y, budget.where(active, 0.),
                           args.seed, OUT / f"{name}_seed{args.seed}", metadata)


if __name__ == "__main__":
    main()
