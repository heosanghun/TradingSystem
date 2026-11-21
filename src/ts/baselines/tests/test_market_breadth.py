import numpy as np
import pandas as pd

from ts.baselines.breadth_student import breadth_inputs, six_hour_inputs
from ts.baselines.cycle_student import allowed_dates, economic_labels, training_indices
from ts.baselines.signed_cycle_student import sell_labels, sell_states


def test_breadth_requires_a_complete_panel_and_counts_an_exact_tie():
    dates = pd.date_range("2020-01-01 23:00", periods=40, freq="D", tz="UTC")
    increasing = pd.Series(100 + np.arange(len(dates)), index=dates)
    panel = pd.DataFrame({"a": increasing, "b": increasing, "c": 10000 / increasing, "d": 10000 / increasing})
    features, state = breadth_inputs(increasing, panel)
    assert features.iloc[-1].breadth_30 == .5 and state.iloc[-1].slow == 1
    panel.loc[dates[-1], "c"] = np.nan
    features, state = breadth_inputs(increasing, panel)
    assert np.isnan(features.iloc[-1].breadth_30) and np.isnan(state.iloc[-1].slow)


def test_future_hourly_bars_cannot_change_a_closed_six_hour_breakout():
    dates = pd.date_range("2020-01-01", periods=24 * 80, freq="h", tz="UTC")
    bars = pd.DataFrame({"close": np.arange(len(dates)) + 100.}, index=dates)
    cut = 24 * 70
    prefix, signals = six_hour_inputs(bars.iloc[:cut])
    bars.iloc[cut:] *= .1
    full, later = six_hour_inputs(bars)
    np.testing.assert_array_equal(prefix, full.loc[prefix.index])
    np.testing.assert_array_equal(signals, later.loc[signals.index])
    assert signals.iloc[-1].sum() == 2


def test_cycle_training_uses_complete_forward_labels_only_on_permitted_dates():
    dates = pd.date_range("2019-01-01 23:00", periods=300, freq="D", tz="UTC")
    close = pd.Series(np.arange(len(dates)) + 100., index=dates)
    close.iloc[245] = 10.
    eligible = allowed_dates(close).fillna(False).to_numpy(dtype=bool)
    template = pd.Series(.5, index=dates)
    label = economic_labels(close, template, template, template * .002)
    cutoff = dates[260]
    train = training_indices(dates, dates[220], cutoff, eligible, label, True)
    assert 245 not in train and train[-1] == 252
    altered = close.copy()
    altered.loc[cutoff:] *= 10
    next_labels = economic_labels(altered, template, template, template * .002)
    np.testing.assert_array_equal(next_labels.iloc[train], label.iloc[train])


def test_sell_expert_learns_signed_outcomes_and_exits_on_an_upside_breakout():
    dates = pd.date_range("2020-01-01", periods=90, freq="D", tz="UTC")
    close = pd.Series(200. - np.arange(len(dates)), index=dates)
    budget = pd.Series(.5, index=dates)
    votes = sell_states(close, budget)
    assert votes.loc[dates[70], ["fast", "slow"]].sum() == 2
    label = sell_labels(close, votes, budget, budget * .002)
    assert -1 <= label.iloc[70] < -.5
    close.iloc[71] = 201
    after = sell_states(close, budget)
    assert after.loc[dates[71], ["fast", "slow"]].sum() == 0
    np.testing.assert_array_equal(after.iloc[:71], votes.iloc[:71])
