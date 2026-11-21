import numpy as np
import pandas as pd

from ts.baselines.breakout_student import daily_inputs, windows
from ts.baselines.early_breakout_audit import policy_inputs, risk_policy
from ts.baselines.panel_student import utility_actions


def test_today_can_break_yesterdays_range_and_future_edits_do_not_move_that_signal():
    times = pd.date_range("2020-01-01", periods=24*100, freq="h", tz="UTC")
    i = np.arange(len(times))
    close = 100 + np.sin(i/31)
    start = 80 * 24
    close[start:start + 24] = 120
    bars = pd.DataFrame({"close": close, "volume": np.ones(len(i))}, index=times)
    observed = bars.iloc[:start + 24]
    f, rule = daily_inputs(observed)
    assert f.state.iloc[-1] == 1 and rule.breakout.iloc[-1] > 0
    bars.loc[bars.index > observed.index[-1], :] *= 3
    g, rule_after = daily_inputs(bars)
    np.testing.assert_array_equal(rule, rule_after.loc[rule.index])
    dates, x = windows(f)
    full_dates, full_x = windows(g)
    np.testing.assert_array_equal(x, full_x[full_dates <= dates[-1]])


def test_utility_labels_require_all_seven_future_days_and_charge_round_trip_cost():
    index = pd.date_range("2020-01-01", periods=12 * 24, freq="h", tz="UTC")
    close = np.full(len(index), 100.)
    bars = pd.DataFrame({"close": close, "high": close, "low": close}, index=index)
    times = index[index.hour == 23]
    actions, valid = utility_actions(bars, times)
    assert valid.sum() == 5 and (actions[valid] == 2).all()
    bars *= np.exp(np.arange(len(index))[:, None] * .0002)
    actions, valid = utility_actions(bars, times)
    assert (actions[valid] == 4).all()


def test_asymmetric_exit_uses_previous_daily_closes_and_keeps_its_history():
    times = pd.date_range("2020-01-01", periods=24*100, freq="h", tz="UTC")
    day = np.arange(len(times)) // 24
    close = (100 + day).astype(float)
    close[day >= 80] = 20
    bars = pd.DataFrame({"close": close, "volume": 1.}, index=times)
    states = policy_inputs(bars)
    assert (states.iloc[79][["fast", "slow"]] == 1).all()
    assert (states.iloc[80][["fast", "slow"]] == 0).all()
    prefix = risk_policy(bars.iloc[:80*24])
    np.testing.assert_array_equal(prefix, risk_policy(bars).loc[prefix.index])
