import numpy as np
import pandas as pd

from ts.baselines.conditional_budget import risk_budgets
from ts.baselines.trend_entry_student import FILTER_NAMES, RECOVERY_NAMES, entry_signals, entry_state, first_recovery


def test_exit_overrides_entry_and_persists():
    enter = pd.Series([False, True, False, True, False, True])
    leave = pd.Series([False, False, False, True, False, False])
    assert entry_state(enter, leave).tolist() == [False, True, True, False, False, True]


def test_signals_use_only_observed_market_history_and_no_calendar_dates():
    rng = np.random.default_rng(82)
    times = pd.date_range("2019-01-01 23:00", periods=750, tz="UTC", freq="D")
    prices = pd.Series(np.exp(rng.normal(.002, .024, len(times)).cumsum()), index=times)
    volume = pd.Series(rng.lognormal(size=len(times)), index=times)
    market = pd.DataFrame({i: prices.shift(i) for i in range(4)})
    reference = entry_signals(prices, volume, market)
    pd.testing.assert_frame_equal(reference.iloc[:560],
                                  entry_signals(prices.iloc[:560], volume.iloc[:560], market.iloc[:560]))
    shifted = entry_signals(prices.shift(freq="13D"), volume.shift(freq="13D"), market.shift(freq="13D"))
    pd.testing.assert_frame_equal(reference.shift(freq="13D"), shifted)
    assert reference.any().all()
    above_only = reference.drop(columns=["P34", *RECOVERY_NAMES, *FILTER_NAMES])
    assert not above_only[prices.le(prices.rolling(200).mean())].any().any()
    assert not reference.P34[prices.pct_change(30, fill_method=None).le(0)].any()
    assert reference.loc[reference.P37, list(RECOVERY_NAMES)].all().all()


def test_recovery_window_fires_once_and_expires():
    event = pd.Series([True, False, False, False, True, False, False, False])
    signal = pd.Series([True, False, True, True, True, False, False, True])
    assert first_recovery(event, signal, 2).tolist() == [False, False, True, False, False, False, False, False]


def test_risk_budgets_use_completed_hours_and_stay_in_spot_range():
    clock = pd.date_range("2018-01-01", periods=110 * 24, tz="UTC", freq="h")
    rng = np.random.default_rng(14)
    prices = pd.DataFrame({"close": np.exp(rng.normal(.0002, .004, len(clock)).cumsum())}, index=clock)
    whole = risk_budgets(prices)
    prefix = risk_budgets(prices.iloc[:100 * 24])
    pd.testing.assert_frame_equal(prefix, whole.loc[prefix.index])
    mature = whole.iloc[-15:]
    assert np.isfinite(mature).all().all() and (mature.ge(0) & mature.le(1)).all().all()
