import numpy as np
import pandas as pd

from ts.baselines.audit_momentum_condition import statistics
from ts.baselines.momentum_condition_student import momentum


def test_threshold_does_not_use_future_prices_or_calendar_year():
    dates = pd.date_range("2018-02-01", periods=80, freq="D", tz="UTC")
    close = pd.Series(100 + 8*np.sin(np.arange(len(dates))/10), index=dates)
    close.iloc[55] = close.iloc[25]
    reference = momentum(close, 30)
    assert not reference.iloc[55]
    altered = close.copy()
    altered.iloc[56:] *= 10
    np.testing.assert_array_equal(momentum(altered, 30).iloc[:56], reference.iloc[:56])
    np.testing.assert_array_equal(momentum(close.shift(freq="13D"), 30), reference)


def test_independent_sortino_uses_all_observations_and_mdd_starts_at_initial_equity():
    r = np.array([-.05, .15, 0., 0.])
    got = statistics(r)
    assert np.isclose(got["sortino"], (.10/4)/(.05/2)*np.sqrt(8760))
    assert np.isclose(got["mdd_pct"], -5.)
