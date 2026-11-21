import numpy as np
import pandas as pd

from ts.baselines.weekly_student import WEEK, decision_windows, train_rows, weekly_inputs
from ts.data.forward_labels import within_split_next_bar


def test_labels_inside_split_even_across_held_out_blocks():
    rows = np.array([1, 2, 3, 7, 8, 9, 10])
    np.testing.assert_array_equal(within_split_next_bar(rows), [1, 2, 7, 8, 9])
    times = pd.date_range("2023-12-01 23:00", periods=40, freq="D", tz="UTC")
    start = pd.Timestamp("2024-01-01", tz="UTC")
    valid = np.ones(len(times), dtype=bool)
    valid[10] = False
    selected = train_rows(times, times[0], start, valid)
    assert 10 not in selected
    assert (times[selected] + WEEK < start).all()
    assert times[selected[-1]] + WEEK == start - pd.Timedelta(hours=1)


def test_future_ohlcv_cannot_change_a_decision_window():
    clock = pd.date_range("2023-04-01", periods=24*110, freq="h", tz="UTC")
    ix = np.arange(len(clock))
    close = np.exp(6 + .1*np.sin(ix/11) + .01*np.cos(ix/30))
    bars = pd.DataFrame({"close":close, "high":1.01*close, "low":.99*close,
                         "volume":1000+np.sin(ix)}, index=clock)
    end = 24*100
    subset, prefix = decision_windows(weekly_inputs(bars.iloc[:end])[0])
    bars.iloc[end:] *= 20
    full, windows = decision_windows(weekly_inputs(bars)[0])
    np.testing.assert_array_equal(subset, full[full < end])
    np.testing.assert_array_equal(prefix, windows[full < end])
