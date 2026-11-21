import numpy as np
import pandas as pd
import torch

from ts.baselines.semivariance_student import cohort, path_labels


def test_same_terminal_return_with_larger_downside_path_lowers_training_label():
    ix = pd.date_range("2017-01-01", periods=24, freq="D", tz="UTC")
    close = pd.Series(np.linspace(80, 90, len(ix)), index=ix)
    weight, vol, fee = (pd.Series(k, index=ix) for k in (.5, .4, .001))
    low = path_labels(close, weight, vol, fee, vol*0)
    high = path_labels(close, weight, vol, fee, vol*.02)
    assert (high.iloc[:-7] < low.iloc[:-7]).all() and high.iloc[-7:].isna().all()


def test_decision_inputs_and_completed_labels_invariant_to_future_and_calendar():
    ix = pd.date_range("2018-01-01", periods=24*275, freq="h", tz="UTC")
    seq = np.arange(len(ix))
    close = np.exp(5 + .00004*seq + .08*np.sin(seq/90))
    bars = pd.DataFrame(dict(open=close, high=close*1.01, low=close*.99, close=close,
                             volume=1000+np.sin(seq/48)), index=ix)
    t, x, state, y, eligible, rule = cohort(bars, bars.close, 11, True)
    cutoff = ix[-24*30]
    new = bars.copy()
    factors = 1.25 + .2*np.sin(seq[ix >= cutoff])
    new.loc[ix >= cutoff, ["open", "high", "low", "close"]] *= factors[:, None]
    u, z, other, after, ok, signal = cohort(new, new.close, 11, True)
    assert t.equals(u)
    prefix = t < cutoff
    torch.testing.assert_close(x[prefix], z[prefix], atol=0., rtol=0.)
    torch.testing.assert_close(state[prefix], other[prefix], atol=0., rtol=0.)
    np.testing.assert_array_equal(eligible[prefix], ok[prefix])
    np.testing.assert_array_equal(rule[prefix], signal[prefix])
    complete = t + pd.Timedelta(days=7) < cutoff
    np.testing.assert_array_equal(y[complete], after[complete])
    shifted = bars.shift(freq="13D")
    moved, z, other, after, ok, signal = cohort(shifted, shifted.close, 11, True)
    assert (moved == t + pd.Timedelta(days=13)).all()
    torch.testing.assert_close(z, x, atol=0., rtol=0.)
    torch.testing.assert_close(other, state, atol=0., rtol=0.)
    np.testing.assert_array_equal(ok, eligible)
    np.testing.assert_array_equal(after, y)
