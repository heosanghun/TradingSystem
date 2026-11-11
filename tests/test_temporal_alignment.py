from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ts.data.lob_processor import LOBWindowProcessor
from ts.data.news_fetcher import PointInTimeNewsStore
from ts.data.temporal_guard import (
    TemporalLeakageError,
    assert_no_future_leakage,
    assert_purged_train_horizon,
    assert_strictly_before,
)
from ts.eval.walk_forward import generate_phase2_walk_forward_folds


def test_temporal_guards_reject_leakage() -> None:
    t1 = pd.Timestamp("2022-05-10T12:00:00Z")
    t2 = pd.Timestamp("2022-05-10T12:00:00Z")
    with pytest.raises(TemporalLeakageError):
        assert_strictly_before(t1, t2, context="same timestamp")

    news_ts = pd.DatetimeIndex(["2022-05-10T11:00:00Z", "2022-05-10T12:00:00Z"])
    with pytest.raises(TemporalLeakageError):
        assert_no_future_leakage(news_ts, t1, strict_less=True)

    with pytest.raises(TemporalLeakageError):
        assert_purged_train_horizon(
            last_labeled_bar_idx=900,
            forward_horizon_bars=120,
            train_fold_end_idx_exclusive=1000,
        )


def test_walk_forward_9_folds_exact_boundaries() -> None:
    idx = pd.date_range("2020-08-01", "2023-12-31 23:00:00", freq="1h", tz="UTC")
    folds = generate_phase2_walk_forward_folds(idx)
    assert len(folds) == 9
    assert folds[0].train_start == pd.Timestamp("2020-10-01T00:00:00Z")
    assert folds[0].test_start == pd.Timestamp("2021-10-01T00:00:00Z")
    assert folds[-1].test_end == pd.Timestamp("2023-12-31T23:59:59Z")

    # Verify disjoint contiguous OOS coverage across all 9 folds
    seen = set()
    for f in folds:
        assert idx[f.train_indices[-1]] < idx[f.test_indices[0]]
        overlap = seen.intersection(set(f.test_indices.tolist()))
        assert len(overlap) == 0
        seen.update(f.test_indices.tolist())


def test_news_store_rejects_fake_placeholders_and_filters_causally() -> None:
    fake_df = pd.DataFrame(
        {
            "published_at": ["2022-05-01T01:00:00Z"],
            "headline": ["Crypto News Article 1"],
        }
    )
    with pytest.raises(ValueError, match="synthetic"):
        PointInTimeNewsStore(fake_df)

    real_df = pd.DataFrame(
        {
            "published_at": [
                "2022-05-01T00:30:00Z",
                "2022-05-01T00:30:00Z",  # duplicate
                "2022-05-01T01:00:00Z",  # exact boundary -> must be excluded from 01:00 query!
                "2022-05-01T02:15:00Z",
            ],
            "headline": [
                "Federal Reserve signals 50 bps rate hike ahead of May meeting",
                "Federal Reserve signals 50 bps rate hike ahead of May meeting",
                "Bitcoin spot volume surges during Asian session",
                "US Treasury yields climb above three percent",
            ],
            "logit_neg": [1.5, 1.5, -0.5, 1.2],
            "logit_neu": [0.2, 0.2, 0.4, 0.3],
            "logit_pos": [-1.0, -1.0, 1.4, -0.8],
        }
    )
    store = PointInTimeNewsStore(real_df)
    assert len(store.df) == 3  # deduplicated

    q = store.query_before(pd.Timestamp("2022-05-01T01:00:00Z"), lookback_hours=24.0)
    assert len(q) == 1
    assert "Federal Reserve" in q.iloc[0]["headline"]

    h_idx = pd.date_range("2022-05-01T00:00:00Z", periods=4, freq="1h", tz="UTC")
    senti_summary = store.build_hourly_sentiment_summary(h_idx, ewma_span_bars=24)
    assert senti_summary.shape == (4, 8)
    assert not np.isnan(senti_summary.to_numpy()).any()


def test_lob_window_processor_l2_and_subbar_shapes() -> None:
    rng = np.random.default_rng(7)
    n_ticks = 250
    mid = 30000.0 + np.cumsum(rng.normal(0, 2.0, size=n_ticks))
    bid_p = np.column_stack([mid - (i + 1) * 0.5 for i in range(10)])
    ask_p = np.column_stack([mid + (i + 1) * 0.5 for i in range(10)])
    bid_v = rng.uniform(0.1, 5.0, size=(n_ticks, 10))
    ask_v = rng.uniform(0.1, 5.0, size=(n_ticks, 10))
    tr_v = rng.uniform(0.01, 2.0, size=n_ticks)
    tr_s = rng.choice([-1.0, 0.0, 1.0], size=n_ticks)

    proc = LOBWindowProcessor(seq_len=100, num_features=40, windows_per_bar=10)
    feats = proc.compute_40_lob_features(bid_p, bid_v, ask_p, ask_v, tr_v, tr_s)
    assert feats.shape == (n_ticks, 40)

    wins = proc.sample_bar_windows_for_distillation(feats)
    assert len(wins) == 10
    assert wins[0].shape == (40, 100)
    assert np.allclose(np.mean(wins[0], axis=1), 0.0, atol=1e-4)
