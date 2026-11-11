from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ts.study1.chart_renderer import CandlestickChartRenderer
from ts.study1.encoders import CrossModalStateFusion, MultimodalStateEncoder, TechnicalIndicatorCalculator
from ts.study1.ensemble import DynamicRegimeEnsemble
from ts.study1.regime_classifier import XGBoostRegimeClassifier
from ts.study1.regime_labeling import REGIME_BULL, REGIME_SIDEWAYS, TrendScanningLabeler


def _make_synthetic_ohlcv(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    idx = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
    close = 40000.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.006, size=n)))
    open_ = np.roll(close, 1)
    open_[0] = close[0] * 0.999
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0005, 0.004, size=n))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.0005, 0.004, size=n))
    vol = rng.uniform(100.0, 2000.0, size=n)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
            "taker_buy_base_vol": 0.52 * vol,
        },
        index=idx,
    )


def test_chart_renderer_and_2token_cross_modal_attention() -> None:
    df = _make_synthetic_ohlcv(80)
    renderer = CandlestickChartRenderer(window_bars=60, image_size=(224, 224))
    imgs = renderer.render_series_batch(df.iloc[:5])
    assert imgs.shape == (5, 3, 224, 224)

    fusion = CrossModalStateFusion(d_visual=128, d_tech=64, d_senti=32, d_model=128)
    f_vis = torch.randn(4, 128)
    f_tech = torch.randn(4, 64)
    f_senti = torch.randn(4, 32)
    s_t, attn = fusion.forward_with_weights(f_vis, f_tech, f_senti)
    assert s_t.shape == (4, 128)
    assert attn.shape == (4, 2)  # Genuine 2-token attention over [Visual, Tech]!
    assert torch.allclose(attn.sum(dim=-1), torch.ones(4), atol=1e-5)
    assert not torch.allclose(attn[:, 0], torch.ones(4))


def test_trend_scanning_purging_and_hysteresis_gate() -> None:
    df = _make_synthetic_ohlcv(260)
    labeler = TrendScanningLabeler(horizon_min=24, horizon_max=60, horizon_step=12, t_threshold=1.8)
    labels = labeler.fit_label_train_fold(df["close"])
    assert len(labels) == 260
    # Trailing 60 bars must be strictly purged (-1) so test fold is never leaked
    assert np.all(labels[-60:] == -1)
    assert np.all(labels[:-60] >= 0)

    # Hysteresis gate theta = 0.60 check
    probs = np.array(
        [
            [0.55, 0.25, 0.20],  # max = 0.55 < 0.60 -> stays initial Sideways (2)
            [0.65, 0.20, 0.15],  # max = 0.65 >= 0.60 -> switches to Bull (0)
            [0.30, 0.58, 0.12],  # max = 0.58 < 0.60 -> stays Bull (0)
        ],
        dtype=np.float32,
    )
    regimes = XGBoostRegimeClassifier.apply_hysteresis_gate(probs, threshold=0.60, initial_regime=REGIME_SIDEWAYS)
    assert regimes.tolist() == [REGIME_SIDEWAYS, REGIME_BULL, REGIME_BULL]


def test_study1_encoder_xgb_and_dynamic_ensemble_rollout() -> None:
    df = _make_synthetic_ohlcv(120)
    tech = TechnicalIndicatorCalculator(zscore_window=30).compute(df).to_numpy(dtype=np.float32)
    charts = CandlestickChartRenderer(window_bars=20).render_series_batch(df[:40])
    senti = np.zeros((40, 8), dtype=np.float32)
    senti[:, 2] = 1.0
    labels = np.array([i % 3 for i in range(40)], dtype=np.int64)

    enc = MultimodalStateEncoder(pretrained_resnet=False)
    losses = enc.pretrain_on_fold_regimes(charts, tech[:40], senti, labels, epochs=1, batch_size=16, device=torch.device("cpu"))
    assert len(losses) == 1
    states, mods = enc.encode_dataset(charts, tech[:40], senti, batch_size=16, device=torch.device("cpu"))
    assert states.shape == (40, 128)
    assert mods.shape == (40, 224)

    clf = XGBoostRegimeClassifier(n_estimators=10, max_depth=3).fit(states[:30], labels[:30])
    pred_r, _ = clf.predict_regimes_causal(states[30:])
    assert pred_r.shape == (10,)

    ens = DynamicRegimeEnsemble(
        state_dim=128,
        experts_per_regime=3,
        rolling_sharpe_window_bars=24,
        ppo_kwargs={"device": torch.device("cpu"), "minibatch_size": 16, "ppo_epochs": 1},
    )
    rets = np.random.default_rng(1).normal(0, 0.005, size=40)
    costs = np.full(40, 0.0005)
    ens.train_all_pools(states[:30], labels[:30], rets[:30], costs[:30], total_rollouts=1)

    probs_t10, acts_t10, w_t10 = ens.rollout_causal_policy(
        states=states[30:],
        active_regimes=pred_r,
        bar_asset_returns=rets[30:],
        one_way_cost_rates=costs[30:],
        temperature=10.0,
        num_experts_subset=3,
        warmup_states=states[:30],
        warmup_returns=rets[:30],
        warmup_costs=costs[:30],
    )
    assert probs_t10.shape == (10, 5)
    assert acts_t10.shape == (10,)
    assert np.all(np.isin(w_t10, [-1.0, -0.5, 0.0, 0.5, 1.0]))
