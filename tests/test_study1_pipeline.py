from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import torch

from ts.study1.attribution import compute_modality_group_shapley
from ts.study1.chart_renderer import CandlestickChartRenderer, normalize_chart_batch_to_tensor
from ts.study1.encoders import CrossModalStateFusion, MultimodalStateEncoder, TechnicalIndicatorCalculator
from ts.study1.ensemble import DynamicRegimeEnsemble
from ts.study1.per_buffer import SegmentTreePERBuffer, _SumMinSegmentTree
from ts.study1.ppo_expert import compute_regime_rewards
from ts.study1.regime_classifier import XGBoostRegimeClassifier
from ts.study1.regime_labeling import REGIME_BEAR, REGIME_BULL, REGIME_SIDEWAYS, TrendScanningLabeler


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
    assert attn.shape == (4, 2)  # 2-token attention over [Visual, Tech]
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


def test_uint8_lazy_chart_normalization_matches_float32_and_default_pretrained_signature() -> None:
    sig = inspect.signature(MultimodalStateEncoder.__init__)
    assert sig.parameters["pretrained_resnet"].default is True

    df = _make_synthetic_ohlcv(75)
    renderer = CandlestickChartRenderer(window_bars=30, image_size=(224, 224))
    charts_f32 = renderer.render_series_batch(df)
    charts_u8 = renderer.render_series_uint8(df)

    assert charts_u8.dtype == np.uint8
    assert charts_u8.shape == (75, 224, 224, 3)
    assert charts_u8.nbytes * 4 == charts_f32.nbytes

    norm_from_u8 = normalize_chart_batch_to_tensor(charts_u8[:12], torch.device("cpu")).numpy()
    assert np.allclose(norm_from_u8, charts_f32[:12], atol=1e-6)

    tech = TechnicalIndicatorCalculator(zscore_window=20).compute(df).to_numpy(dtype=np.float32)
    senti = np.zeros((75, 8), dtype=np.float32)
    senti[:, 1] = 0.25
    sub_idx = np.arange(15, 65)

    torch.manual_seed(777)
    enc = MultimodalStateEncoder(pretrained_resnet=False)
    enc.eval()

    st_f32, mod_f32 = enc.encode_dataset(
        charts_f32[sub_idx], tech[sub_idx], senti[sub_idx], batch_size=16, device=torch.device("cpu")
    )
    st_u8, mod_u8 = enc.encode_dataset(
        charts_u8, tech, senti, batch_size=16, device=torch.device("cpu"), indices=sub_idx
    )
    assert np.allclose(st_u8, st_f32, atol=1e-5)
    assert np.allclose(mod_u8, mod_f32, atol=1e-5)


def test_variant_a_modality_ablation_preserves_regime_and_ensemble_pipeline() -> None:
    df = _make_synthetic_ohlcv(90)
    charts_a = CandlestickChartRenderer(window_bars=20).render_series_uint8(df)
    charts_b = np.full_like(charts_a, 255)
    tech = TechnicalIndicatorCalculator(zscore_window=20).compute(df).to_numpy(dtype=np.float32)
    senti_a = np.zeros((90, 8), dtype=np.float32)
    senti_b = np.ones((90, 8), dtype=np.float32) * 3.0
    labels = np.array([i % 3 for i in range(60)], dtype=np.int64)
    tr_idx = np.arange(0, 60)
    te_idx = np.arange(60, 90)

    enc_tech = MultimodalStateEncoder(pretrained_resnet=False, modality_mode="tech_only")
    assert enc_tech.visual_encoder is None and enc_tech.senti_encoder is None
    enc_tech.pretrain_on_fold_regimes(
        charts_a, tech, senti_a, labels, epochs=1, batch_size=16, device=torch.device("cpu"), indices=tr_idx
    )
    st_te_1, _ = enc_tech.encode_dataset(charts_a, tech, senti_a, device=torch.device("cpu"), indices=te_idx)
    st_te_2, _ = enc_tech.encode_dataset(charts_b, tech, senti_b, device=torch.device("cpu"), indices=te_idx)
    assert st_te_1.shape == (30, 128)
    assert np.allclose(st_te_1, st_te_2, atol=1e-7)

    enc_vt = MultimodalStateEncoder(pretrained_resnet=False, modality_mode="visual_tech_no_senti")
    assert enc_vt.visual_encoder is not None and enc_vt.senti_encoder is None
    st_vt_1, _ = enc_vt.encode_dataset(charts_a, tech, senti_a, device=torch.device("cpu"), indices=te_idx)
    st_vt_2, _ = enc_vt.encode_dataset(charts_a, tech, senti_b, device=torch.device("cpu"), indices=te_idx)
    assert np.allclose(st_vt_1, st_vt_2, atol=1e-7)


def test_group_shapley_active_regime_chosen_logit_3seed_stability() -> None:
    rng = np.random.default_rng(2026)
    n_bars = 640
    concat_mods = rng.normal(0.0, 1.0, size=(n_bars, 224)).astype(np.float32)

    torch.manual_seed(99)
    fusion = CrossModalStateFusion(d_visual=128, d_tech=64, d_senti=32, d_model=128)
    ens = DynamicRegimeEnsemble(
        state_dim=128,
        experts_per_regime=5,
        rolling_sharpe_window_bars=24,
        ppo_kwargs={"device": torch.device("cpu")},
    )

    with torch.no_grad():
        st = fusion(
            torch.from_numpy(concat_mods[:, :128]),
            torch.from_numpy(concat_mods[:, 128:192]),
            torch.from_numpy(concat_mods[:, 192:]),
        ).numpy()

    active_regimes = np.array([i % 3 for i in range(n_bars)], dtype=np.int64)
    rets = rng.normal(0.0, 0.004, size=n_bars)
    costs = np.full(n_bars, 0.0005)
    _, chosen_actions, _ = ens.rollout_causal_policy(
        states=st,
        active_regimes=active_regimes,
        bar_asset_returns=rets,
        one_way_cost_rates=costs,
        temperature=10.0,
        num_experts_subset=5,
    )
    assert ens.last_rollout_expert_weights_ is not None

    shares = []
    for sd in (42, 43, 44):
        res = compute_modality_group_shapley(
            concat_modalities=concat_mods,
            fusion_module=fusion,
            ensemble=ens,
            active_regimes=active_regimes,
            chosen_actions=chosen_actions,
            expert_weights=ens.last_rollout_expert_weights_,
            num_experts_subset=5,
            n_background=256,
            n_explain=400,
            seed=sd,
            device=torch.device("cpu"),
        )
        assert np.isclose(res.visual_pct + res.tech_pct + res.sentiment_pct, 100.0, atol=1e-4)
        assert np.isclose(res.tech_plus_sentiment_pct, res.tech_pct + res.sentiment_pct, atol=1e-5)
        shares.append([res.visual_pct, res.tech_pct, res.sentiment_pct])

    shares_arr = np.asarray(shares, dtype=np.float64)
    seed_stds = np.std(shares_arr, axis=0, ddof=1)
    assert np.all(seed_stds <= 2.0), f"3-seed std exceeded 2%p: {seed_stds}"

    # Constant sentiment input must yield exact zero sentiment attribution
    const_mods = concat_mods.copy()
    const_mods[:, 192:] = const_mods[0:1, 192:]
    res_const = compute_modality_group_shapley(
        concat_modalities=const_mods,
        fusion_module=fusion,
        ensemble=ens,
        active_regimes=active_regimes,
        chosen_actions=chosen_actions,
        expert_weights=ens.last_rollout_expert_weights_,
        num_experts_subset=5,
        n_background=128,
        n_explain=128,
        seed=42,
        device=torch.device("cpu"),
    )
    assert abs(res_const.sentiment_pct) < 1e-6


def test_vectorized_per_rewards_and_ppo_convergence_logging(tmp_path) -> None:
    import json
    from collections import deque

    rng = np.random.default_rng(31415)

    # 1. Segment tree bulk build, batch prefix lookup, and duplicate-index priority updates
    td_init = rng.normal(0.0, 1.5, size=137)
    per_vec = SegmentTreePERBuffer(alpha=0.6, eps=1e-6)
    per_vec.build_from_td_errors(td_init)

    tree_ref = _SumMinSegmentTree(len(td_init))
    for i, val in enumerate(np.power(np.abs(td_init) + 1e-6, 0.6)):
        tree_ref.set(i, float(val))

    assert np.allclose(per_vec.tree.sum_tree, tree_ref.sum_tree, atol=1e-12)
    assert np.isclose(per_vec.tree.min_positive(), tree_ref.min_positive(), atol=1e-12)

    masses = rng.uniform(0.0, tree_ref.total_sum(), size=64)
    batch_idx = per_vec.tree.find_prefixsum_idx_batch(masses)
    scalar_idx = np.array([tree_ref.find_prefixsum_idx(float(m)) for m in masses], dtype=np.int64)
    assert np.array_equal(batch_idx, scalar_idx)

    dup_indices = np.array([5, 12, 5, 100, 12, 0, 136], dtype=np.int64)
    new_tds = np.array([0.1, 0.9, 2.5, 0.4, 0.05, 1.1, 0.3], dtype=np.float64)
    per_vec.update_priorities(dup_indices, new_tds)
    for idx_i, td_i in zip(dup_indices, new_tds):
        tree_ref.set(int(idx_i), float((abs(td_i) + 1e-6) ** 0.6))
    assert np.allclose(per_vec.tree.sum_tree, tree_ref.sum_tree, atol=1e-12)
    assert np.isclose(per_vec.tree.min_positive(), tree_ref.min_positive(), atol=1e-12)

    # Zero-TD outlier must not collapse importance weights toward zero at beta=1.0
    per_vec.update_priorities(np.array([0], dtype=np.int64), np.array([0.0], dtype=np.float64))
    _, is_w_beta1 = per_vec.sample(batch_size=64, beta=1.0, rng=rng)
    assert np.isclose(float(np.mean(is_w_beta1)), 1.0, atol=1e-5)
    assert float(np.min(is_w_beta1)) > 0.05

    # 2. Vectorized regime rewards vs step-by-step reference
    n_steps = 96
    mkt_ret = rng.normal(0.0001, 0.008, size=n_steps)
    weights = rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=n_steps)
    costs = rng.uniform(0.0004, 0.0012, size=n_steps)

    for reg_id in (REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS):
        rew_vec, net_vec = compute_regime_rewards(
            regime_id=reg_id,
            bar_asset_returns=mkt_ret,
            chosen_weights=weights,
            one_way_cost_rates=costs,
            bear_window=24,
            bear_dd_floor=1e-4,
            bear_cost_penalty=1.0,
            sideways_turnover_penalty=0.002,
        )
        rew_ref = np.zeros(n_steps, dtype=np.float32)
        net_ref = np.zeros(n_steps, dtype=np.float32)
        buf = deque(maxlen=24)
        prev_w = 0.0
        for t in range(n_steps):
            dw = abs(float(weights[t]) - prev_w)
            c_t = float(costs[t]) * dw
            nr = float(weights[t]) * float(mkt_ret[t]) - c_t
            net_ref[t] = nr
            buf.append(nr)
            if reg_id == REGIME_BULL:
                rew_ref[t] = float(np.log(max(1.0 + nr, 1e-6)))
            elif reg_id == REGIME_BEAR:
                arr = np.asarray(buf, dtype=np.float64)
                dd = float(np.sqrt(np.mean(np.minimum(0.0, arr) ** 2)))
                st = float(np.mean(arr)) / max(dd, 1e-4)
                rew_ref[t] = float(np.clip(st, -10.0, 10.0) - 1.0 * c_t)
            else:
                rew_ref[t] = float(nr - 0.002 * dw)
            prev_w = float(weights[t])
        assert np.allclose(net_vec, net_ref, atol=1e-6)
        assert np.allclose(rew_vec, rew_ref, atol=1e-5)

    # 3. Telemetry logging per fold, regime, and seed
    states = rng.normal(0.0, 1.0, size=(90, 32)).astype(np.float32)
    regimes = np.array([i % 3 for i in range(90)], dtype=np.int64)
    ens = DynamicRegimeEnsemble(
        state_dim=32,
        experts_per_regime=2,
        ppo_kwargs={"device": torch.device("cpu"), "minibatch_size": 32, "ppo_epochs": 2},
    )
    ens.train_all_pools(
        train_states=states,
        train_regimes=regimes,
        next_bar_returns=mkt_ret[:90],
        cost_rates=costs[:90],
        total_rollouts=3,
        log_dir=tmp_path,
        fold_id=1,
        tag_prefix="unit",
    )
    saved = sorted(tmp_path.glob("unit_fold1_regime*_seed*.json"))
    assert len(saved) == 6
    sample_payload = json.loads(saved[0].read_text(encoding="utf-8"))
    assert sample_payload["fold_id"] == 1
    assert sample_payload["total_rollouts"] == 3
    assert len(sample_payload["rollout_logs"]) == 3
    required_keys = {
        "rollout",
        "mean_reward",
        "policy_entropy",
        "mean_turnover",
        "greedy_mean_turnover",
        "value_loss",
        "actor_loss",
        "greedy_action_dist",
    }
    assert required_keys.issubset(sample_payload["rollout_logs"][0].keys())
