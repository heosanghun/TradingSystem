"""
End-to-End Integration Test of the Complete `fig1` Overall System Architecture.

Verifies that all 18 architectural components in `doc/논문수정 가이드/4_fig1/fig1_3_system_overview_academic.png`
execute seamlessly in sequence:
  - Inputs ① (OHLCV), ② (Raw News Headlines -> FinBERT 3-class logits -> 24h EWMA), ③ (Multi-Asset Returns), ④ (40x100 LOB)
  - Offline System 2 Teacher:
      * Study 1: Candlestick 224x224x3 -> ResNet-18+CBAM(128) + Tech(64) + Senti(32) -> CrossAttention(128)
                 -> Purged TrendScanning + XGBoost(theta=0.60) -> 3-Regime PPO+PER Boltzmann Ensemble
      * Study 2: Granger + KDE TE -> Edge Verifier -> Spectral Hypergraph H_t -> Grounded 3-Round Debate
                 with CitationVerifier -> Meta PM (W_final, C_t)
  - Teacher Signal Conversion: P_teacher^tau (sigma_w=0.25, tau=2.0) + FinBERT [CLS](768) + 10-window broadcast
  - Offline Study 3 Reasoning Distillation: L_total = 0.4 L_task + 0.4 L_policy + 0.2 L_reasoning
  - Online System 1 Execution: `Student1DCNN.act_online(lob_40x100)` outputting 5 discrete actions
    `(-1.0, -0.5, 0.0, +0.5, +1.0)` with ZERO Teacher / LLM / FinBERT / projection-head calls!
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ts.data.lob_processor import LOBWindowProcessor
from ts.data.news_fetcher import PointInTimeNewsStore
from ts.eval.backtester import CausalPortfolioBacktester
from ts.eval.cost_model import compute_phase2_bar_base_cost_rates
from ts.study1.chart_renderer import CandlestickChartRenderer
from ts.study1.encoders import MultimodalStateEncoder, TechnicalIndicatorCalculator
from ts.study1.ensemble import DynamicRegimeEnsemble
from ts.study1.ppo_expert import ACTION_WEIGHTS
from ts.study1.regime_classifier import XGBoostRegimeClassifier
from ts.study1.regime_labeling import REGIME_NAMES, TrendScanningLabeler
from ts.study2.causality import discover_candidate_edges
from ts.study2.citation_verifier import CitationVerifier
from ts.study2.debate import GroundedDebateEngine
from ts.study2.hypergraph import build_daily_hypergraph
from ts.study2.llm_client import LLMClient
from ts.study3.distillation import StudentDistillationTrainer
from ts.study3.teacher_signal import FinBERTReasoningEncoder, build_aligned_distillation_dataset


def test_complete_fig1_end_to_end_dataflow() -> None:
    rng = np.random.default_rng(20261112)
    n_bars = 96
    idx = pd.date_range("2022-05-01 00:00:00", periods=n_bars, freq="1h", tz="UTC")

    # -------------------------------------------------------------------------
    # 1. Inputs ①, ②, ③, ④
    # -------------------------------------------------------------------------
    # Input ①: BTC/USDT 1h OHLCV
    log_ret = rng.normal(0.0002, 0.012, size=n_bars)
    close = 30000.0 * np.exp(np.cumsum(log_ret))
    open_ = np.roll(close, 1)
    open_[0] = close[0] * 0.999
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.001, 0.008, size=n_bars))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.001, 0.008, size=n_bars))
    vol = rng.uniform(100.0, 1200.0, size=n_bars)
    df_ohlcv = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )

    # Input ②: Point-in-time raw headlines (no precomputed logits -> auto-scored!)
    news_df = pd.DataFrame(
        {
            "published_at": [idx[10] + pd.Timedelta(minutes=15), idx[60] + pd.Timedelta(minutes=25)],
            "headline": [
                "Federal Reserve signals tighter monetary liquidity across risk assets",
                "Institutional spot demand stabilizes Bitcoin order book depth",
            ],
            "source": ["Reuters", "CoinDesk"],
        }
    )
    news_store = PointInTimeNewsStore(news_df)
    scored_cnt = news_store.populate_missing_logits_with_finbert(offline_hash_mode=True)
    assert scored_cnt == 2

    # Input ③: Cross-asset & macro causal return matrix (lagged driver -> BTCUSDT)
    driver = rng.normal(0.0, 0.015, size=n_bars)
    btc_driven = np.roll(driver, 1) * 0.65 + rng.normal(0.0, 0.006, size=n_bars)
    eth_driven = np.roll(driver, 2) * 0.50 + rng.normal(0.0, 0.007, size=n_bars)
    returns_multi = pd.DataFrame(
        {
            "BTCUSDT": btc_driven,
            "ETHUSDT": eth_driven,
            "DXY_PROXY": driver,
            "NDX_PROXY": -0.5 * driver + rng.normal(0.0, 0.008, size=n_bars),
        },
        index=idx,
    )

    # Input ④: 40 features x 100 ticks LOB / Microstructure windows (10 windows per train bar, 1 online per test bar)
    lob_proc = LOBWindowProcessor(seq_len=100, num_features=40, windows_per_bar=10)
    per_bar_lob_wins = []
    online_lob_wins = np.empty((n_bars, 40, 100), dtype=np.float32)
    for t in range(n_bars):
        w_t, _ = lob_proc.extract_from_subbar_klines(df_ohlcv, t)
        online_lob_wins[t] = w_t
        per_bar_lob_wins.append([w_t for _ in range(10)])

    tr_slice = np.arange(0, 72)
    te_slice = np.arange(72, n_bars)

    # -------------------------------------------------------------------------
    # 2. Offline System 2 Teacher — Study 1 (Multimodal Regime Ensemble RL)
    # -------------------------------------------------------------------------
    charts = CandlestickChartRenderer(window_bars=60, image_size=224).render_series_batch(df_ohlcv)
    tech_x = TechnicalIndicatorCalculator(zscore_window=30).compute(df_ohlcv).to_numpy(dtype=np.float32)
    senti_x = news_store.build_hourly_sentiment_summary(idx, ewma_span_bars=24).to_numpy(dtype=np.float32)

    tr_labels = TrendScanningLabeler(horizon_min=6, horizon_max=24, t_threshold=1.5).fit_label_train_fold(
        df_ohlcv["close"].iloc[tr_slice]
    )
    encoder = MultimodalStateEncoder(pretrained_resnet=False)
    encoder.pretrain_on_fold_regimes(
        charts[tr_slice], tech_x[tr_slice], senti_x[tr_slice], tr_labels, epochs=1, batch_size=32, device=torch.device("cpu")
    )
    tr_states, _ = encoder.encode_dataset(charts[tr_slice], tech_x[tr_slice], senti_x[tr_slice], device=torch.device("cpu"))
    te_states, _ = encoder.encode_dataset(charts[te_slice], tech_x[te_slice], senti_x[te_slice], device=torch.device("cpu"))
    assert tr_states.shape == (len(tr_slice), 128)
    assert te_states.shape == (len(te_slice), 128)

    xgb_gate = XGBoostRegimeClassifier(n_estimators=20, max_depth=3, switch_threshold=0.60)
    xgb_gate.fit(tr_states, tr_labels)
    tr_regimes, _ = xgb_gate.predict_regimes_causal(tr_states)

    next_ret = np.zeros(n_bars, dtype=np.float64)
    next_ret[:-1] = (close[1:] / close[:-1]) - 1.0
    base_costs = compute_phase2_bar_base_cost_rates(df_ohlcv)

    ensemble = DynamicRegimeEnsemble(
        state_dim=128,
        experts_per_regime=3,
        rolling_sharpe_window_bars=48,
        ppo_kwargs={"device": torch.device("cpu")},
    )
    ensemble.train_all_pools(tr_states, tr_regimes, next_ret[tr_slice], base_costs[tr_slice], total_rollouts=1)
    tr_ens_probs, _, _ = ensemble.rollout_causal_policy(
        states=tr_states,
        active_regimes=tr_regimes,
        bar_asset_returns=next_ret[tr_slice],
        one_way_cost_rates=base_costs[tr_slice],
        temperature=10.0,
    )
    assert tr_ens_probs.shape == (len(tr_slice), 5)

    # -------------------------------------------------------------------------
    # 3. Offline System 2 Teacher — Study 2 (Dynamic Hypergraph + Grounded Debate)
    # -------------------------------------------------------------------------
    llm = LLMClient(backend="deterministic_econometric")
    verifier = CitationVerifier(te_tolerance=0.05)
    debate = GroundedDebateEngine(llm_client=llm, citation_verifier=verifier, max_rounds=3)

    cands = discover_candidate_edges(
        returns_multi.iloc[tr_slice], granger_lag=4, granger_p_threshold=0.25, te_lag=1, te_top_quantile=0.50
    )
    hg = build_daily_hypergraph(
        nodes=list(returns_multi.columns),
        candidates=cands,
        llm_client=llm,
        news_context=news_store.format_headlines_for_llm(idx[71]),
    )

    w_finals_tr = np.zeros(len(tr_slice), dtype=np.float32)
    consensus_tr = []
    for i_pos, bar_i in enumerate(tr_slice):
        out = debate.run_bar_debate(
            hypergraph=hg,
            news_headlines_text=news_store.format_headlines_for_llm(idx[bar_i] + pd.Timedelta(hours=1)),
            study1_regime_name=REGIME_NAMES[int(tr_regimes[i_pos])],
            study1_probs=tr_ens_probs[i_pos],
            target_asset="BTCUSDT",
        )
        w_finals_tr[i_pos] = out.w_final
        consensus_tr.append(out.consensus_text)

    assert np.all((w_finals_tr >= -1.0) & (w_finals_tr <= 1.0))
    assert verifier.stats.citation_match_rate == 1.0

    # -------------------------------------------------------------------------
    # 4. Offline Study 3 Reasoning Distillation (Teacher -> Student1DCNN)
    # -------------------------------------------------------------------------
    finbert = FinBERTReasoningEncoder(offline_hash_mode=True)
    distill_pack = build_aligned_distillation_dataset(
        per_bar_lob_windows=[per_bar_lob_wins[i] for i in tr_slice],
        w_final_per_bar=w_finals_tr,
        consensus_texts_per_bar=consensus_tr,
        next_bar_returns=next_ret[tr_slice],
        one_way_cost_rates=base_costs[tr_slice],
        finbert_encoder=finbert,
    )
    assert distill_pack.lob_windows.shape == (len(tr_slice) * 10, 40, 100)
    assert distill_pack.teacher_reasoning.shape == (len(tr_slice) * 10, 768)

    trainer = StudentDistillationTrainer(
        lambda_reasoning=0.2,
        batch_size=64,
        epochs=2,
        device=torch.device("cpu"),
    )
    student, logs = trainer.fit(distill_pack)
    assert len(logs) == 2 and "total_loss" in logs[0]

    # -------------------------------------------------------------------------
    # 5. Online System 1 Execution (`act_online`) — Zero Teacher/LLM/Projection Calls!
    # -------------------------------------------------------------------------
    def _forbidden_online_call(*_args, **_kwargs):
        raise AssertionError("Offline Teacher / LLM / Projection component must NEVER run during online System 1 inference!")

    llm.generate_json = _forbidden_online_call  # type: ignore[method-assign]
    finbert.encode_texts = _forbidden_online_call  # type: ignore[method-assign]
    student.projection_head.forward = _forbidden_online_call  # type: ignore[method-assign]
    student.critic.forward = _forbidden_online_call  # type: ignore[method-assign]

    te_lob_tensor = torch.as_tensor(online_lob_wins[te_slice], dtype=torch.float32)
    online_actions, online_probs = student.act_online(te_lob_tensor)

    assert online_actions.shape == (len(te_slice),)
    assert online_probs.shape == (len(te_slice), 5)
    assert set(online_actions.tolist()).issubset({0, 1, 2, 3, 4})

    target_weights = ACTION_WEIGHTS[online_actions.numpy()]
    assert set(np.round(target_weights, 2).tolist()).issubset({-1.0, -0.5, 0.0, 0.5, 1.0})

    bt_res = CausalPortfolioBacktester(initial_capital=10_000.0).run(
        df_ohlcv.iloc[te_slice], target_weights, phase=2
    )
    assert len(bt_res.equity_curve) == len(te_slice)
    assert np.isfinite(bt_res.summary.sharpe)
