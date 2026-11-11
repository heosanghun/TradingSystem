"""
WP2 — Study 1 Walk-Forward Execution (Multimodal Encoders + XGBoost Regime + Dynamic PPO+PER Ensemble).

Measures & records:
- Table 6.9 & Table 6.15 Row 3: Period II (`2022`) KernelSHAP modality attribution (`Visual` vs `Tech + Sentiment`)
- Table 6.10: `Variant D` (Study 1 standalone without Teacher debate/distillation), `Variant A` (Tech-only), `Variant B` (Static 1/N), `Variant C` (Single agent K=1)
- Table 6.12 (5 blank cells): Ensemble temperature sensitivity `T ∈ {1, 5, 10, 20, 'static_1_over_N'}`
- Table 6.13 (9 blank cells): Experts per regime sensitivity `K ∈ {3, 5, 7}` (`{Sharpe, Sortino, MDD}`)
- Exports OOS regime sequence & action probabilities (`runs/study1/study1_oos_signals.npz`) for Study 2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from ts.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, SystemConfig
from ts.data.binance_fetcher import BinanceDataFetcher
from ts.data.news_fetcher import PointInTimeNewsStore
from ts.eval.backtester import CausalPortfolioBacktester
from ts.eval.cost_model import compute_phase2_bar_base_cost_rates
from ts.eval.shap_analysis import ModalitySHAPResult, compute_modality_kernel_shap
from ts.eval.walk_forward import generate_phase2_walk_forward_folds
from ts.ledger import NumberLedger
from ts.study1.chart_renderer import CandlestickChartRenderer
from ts.study1.encoders import MultimodalStateEncoder, TechnicalIndicatorCalculator
from ts.study1.ensemble import DynamicRegimeEnsemble
from ts.study1.ppo_expert import ACTION_WEIGHTS, RegimePPOExpert
from ts.study1.regime_classifier import XGBoostRegimeClassifier
from ts.study1.regime_labeling import REGIME_BEAR, REGIME_SIDEWAYS, TrendScanningLabeler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Study 1 Walk-Forward Evaluation & Sensitivity Sweeps.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--news-file", type=str, default="", help="Optional CSV/Parquet news archive with published_at & headline.")
    parser.add_argument("--offline-finbert-hash", action="store_true", help="Skip HuggingFace FinBERT download when scoring raw headlines offline.")
    parser.add_argument("--pretrain-epochs", type=int, default=4)
    parser.add_argument("--ppo-rollouts", type=int, default=8)
    args = parser.parse_args()

    cfg = SystemConfig.from_json(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = PROJECT_ROOT / "data" / "cache"
    out_dir = PROJECT_ROOT / "runs" / "study1"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = "runs/study1/metrics.json"
    script_rel = "scripts/run_study1.py"

    binance = BinanceDataFetcher(cache_dir=cache_dir)
    df_all, _ = binance.fetch_ohlcv(
        symbol=cfg.data.primary_symbol,
        interval=cfg.data.bar_interval,
        start_utc=cfg.data.warmup_start,
        end_utc=cfg.data.eval_end,
        realign_grid=False,
    )

    news_store = (
        PointInTimeNewsStore.from_file(args.news_file)
        if args.news_file and Path(args.news_file).exists()
        else PointInTimeNewsStore()
    )
    if not news_store.df.empty:
        n_scored = news_store.populate_missing_logits_with_finbert(
            model_name=cfg.study3.finbert_model,
            offline_hash_mode=args.offline_finbert_hash,
            device=device,
        )
        if n_scored > 0:
            print(f"[Study 1] Scored {n_scored} raw news headlines into 3-class sentiment logits.")

    print(f"[Study 1] Rendering 60-bar candlestick charts and causal features for {len(df_all)} bars on {device}...")
    chart_renderer = CandlestickChartRenderer(
        window_bars=cfg.study1_encoders.candlestick_window_bars,
        image_size=cfg.study1_encoders.image_size,
    )
    charts_all = chart_renderer.render_series_batch(df_all)
    tech_df = TechnicalIndicatorCalculator(zscore_window=cfg.study1_encoders.tech_zscore_window).compute(df_all)
    tech_all = tech_df.to_numpy(dtype=np.float32)
    senti_df = news_store.build_hourly_sentiment_summary(
        pd.DatetimeIndex(df_all.index), ewma_span_bars=cfg.study1_encoders.senti_ewma_span_bars
    )
    senti_all = senti_df.to_numpy(dtype=np.float32)

    closes = df_all["close"].to_numpy(dtype=np.float64)
    next_ret_all = np.zeros(len(df_all), dtype=np.float64)
    next_ret_all[:-1] = (closes[1:] / np.maximum(closes[:-1], 1e-8)) - 1.0
    base_costs_all = compute_phase2_bar_base_cost_rates(
        df_all,
        fee_rate=cfg.evaluation.phase2_fee_rate,
        atr_period=cfg.evaluation.phase2_atr_period,
        kappa=cfg.evaluation.phase2_kappa,
    )

    folds = generate_phase2_walk_forward_folds(
        pd.DatetimeIndex(df_all.index),
        first_train_start=cfg.data.first_train_start,
        train_months=cfg.evaluation.walk_forward_train_months,
        test_months=cfg.evaluation.walk_forward_test_months,
        step_months=cfg.evaluation.walk_forward_step_months,
        num_folds=cfg.evaluation.walk_forward_num_folds,
    )

    k_max = cfg.study1_ppo.max_experts_per_regime_for_sensitivity  # 7 experts trained once per regime!
    temp_keys = [str(t) for t in cfg.evaluation.sensitivity_temp_grid]
    expert_keys = [int(k) for k in cfg.evaluation.sensitivity_experts_grid]

    oos_indices_all: List[np.ndarray] = []
    oos_regimes_all: List[np.ndarray] = []
    oos_ens_probs_all: List[np.ndarray] = []
    fold1_tr_indices: np.ndarray | None = None
    fold1_tr_regimes: np.ndarray | None = None
    fold1_tr_probs: np.ndarray | None = None
    oos_var_d_weights: List[np.ndarray] = []
    oos_var_a_weights: List[np.ndarray] = []
    oos_var_b_weights: List[np.ndarray] = []
    oos_var_c_weights: List[np.ndarray] = []
    oos_temp_weights: Dict[str, List[np.ndarray]] = {tk: [] for tk in temp_keys}
    oos_k_weights: Dict[int, List[np.ndarray]] = {kk: [] for kk in expert_keys}

    shap_period2_results: List[ModalitySHAPResult] = []
    prev_last_regime = REGIME_SIDEWAYS

    for fold in folds:
        print(f"[Study 1] Fold {fold.fold_id}/{len(folds)}: Train [{fold.train_start.date()}..{fold.train_end.date()}] -> Test [{fold.test_start.date()}..{fold.test_end.date()}]")
        tr_idx = fold.train_indices
        te_idx = fold.test_indices

        # 1. Strictly purged forward Trend Scanning labels on training fold only
        labeler = TrendScanningLabeler(
            horizon_min=cfg.study1_encoders.trend_scan_min_horizon,
            horizon_max=cfg.study1_encoders.trend_scan_max_horizon,
            t_threshold=cfg.study1_encoders.trend_scan_t_threshold,
        )
        tr_labels = labeler.fit_label_train_fold(df_all["close"].iloc[tr_idx])

        # 2. Fold-local encoder auxiliary pretraining -> freeze
        encoder = MultimodalStateEncoder(pretrained_resnet=False)
        encoder.pretrain_on_fold_regimes(
            chart_imgs=charts_all[tr_idx],
            tech_x=tech_all[tr_idx],
            senti_x=senti_all[tr_idx],
            regime_labels=tr_labels,
            epochs=args.pretrain_epochs,
            device=device,
        )
        tr_states, _ = encoder.encode_dataset(charts_all[tr_idx], tech_all[tr_idx], senti_all[tr_idx], device=device)
        te_states, te_concat_mods = encoder.encode_dataset(charts_all[te_idx], tech_all[te_idx], senti_all[te_idx], device=device)

        # 3. XGBoost regime classifier with theta = 0.60 hysteresis
        clf = XGBoostRegimeClassifier(
            n_estimators=cfg.study1_encoders.xgb_n_estimators,
            max_depth=cfg.study1_encoders.xgb_max_depth,
            learning_rate=cfg.study1_encoders.xgb_learning_rate,
            switch_threshold=cfg.study1_encoders.regime_switch_threshold,
            random_state=42 + fold.fold_id,
        )
        clf.fit(tr_states, tr_labels)
        tr_pred_regimes, _ = clf.predict_regimes_causal(tr_states, initial_regime=prev_last_regime)
        te_pred_regimes, _ = clf.predict_regimes_causal(
            te_states,
            initial_regime=int(tr_pred_regimes[-1]) if len(tr_pred_regimes) > 0 else prev_last_regime,
        )
        prev_last_regime = int(te_pred_regimes[-1])

        # 4. Train K_max = 7 PPO+PER experts per regime once on this fold
        ens = DynamicRegimeEnsemble(
            state_dim=cfg.study1_encoders.cross_attn_dim,
            experts_per_regime=k_max,
            rolling_sharpe_window_bars=cfg.study1_ppo.rolling_sharpe_window_bars,
            bars_per_year=cfg.evaluation.btc_bars_per_year,
            base_seed=1000 * fold.fold_id,
            ppo_kwargs={"device": device},
        )
        ens.train_all_pools(
            train_states=tr_states,
            train_regimes=tr_pred_regimes,
            next_bar_returns=next_ret_all[tr_idx],
            cost_rates=base_costs_all[tr_idx],
            total_rollouts=args.ppo_rollouts,
        )

        # Primary Study 1 (`K = 5, T = 10.0` = Variant D)
        if fold.fold_id == 1:
            tr_probs_d, _, _ = ens.rollout_causal_policy(
                states=tr_states,
                active_regimes=tr_pred_regimes,
                bar_asset_returns=next_ret_all[tr_idx],
                one_way_cost_rates=base_costs_all[tr_idx],
                temperature=cfg.study1_ppo.ensemble_temperature,
                num_experts_subset=cfg.study1_ppo.experts_per_regime,
            )
            fold1_tr_indices = tr_idx
            fold1_tr_regimes = tr_pred_regimes
            fold1_tr_probs = tr_probs_d

        ens_probs_d, _, w_d = ens.rollout_causal_policy(
            states=te_states,
            active_regimes=te_pred_regimes,
            bar_asset_returns=next_ret_all[te_idx],
            one_way_cost_rates=base_costs_all[te_idx],
            temperature=cfg.study1_ppo.ensemble_temperature,
            num_experts_subset=cfg.study1_ppo.experts_per_regime,
            warmup_states=tr_states,
            warmup_returns=next_ret_all[tr_idx],
            warmup_costs=base_costs_all[tr_idx],
        )
        oos_indices_all.append(te_idx)
        oos_regimes_all.append(te_pred_regimes)
        oos_ens_probs_all.append(ens_probs_d)
        oos_var_d_weights.append(w_d)

        # Variant B (`static_1_over_N`, K=5) & Variant C (`K=1`, single expert)
        _, _, w_b = ens.rollout_causal_policy(
            states=te_states,
            active_regimes=te_pred_regimes,
            bar_asset_returns=next_ret_all[te_idx],
            one_way_cost_rates=base_costs_all[te_idx],
            temperature="static_1_over_N",
            num_experts_subset=cfg.study1_ppo.experts_per_regime,
            warmup_states=tr_states,
            warmup_returns=next_ret_all[tr_idx],
            warmup_costs=base_costs_all[tr_idx],
        )
        oos_var_b_weights.append(w_b)

        _, _, w_c = ens.rollout_causal_policy(
            states=te_states,
            active_regimes=te_pred_regimes,
            bar_asset_returns=next_ret_all[te_idx],
            one_way_cost_rates=base_costs_all[te_idx],
            temperature=cfg.study1_ppo.ensemble_temperature,
            num_experts_subset=1,
            warmup_states=tr_states,
            warmup_returns=next_ret_all[tr_idx],
            warmup_costs=base_costs_all[tr_idx],
        )
        oos_var_c_weights.append(w_c)

        # Variant A (Multimodal removed -> Technical indicator state only)
        tech_agent = RegimePPOExpert(
            regime_id=0, seed=500 + fold.fold_id, state_dim=tech_all.shape[1], device=device
        )
        tech_agent.train_on_fold(
            states=tech_all[tr_idx],
            next_bar_returns=next_ret_all[tr_idx],
            cost_rates=base_costs_all[tr_idx],
            total_rollouts=args.ppo_rollouts,
        )
        va_probs = tech_agent.predict_probs(tech_all[te_idx])
        oos_var_a_weights.append(ACTION_WEIGHTS[np.argmax(va_probs, axis=-1)])

        # Table 6.12 Temperature Sensitivity (`K = 5`)
        for t_val in cfg.evaluation.sensitivity_temp_grid:
            _, _, w_t = ens.rollout_causal_policy(
                states=te_states,
                active_regimes=te_pred_regimes,
                bar_asset_returns=next_ret_all[te_idx],
                one_way_cost_rates=base_costs_all[te_idx],
                temperature=t_val,
                num_experts_subset=cfg.study1_ppo.experts_per_regime,
                warmup_states=tr_states,
                warmup_returns=next_ret_all[tr_idx],
                warmup_costs=base_costs_all[tr_idx],
            )
            oos_temp_weights[str(t_val)].append(w_t)

        # Table 6.13 Experts-per-Regime Sensitivity (`K ∈ {3, 5, 7}`, `T = 10.0`)
        for k_val in expert_keys:
            _, _, w_k = ens.rollout_causal_policy(
                states=te_states,
                active_regimes=te_pred_regimes,
                bar_asset_returns=next_ret_all[te_idx],
                one_way_cost_rates=base_costs_all[te_idx],
                temperature=cfg.study1_ppo.ensemble_temperature,
                num_experts_subset=k_val,
                warmup_states=tr_states,
                warmup_returns=next_ret_all[tr_idx],
                warmup_costs=base_costs_all[tr_idx],
            )
            oos_k_weights[k_val].append(w_k)

        # Table 6.9 KernelSHAP on Period II (`2022`) folds (Folds 2, 3, 4, 5)
        if fold.test_start.year == 2022:
            shap_res = compute_modality_kernel_shap(
                concat_modalities=te_concat_mods,
                fusion_module=encoder.fusion,
                ensemble=ens,
                target_regime=REGIME_BEAR,
                seed=42 + fold.fold_id,
                device=device,
            )
            shap_period2_results.append(shap_res)

    cat_idx = np.concatenate(oos_indices_all)
    eval_df = df_all.iloc[cat_idx].copy()
    bt = CausalPortfolioBacktester(
        initial_capital=cfg.evaluation.initial_capital_usdt,
        bars_per_year=cfg.evaluation.btc_bars_per_year,
    )

    res_var_d = bt.run_decisions(eval_df, np.concatenate(oos_var_d_weights), phase=2)
    res_var_a = bt.run_decisions(eval_df, np.concatenate(oos_var_a_weights), phase=2)
    res_var_b = bt.run_decisions(eval_df, np.concatenate(oos_var_b_weights), phase=2)
    res_var_c = bt.run_decisions(eval_df, np.concatenate(oos_var_c_weights), phase=2)

    table_6_12 = {}
    for tk in temp_keys:
        r_t = bt.run_decisions(eval_df, np.concatenate(oos_temp_weights[tk]), phase=2)
        table_6_12[tk] = r_t.summary.to_dict()

    table_6_13 = {}
    for kk in expert_keys:
        r_k = bt.run_decisions(eval_df, np.concatenate(oos_k_weights[kk]), phase=2)
        table_6_13[str(kk)] = r_k.summary.to_dict()

    if shap_period2_results:
        vis_mean = float(np.mean([s.visual_pct for s in shap_period2_results]))
        tech_mean = float(np.mean([s.tech_pct for s in shap_period2_results]))
        senti_mean = float(np.mean([s.sentiment_pct for s in shap_period2_results]))
    else:
        vis_mean, tech_mean, senti_mean = 0.0, 0.0, 0.0

    table_6_9 = {
        "visual_pct": vis_mean,
        "tech_pct": tech_mean,
        "sentiment_pct": senti_mean,
        "tech_plus_sentiment_pct": tech_mean + senti_mean,
    }

    cat_oos_regimes = np.concatenate(oos_regimes_all)
    cat_oos_probs = np.concatenate(oos_ens_probs_all)
    if fold1_tr_indices is not None and fold1_tr_regimes is not None and fold1_tr_probs is not None:
        teacher_bar_indices = np.concatenate([fold1_tr_indices, cat_idx])
        teacher_regimes = np.concatenate([fold1_tr_regimes, cat_oos_regimes])
        teacher_ens_probs = np.concatenate([fold1_tr_probs, cat_oos_probs], axis=0)
        teacher_is_oos = np.concatenate(
            [np.zeros(len(fold1_tr_indices), dtype=bool), np.ones(len(cat_idx), dtype=bool)]
        )
    else:
        teacher_bar_indices = cat_idx
        teacher_regimes = cat_oos_regimes
        teacher_ens_probs = cat_oos_probs
        teacher_is_oos = np.ones(len(cat_idx), dtype=bool)

    np.savez_compressed(
        out_dir / "study1_oos_signals.npz",
        oos_bar_indices=cat_idx,
        oos_regimes=cat_oos_regimes,
        oos_ens_probs=cat_oos_probs,
        oos_var_d_weights=np.concatenate(oos_var_d_weights),
        teacher_bar_indices=teacher_bar_indices,
        teacher_regimes=teacher_regimes,
        teacher_ens_probs=teacher_ens_probs,
        teacher_is_oos=teacher_is_oos,
    )

    metrics_payload = {
        "config_sha256": cfg.config_sha256,
        "ablation_study1_variants": {
            "Variant_A_no_multimodal": res_var_a.summary.to_dict(),
            "Variant_B_static_1_over_N": res_var_b.summary.to_dict(),
            "Variant_C_single_expert": res_var_c.summary.to_dict(),
            "Variant_D_study1_standalone": res_var_d.summary.to_dict(),
        },
        "table_6_9_shap_period2": table_6_9,
        "table_6_12_temperature_sensitivity": table_6_12,
        "table_6_13_experts_sensitivity": table_6_13,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Record into `ledger/numbers.csv` (including all 5 blanks of Table 6.12 & all 9 blanks of Table 6.13!)
    ledger = NumberLedger()
    rec_kw = dict(script_path=script_rel, config_sha256=cfg.config_sha256, run_log_path=log_path)

    ledger.record("T6.9_SHAP_VISUAL_PCT", "Table 6.9", "Period II SHAP Visual share", vis_mean, "{:.2f}", "%", **rec_kw)
    ledger.record("T6.9_SHAP_TECH_SENTI_PCT", "Table 6.9", "Period II SHAP Tech+Sentiment share", tech_mean + senti_mean, "{:.2f}", "%", **rec_kw)

    ledger.record("T6.10_VAR_A_SHARPE", "Table 6.10", "Variant A (No Multimodal) Phase 2 Sharpe", res_var_a.summary.sharpe, "{:.2f}", "", **rec_kw)
    ledger.record("T6.10_VAR_B_SHARPE", "Table 6.10", "Variant B (Static 1/N) Phase 2 Sharpe", res_var_b.summary.sharpe, "{:.2f}", "", **rec_kw)
    ledger.record("T6.10_VAR_C_SHARPE", "Table 6.10", "Variant C (Single Agent) Phase 2 Sharpe", res_var_c.summary.sharpe, "{:.2f}", "", **rec_kw)
    ledger.record("T6.10_VAR_D_SHARPE", "Table 6.10", "Variant D (Study 1 Standalone) Phase 2 Sharpe", res_var_d.summary.sharpe, "{:.2f}", "", **rec_kw)
    ledger.record("T6.10_VAR_D_MDD", "Table 6.10", "Variant D (Study 1 Standalone) Phase 2 MDD", res_var_d.summary.mdd_pct, "{:.2f}", "%", **rec_kw)

    for tk, d_t in table_6_12.items():
        clean_k = tk.replace(".", "_")
        ledger.record(f"T6.12_TEMP_{clean_k}_SHARPE", "Table 6.12", f"Ensemble Temp={tk} Phase 2 Sharpe", float(d_t["sharpe"]), "{:.2f}", "", **rec_kw)

    for kk_str, d_k in table_6_13.items():
        ledger.record(f"T6.13_EXP_{kk_str}_SHARPE", "Table 6.13", f"Experts K={kk_str} Phase 2 Sharpe", float(d_k["sharpe"]), "{:.2f}", "", **rec_kw)
        ledger.record(f"T6.13_EXP_{kk_str}_SORTINO", "Table 6.13", f"Experts K={kk_str} Phase 2 Sortino", float(d_k["sortino"]), "{:.2f}", "", **rec_kw)
        ledger.record(f"T6.13_EXP_{kk_str}_MDD", "Table 6.13", f"Experts K={kk_str} Phase 2 MDD", float(d_k["mdd_pct"]), "{:.2f}", "%", **rec_kw)

    print("[Study 1] Completed and recorded into ledger/numbers.csv.")


if __name__ == "__main__":
    main()
