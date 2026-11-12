"""
Study 1 Walk-Forward Execution (Multimodal Encoders, XGBoost Regime Gate, and Dynamic PPO+PER Ensemble).

Measures and records:
- Table 6.9: Period II (2022) group-level Shapley modality attribution (`visual_pct`, `tech_pct`, `sentiment_pct`)
- Table 6.10: `Variant D` (Study 1 standalone), `Variant A` (Modality ablation keeping regime gate and ensemble),
  `Variant B` (Static 1/N weights), `Variant C` (Single expert K=1)
- Table 6.12: Ensemble temperature sensitivity `T in {1, 5, 10, 20, 'static_1_over_N'}`
- Table 6.13: Experts per regime sensitivity `K in {3, 5, 7}` (`{Sharpe, Sortino, MDD}`)
- Exports out-of-sample regime sequence and action probabilities (`runs/study1/study1_oos_signals.npz`) for Study 2.
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from ts.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, SystemConfig
from ts.data.binance_fetcher import BinanceDataFetcher
from ts.data.news_fetcher import PointInTimeNewsStore
from ts.eval.backtester import CausalPortfolioBacktester
from ts.eval.cost_model import compute_phase2_bar_base_cost_rates
from ts.eval.walk_forward import WalkForwardFold, generate_phase2_walk_forward_folds
from ts.ledger import NumberLedger, RunContext
from ts.study1.attribution import ModalityAttributionResult, compute_modality_group_shapley
from ts.study1.chart_renderer import CandlestickChartRenderer
from ts.study1.encoders import ModalityMode, MultimodalStateEncoder, TechnicalIndicatorCalculator
from ts.study1.ensemble import DynamicRegimeEnsemble
from ts.study1.regime_classifier import XGBoostRegimeClassifier
from ts.study1.regime_labeling import REGIME_SIDEWAYS, TrendScanningLabeler


@dataclass
class _FoldBranchOutput:
    encoder: MultimodalStateEncoder
    ensemble: DynamicRegimeEnsemble
    tr_states: np.ndarray
    te_states: np.ndarray
    te_concat_mods: np.ndarray
    tr_pred_regimes: np.ndarray
    te_pred_regimes: np.ndarray
    ens_probs_te: np.ndarray
    chosen_actions_te: np.ndarray
    weights_te: np.ndarray
    expert_weights_te: np.ndarray
    next_initial_regime: int


def _run_study1_fold_branch(
    *,
    fold: WalkForwardFold,
    tr_labels: np.ndarray,
    charts_all: np.ndarray,
    tech_all: np.ndarray,
    senti_all: np.ndarray,
    next_ret_all: np.ndarray,
    base_costs_all: np.ndarray,
    cfg: SystemConfig,
    modality_mode: ModalityMode,
    pretrained_resnet: bool,
    pretrain_epochs: int,
    ppo_rollouts: int,
    pool_experts: int,
    eval_experts_subset: int,
    prev_initial_regime: int,
    base_seed_offset: int,
    log_dir: Optional[Path],
    tag_prefix: str,
    device: torch.device,
) -> _FoldBranchOutput:
    """
    Shared fold pipeline used by both Variant D (`modality_mode='multimodal'`) and
    Variant A (`modality_mode='tech_only'` or `'visual_tech_no_senti'`), ensuring that
    the two variants differ solely in encoder input modality.
    """
    tr_idx = fold.train_indices
    te_idx = fold.test_indices

    encoder = MultimodalStateEncoder(
        pretrained_resnet=pretrained_resnet,
        modality_mode=modality_mode,
    )
    encoder.pretrain_on_fold_regimes(
        chart_imgs=charts_all,
        tech_x=tech_all,
        senti_x=senti_all,
        regime_labels=tr_labels,
        epochs=pretrain_epochs,
        device=device,
        indices=tr_idx,
    )
    tr_states, _ = encoder.encode_dataset(
        charts_all, tech_all, senti_all, device=device, indices=tr_idx
    )
    te_states, te_concat_mods = encoder.encode_dataset(
        charts_all, tech_all, senti_all, device=device, indices=te_idx
    )

    clf = XGBoostRegimeClassifier(
        n_estimators=cfg.study1_encoders.xgb_n_estimators,
        max_depth=cfg.study1_encoders.xgb_max_depth,
        learning_rate=cfg.study1_encoders.xgb_learning_rate,
        switch_threshold=cfg.study1_encoders.regime_switch_threshold,
        random_state=42 + fold.fold_id + base_seed_offset,
    )
    clf.fit(tr_states, tr_labels)
    tr_pred_regimes, _ = clf.predict_regimes_causal(tr_states, initial_regime=prev_initial_regime)
    te_pred_regimes, _ = clf.predict_regimes_causal(
        te_states,
        initial_regime=int(tr_pred_regimes[-1]) if len(tr_pred_regimes) > 0 else prev_initial_regime,
    )
    next_init = int(te_pred_regimes[-1]) if len(te_pred_regimes) > 0 else prev_initial_regime

    ens = DynamicRegimeEnsemble(
        state_dim=cfg.study1_encoders.cross_attn_dim,
        experts_per_regime=pool_experts,
        rolling_sharpe_window_bars=cfg.study1_ppo.rolling_sharpe_window_bars,
        bars_per_year=cfg.evaluation.btc_bars_per_year,
        base_seed=1000 * fold.fold_id + base_seed_offset,
        ppo_kwargs={
            "hidden_dims": cfg.study1_ppo.hidden_dims,
            "lr_actor": cfg.study1_ppo.lr_actor,
            "lr_critic": cfg.study1_ppo.lr_critic,
            "gamma": cfg.study1_ppo.gamma,
            "gae_lambda": cfg.study1_ppo.gae_lambda,
            "clip_eps": cfg.study1_ppo.clip_eps,
            "entropy_coef": cfg.study1_ppo.entropy_coef,
            "value_coef": cfg.study1_ppo.value_coef,
            "per_alpha": cfg.study1_ppo.per_alpha,
            "per_beta_start": cfg.study1_ppo.per_beta_start,
            "per_beta_end": cfg.study1_ppo.per_beta_end,
            "minibatch_size": cfg.study1_ppo.minibatch_size,
            "ppo_epochs": cfg.study1_ppo.ppo_epochs,
            "device": device,
        },
    )
    ens.train_all_pools(
        train_states=tr_states,
        train_regimes=tr_pred_regimes,
        next_bar_returns=next_ret_all[tr_idx],
        cost_rates=base_costs_all[tr_idx],
        total_rollouts=ppo_rollouts,
        log_dir=log_dir,
        fold_id=fold.fold_id,
        tag_prefix=tag_prefix,
    )

    ens_probs, chosen_acts, w_out = ens.rollout_causal_policy(
        states=te_states,
        active_regimes=te_pred_regimes,
        bar_asset_returns=next_ret_all[te_idx],
        one_way_cost_rates=base_costs_all[te_idx],
        temperature=cfg.study1_ppo.ensemble_temperature,
        num_experts_subset=eval_experts_subset,
        warmup_states=tr_states,
        warmup_returns=next_ret_all[tr_idx],
        warmup_costs=base_costs_all[tr_idx],
    )
    assert ens.last_rollout_expert_weights_ is not None
    exp_w_te = ens.last_rollout_expert_weights_.copy()

    return _FoldBranchOutput(
        encoder=encoder,
        ensemble=ens,
        tr_states=tr_states,
        te_states=te_states,
        te_concat_mods=te_concat_mods,
        tr_pred_regimes=tr_pred_regimes,
        te_pred_regimes=te_pred_regimes,
        ens_probs_te=ens_probs,
        chosen_actions_te=chosen_acts,
        weights_te=w_out,
        expert_weights_te=exp_w_te,
        next_initial_regime=next_init,
    )


def main() -> None:
    default_news = PROJECT_ROOT / "data" / "news_headlines.parquet"
    parser = argparse.ArgumentParser(description="Run Study 1 Walk-Forward Evaluation and Sensitivity Sweeps.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--news-file",
        type=str,
        default=str(default_news) if default_news.exists() else "",
        help="CSV/Parquet news archive with published_at and headline.",
    )
    parser.add_argument("--no-pretrained-resnet", action="store_true", help="Disable ImageNet pretrained ResNet-18 weights in offline tests.")
    parser.add_argument(
        "--variant-a-modality",
        type=str,
        default="tech_only",
        choices=["tech_only", "visual_tech_no_senti"],
        help="Input modality mode for Variant A ablation (default: single-modality tech_only; see D-14).",
    )
    parser.add_argument("--pretrain-epochs", type=int, default=4)
    parser.add_argument("--ppo-rollouts", type=int, default=None)
    args = parser.parse_args()

    cfg = SystemConfig.from_json(args.config)
    default_rollouts = int(getattr(cfg.study1_ppo, "default_rollouts_per_fold", 64))
    ppo_rollouts = int(args.ppo_rollouts) if args.ppo_rollouts is not None else default_rollouts

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = PROJECT_ROOT / "data" / "cache"
    out_dir = PROJECT_ROOT / "runs" / "study1"
    training_logs_dir = out_dir / "training_logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    training_logs_dir.mkdir(parents=True, exist_ok=True)
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
            device=device,
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if n_scored > 0:
            print(f"[Study 1] Scored {n_scored} raw news headlines into 3-class sentiment logits.")

    print(f"[Study 1] Rendering 60-bar candlestick charts and causal features for {len(df_all)} bars on {device}...")
    chart_renderer = CandlestickChartRenderer(
        window_bars=cfg.study1_encoders.candlestick_window_bars,
        image_size=cfg.study1_encoders.image_size,
    )
    charts_all = chart_renderer.render_series_uint8(df_all)
    print(
        f"[Study 1] Chart uint8 buffer allocated: shape={charts_all.shape}, "
        f"dtype={charts_all.dtype}, nbytes={charts_all.nbytes / (1024 ** 3):.2f} GiB"
    )
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

    k_max = cfg.study1_ppo.max_experts_per_regime_for_sensitivity
    temp_keys = [str(t) for t in cfg.evaluation.sensitivity_temp_grid]
    expert_keys = [int(k) for k in cfg.evaluation.sensitivity_experts_grid]
    use_pretrained = not args.no_pretrained_resnet
    var_a_mode: ModalityMode = args.variant_a_modality  # type: ignore[assignment]

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

    shap_period2_results: List[ModalityAttributionResult] = []
    prev_last_regime_d = REGIME_SIDEWAYS
    prev_last_regime_a = REGIME_SIDEWAYS

    for fold in folds:
        print(
            f"[Study 1] Fold {fold.fold_id}/{len(folds)}: "
            f"Train [{fold.train_start.date()}..{fold.train_end.date()}] -> "
            f"Test [{fold.test_start.date()}..{fold.test_end.date()}]"
        )
        tr_idx = fold.train_indices
        te_idx = fold.test_indices

        labeler = TrendScanningLabeler(
            horizon_min=cfg.study1_encoders.trend_scan_min_horizon,
            horizon_max=cfg.study1_encoders.trend_scan_max_horizon,
            t_threshold=cfg.study1_encoders.trend_scan_t_threshold,
        )
        tr_labels = labeler.fit_label_train_fold(df_all["close"].iloc[tr_idx])

        # Primary Multimodal Branch (Variant D, K_max = 7 trained once for sensitivity sweeps)
        branch_d = _run_study1_fold_branch(
            fold=fold,
            tr_labels=tr_labels,
            charts_all=charts_all,
            tech_all=tech_all,
            senti_all=senti_all,
            next_ret_all=next_ret_all,
            base_costs_all=base_costs_all,
            cfg=cfg,
            modality_mode="multimodal",
            pretrained_resnet=use_pretrained,
            pretrain_epochs=args.pretrain_epochs,
            ppo_rollouts=ppo_rollouts,
            pool_experts=k_max,
            eval_experts_subset=cfg.study1_ppo.experts_per_regime,
            prev_initial_regime=prev_last_regime_d,
            base_seed_offset=0,
            log_dir=training_logs_dir,
            tag_prefix="varD",
            device=device,
        )
        prev_last_regime_d = branch_d.next_initial_regime

        if fold.fold_id == 1:
            tr_probs_d, _, _ = branch_d.ensemble.rollout_causal_policy(
                states=branch_d.tr_states,
                active_regimes=branch_d.tr_pred_regimes,
                bar_asset_returns=next_ret_all[tr_idx],
                one_way_cost_rates=base_costs_all[tr_idx],
                temperature=cfg.study1_ppo.ensemble_temperature,
                num_experts_subset=cfg.study1_ppo.experts_per_regime,
            )
            fold1_tr_indices = tr_idx
            fold1_tr_regimes = branch_d.tr_pred_regimes
            fold1_tr_probs = tr_probs_d

        oos_indices_all.append(te_idx)
        oos_regimes_all.append(branch_d.te_pred_regimes)
        oos_ens_probs_all.append(branch_d.ens_probs_te)
        oos_var_d_weights.append(branch_d.weights_te)

        # Table 6.9 3-group Shapley attribution on Period II (2022) folds targeting active regime chosen action logit
        if fold.test_start.year == 2022 and branch_d.encoder.fusion is not None:
            shap_res = compute_modality_group_shapley(
                concat_modalities=branch_d.te_concat_mods,
                fusion_module=branch_d.encoder.fusion,
                ensemble=branch_d.ensemble,
                active_regimes=branch_d.te_pred_regimes,
                chosen_actions=branch_d.chosen_actions_te,
                expert_weights=branch_d.expert_weights_te,
                num_experts_subset=cfg.study1_ppo.experts_per_regime,
                n_background=256,
                n_explain=512,
                seed=42 + fold.fold_id,
                device=device,
            )
            shap_period2_results.append(shap_res)

        # Variant B (static 1/N weights, K = 5) and Variant C (single expert K = 1)
        _, _, w_b = branch_d.ensemble.rollout_causal_policy(
            states=branch_d.te_states,
            active_regimes=branch_d.te_pred_regimes,
            bar_asset_returns=next_ret_all[te_idx],
            one_way_cost_rates=base_costs_all[te_idx],
            temperature="static_1_over_N",
            num_experts_subset=cfg.study1_ppo.experts_per_regime,
            warmup_states=branch_d.tr_states,
            warmup_returns=next_ret_all[tr_idx],
            warmup_costs=base_costs_all[tr_idx],
        )
        oos_var_b_weights.append(w_b)

        _, _, w_c = branch_d.ensemble.rollout_causal_policy(
            states=branch_d.te_states,
            active_regimes=branch_d.te_pred_regimes,
            bar_asset_returns=next_ret_all[te_idx],
            one_way_cost_rates=base_costs_all[te_idx],
            temperature=cfg.study1_ppo.ensemble_temperature,
            num_experts_subset=1,
            warmup_states=branch_d.tr_states,
            warmup_returns=next_ret_all[tr_idx],
            warmup_costs=base_costs_all[tr_idx],
        )
        oos_var_c_weights.append(w_c)

        # Variant A: Identical Study 1 pipeline (XGBoost regime gate + 3-regime K=5 dynamic ensemble),
        # differing from Variant D only in `modality_mode`.
        branch_a = _run_study1_fold_branch(
            fold=fold,
            tr_labels=tr_labels,
            charts_all=charts_all,
            tech_all=tech_all,
            senti_all=senti_all,
            next_ret_all=next_ret_all,
            base_costs_all=base_costs_all,
            cfg=cfg,
            modality_mode=var_a_mode,
            pretrained_resnet=use_pretrained,
            pretrain_epochs=args.pretrain_epochs,
            ppo_rollouts=ppo_rollouts,
            pool_experts=cfg.study1_ppo.experts_per_regime,
            eval_experts_subset=cfg.study1_ppo.experts_per_regime,
            prev_initial_regime=prev_last_regime_a,
            base_seed_offset=50_000,
            log_dir=training_logs_dir,
            tag_prefix=f"varA_{var_a_mode}",
            device=device,
        )
        prev_last_regime_a = branch_a.next_initial_regime
        oos_var_a_weights.append(branch_a.weights_te)
        branch_a.encoder.cpu()
        branch_d.encoder.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Table 6.12 Temperature Sensitivity (K = 5)
        for t_val in cfg.evaluation.sensitivity_temp_grid:
            _, _, w_t = branch_d.ensemble.rollout_causal_policy(
                states=branch_d.te_states,
                active_regimes=branch_d.te_pred_regimes,
                bar_asset_returns=next_ret_all[te_idx],
                one_way_cost_rates=base_costs_all[te_idx],
                temperature=t_val,
                num_experts_subset=cfg.study1_ppo.experts_per_regime,
                warmup_states=branch_d.tr_states,
                warmup_returns=next_ret_all[tr_idx],
                warmup_costs=base_costs_all[tr_idx],
            )
            oos_temp_weights[str(t_val)].append(w_t)

        # Table 6.13 Experts-per-Regime Sensitivity (K in {3, 5, 7}, T = 10.0)
        for k_val in expert_keys:
            _, _, w_k = branch_d.ensemble.rollout_causal_policy(
                states=branch_d.te_states,
                active_regimes=branch_d.te_pred_regimes,
                bar_asset_returns=next_ret_all[te_idx],
                one_way_cost_rates=base_costs_all[te_idx],
                temperature=cfg.study1_ppo.ensemble_temperature,
                num_experts_subset=k_val,
                warmup_states=branch_d.tr_states,
                warmup_returns=next_ret_all[tr_idx],
                warmup_costs=base_costs_all[tr_idx],
            )
            oos_k_weights[k_val].append(w_k)

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
        abs_vis = float(np.mean([s.mean_abs_visual for s in shap_period2_results]))
        abs_tech = float(np.mean([s.mean_abs_tech for s in shap_period2_results]))
        abs_senti = float(np.mean([s.mean_abs_sentiment for s in shap_period2_results]))
        denom = max(abs_vis + abs_tech + abs_senti, 1e-12)
        vis_mean = float(abs_vis / denom * 100.0)
        tech_mean = float(abs_tech / denom * 100.0)
        senti_mean = float(abs_senti / denom * 100.0)
    else:
        vis_mean, tech_mean, senti_mean = 0.0, 0.0, 0.0

    table_6_9 = {
        "visual_pct": vis_mean,
        "tech_pct": tech_mean,
        "sentiment_pct": senti_mean,
        "tech_plus_sentiment_pct": tech_mean + senti_mean,
        "per_fold": [s.to_dict() for s in shap_period2_results],
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
        "pretrained_resnet": use_pretrained,
        "ppo_rollouts_per_fold": ppo_rollouts,
        "variant_a_modality_mode": var_a_mode,
        "news_headlines_loaded": int(len(news_store.df)),
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

    run_ctx = RunContext(
        stage="study1",
        flags={
            "finbert_offline_hash": False,
            "quick_verify": bool(not use_pretrained),
            "reduced_training_budget": bool(ppo_rollouts < default_rollouts or args.pretrain_epochs < 4),
        },
    )
    ledger = NumberLedger(run_context=run_ctx)
    rec_kw = dict(script_path=script_rel, config_sha256=cfg.config_sha256, run_log_path=log_path)

    ledger.record("T6.9_SHAP_VISUAL_PCT", "Table 6.9", "Period II SHAP Visual share", vis_mean, "{:.2f}", "%", **rec_kw)
    ledger.record("T6.9_SHAP_TECH_PCT", "Table 6.9", "Period II SHAP Technical share", tech_mean, "{:.2f}", "%", **rec_kw)
    ledger.record("T6.9_SHAP_SENTIMENT_PCT", "Table 6.9", "Period II SHAP Sentiment share", senti_mean, "{:.2f}", "%", **rec_kw)
    ledger.record("T6.9_SHAP_TECH_SENTI_PCT", "Table 6.9", "Period II SHAP Tech+Sentiment share", tech_mean + senti_mean, "{:.2f}", "%", **rec_kw)

    ledger.record("T6.10_VAR_A_SHARPE", "Table 6.10", f"Variant A ({var_a_mode}) Phase 2 Sharpe", res_var_a.summary.sharpe, "{:.2f}", "", **rec_kw)
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

    print("[Study 1] Completed and recorded ledger entries.")


if __name__ == "__main__":
    main()
