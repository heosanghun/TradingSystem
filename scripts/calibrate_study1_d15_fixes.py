"""
Pre-registered Fold 1 internal validation evaluation for D-15 (b) structural fixes.

Evaluates the candidate set pre-registered in `collab/DECISIONS.md` (commit `84ec746`)
strictly inside Fold 1's 12-month training window (`2020-10-01 .. 2021-09-30`, 8,741 bars):
  - `sub_train`: first 10 months (`2020-10-01 .. 2021-07-31`, 7,283 bars, forward horizon purged)
  - `val`:       last 2 months  (`2021-08-01 .. 2021-09-30`, 1,458 bars)

Out-of-sample test folds (`2021-10-01 .. 2023-12-31`) are physically sliced off prior to
feature calculation and are never accessed.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from ts.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, SystemConfig
from ts.data.binance_fetcher import BinanceDataFetcher
from ts.data.news_fetcher import PointInTimeNewsStore
from ts.eval.backtester import CausalPortfolioBacktester
from ts.eval.cost_model import compute_phase2_bar_base_cost_rates
from ts.eval.walk_forward import generate_phase2_walk_forward_folds
from ts.ledger import read_git_state
from ts.study1.chart_renderer import CandlestickChartRenderer
from ts.study1.encoders import MultimodalStateEncoder, TechnicalIndicatorCalculator
from ts.study1.ensemble import DynamicRegimeEnsemble
from ts.study1.regime_classifier import XGBoostRegimeClassifier
from ts.study1.regime_labeling import (
    REGIME_BEAR,
    REGIME_BULL,
    REGIME_SIDEWAYS,
    TrendScanningLabeler,
)


def _pre_cost_sharpe(df_val: pd.DataFrame, decision_weights: np.ndarray, bars_per_year: int) -> float:
    closes = df_val["close"].to_numpy(dtype=np.float64)
    if len(closes) < 3:
        return 0.0
    asset_ret = np.zeros(len(closes), dtype=np.float64)
    asset_ret[1:] = (closes[1:] / np.maximum(closes[:-1], 1e-8)) - 1.0
    exec_w = np.zeros(len(closes), dtype=np.float64)
    exec_w[1:] = decision_weights[:-1]
    gross_r = exec_w * asset_ret
    sd = float(np.std(gross_r, ddof=1))
    if sd <= 1e-12:
        return 0.0
    return float((np.mean(gross_r) / sd) * np.sqrt(float(bars_per_year)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate pre-registered D-15 (b) structural fixes on Fold 1 internal validation."
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "runs" / "study1" / "fold1_val_d15_calibration.json"),
    )
    parser.add_argument("--preregistered-commit", type=str, default="84ec746")
    args = parser.parse_args()

    cfg = SystemConfig.from_json(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ppo_rollouts = int(cfg.study1_ppo.default_rollouts_per_fold)
    print(f"[D-15 Calibration] Device={device}, ppo_rollouts={ppo_rollouts}", flush=True)

    fetcher = BinanceDataFetcher(cache_dir=PROJECT_ROOT / "data" / "cache")
    df_all, _ = fetcher.fetch_ohlcv(
        symbol=cfg.data.primary_symbol,
        interval=cfg.data.bar_interval,
        start_utc=cfg.data.warmup_start,
        end_utc=cfg.data.eval_end,
        realign_grid=False,
    )

    folds = generate_phase2_walk_forward_folds(
        pd.DatetimeIndex(df_all.index),
        first_train_start=cfg.data.first_train_start,
        train_months=cfg.evaluation.walk_forward_train_months,
        test_months=cfg.evaluation.walk_forward_test_months,
        step_months=cfg.evaluation.walk_forward_step_months,
        num_folds=cfg.evaluation.walk_forward_num_folds,
    )
    fold1 = folds[0]
    fold1_tr_idx = fold1.train_indices
    fold1_ts = df_all.index[fold1_tr_idx]
    val_cutoff = pd.Timestamp("2021-08-01T00:00:00Z")

    sub_tr_idx = fold1_tr_idx[fold1_ts < val_cutoff]
    val_idx = fold1_tr_idx[fold1_ts >= val_cutoff]
    assert np.max(sub_tr_idx) < np.min(val_idx)
    assert np.max(val_idx) < np.min(fold1.test_indices)

    # Physically exclude all test folds (2021-10-01 onward)
    end_pos = int(val_idx[-1]) + 1
    df_f1 = df_all.iloc[:end_pos].copy()
    df_val = df_f1.iloc[val_idx].copy()

    t_s = time.perf_counter()
    renderer = CandlestickChartRenderer(
        window_bars=cfg.study1_encoders.candlestick_window_bars,
        image_size=cfg.study1_encoders.image_size,
    )
    charts_f1 = renderer.render_series_uint8(df_f1)
    tech_f1 = (
        TechnicalIndicatorCalculator(zscore_window=cfg.study1_encoders.tech_zscore_window)
        .compute(df_f1)
        .to_numpy(dtype=np.float32)
    )
    print(f"[D-15 Calibration] Charts + tech prepared in {time.perf_counter() - t_s:.2f}s", flush=True)

    news_path = PROJECT_ROOT / "data" / "news_headlines.parquet"
    if news_path.exists():
        news_store = PointInTimeNewsStore.from_file(news_path)
        news_store.df = news_store.df[news_store.df["published_at"] <= df_f1.index[-1]].reset_index(drop=True)
        news_store.populate_missing_logits_with_finbert(
            model_name=cfg.study3.finbert_model,
            batch_size=32,
            device=device,
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        news_store = PointInTimeNewsStore()
    senti_f1 = news_store.build_hourly_sentiment_summary(
        pd.DatetimeIndex(df_f1.index), ewma_span_bars=cfg.study1_encoders.senti_ewma_span_bars
    ).to_numpy(dtype=np.float32)

    closes = df_f1["close"].to_numpy(dtype=np.float64)
    next_ret_f1 = np.zeros(len(df_f1), dtype=np.float64)
    next_ret_f1[:-1] = (closes[1:] / np.maximum(closes[:-1], 1e-8)) - 1.0
    costs_f1 = compute_phase2_bar_base_cost_rates(
        df_f1,
        fee_rate=cfg.evaluation.phase2_fee_rate,
        atr_period=cfg.evaluation.phase2_atr_period,
        kappa=cfg.evaluation.phase2_kappa,
    )

    label_specs = {
        "B0_t2": TrendScanningLabeler(
            horizon_min=cfg.study1_encoders.trend_scan_min_horizon,
            horizon_max=cfg.study1_encoders.trend_scan_max_horizon,
            t_threshold=2.0,
            trend_quantile=None,
            unit_root_scale=False,
        ),
        "L1_ols_q60": TrendScanningLabeler(
            horizon_min=cfg.study1_encoders.trend_scan_min_horizon,
            horizon_max=cfg.study1_encoders.trend_scan_max_horizon,
            trend_quantile=0.60,
            unit_root_scale=False,
        ),
        "L2_sqrtL_q60": TrendScanningLabeler(
            horizon_min=cfg.study1_encoders.trend_scan_min_horizon,
            horizon_max=cfg.study1_encoders.trend_scan_max_horizon,
            trend_quantile=0.60,
            unit_root_scale=True,
        ),
    }

    encoded_cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
    for l_key, labeler in label_specs.items():
        t_e = time.perf_counter()
        sub_labels = labeler.fit_label_train_fold(df_f1["close"].iloc[sub_tr_idx])
        torch.manual_seed(42)
        encoder = MultimodalStateEncoder(pretrained_resnet=True, modality_mode="multimodal")
        encoder.pretrain_on_fold_regimes(
            chart_imgs=charts_f1,
            tech_x=tech_f1,
            senti_x=senti_f1,
            regime_labels=sub_labels,
            epochs=4,
            batch_size=32,
            device=device,
            indices=sub_tr_idx,
        )
        sub_states, _ = encoder.encode_dataset(
            charts_f1, tech_f1, senti_f1, batch_size=64, device=device, indices=sub_tr_idx
        )
        val_states, _ = encoder.encode_dataset(
            charts_f1, tech_f1, senti_f1, batch_size=64, device=device, indices=val_idx
        )
        encoder.cpu()
        del encoder
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        encoded_cache[l_key] = (sub_labels, sub_states, val_states, float(labeler.tau_trend_))
        valid_l = sub_labels[sub_labels >= 0]
        cnts = np.bincount(valid_l, minlength=3).tolist()
        print(
            f"[D-15 Calibration] Encoded {l_key}: tau={labeler.tau_trend_:.4f}, "
            f"sub_train_labels={cnts} ({time.perf_counter() - t_e:.1f}s)",
            flush=True,
        )

    gate_specs = {
        "G0": {"balance_classes": True, "confirm_bars": 1},
        "G1": {"balance_classes": False, "confirm_bars": 1},
        "G2": {"balance_classes": False, "confirm_bars": 2},
    }

    eval_plan = [
        # Diagnostic reference rows (not part of the 6-candidate selection pool)
        {"id": "B0_legacy", "label": "B0_t2", "markov": False, "gate": "G0", "in_selection_pool": False},
        {"id": "L2_M0_G1_ablation", "label": "L2_sqrtL_q60", "markov": False, "gate": "G1", "in_selection_pool": False},
        # Pre-registered 6-candidate grid {L1, L2} x {M1} x {G0, G1, G2}
        {"id": "L1_M1_G0", "label": "L1_ols_q60", "markov": True, "gate": "G0", "in_selection_pool": True},
        {"id": "L1_M1_G1", "label": "L1_ols_q60", "markov": True, "gate": "G1", "in_selection_pool": True},
        {"id": "L1_M1_G2", "label": "L1_ols_q60", "markov": True, "gate": "G2", "in_selection_pool": True},
        {"id": "L2_M1_G0", "label": "L2_sqrtL_q60", "markov": True, "gate": "G0", "in_selection_pool": True},
        {"id": "L2_M1_G1", "label": "L2_sqrtL_q60", "markov": True, "gate": "G1", "in_selection_pool": True},
        {"id": "L2_M1_G2", "label": "L2_sqrtL_q60", "markov": True, "gate": "G2", "in_selection_pool": True},
    ]

    bt = CausalPortfolioBacktester(
        initial_capital=cfg.evaluation.initial_capital_usdt,
        bars_per_year=cfg.evaluation.btc_bars_per_year,
    )

    rows: List[Dict[str, object]] = []
    for item in eval_plan:
        cid = str(item["id"])
        l_key = str(item["label"])
        use_m1 = bool(item["markov"])
        g_key = str(item["gate"])
        in_pool = bool(item["in_selection_pool"])
        g_cfg = gate_specs[g_key]

        t0 = time.perf_counter()
        sub_labels, sub_states, val_states, tau_val = encoded_cache[l_key]
        valid_l = sub_labels[sub_labels >= 0]
        lbl_counts = np.bincount(valid_l, minlength=3)
        lbl_shares = (lbl_counts / max(len(valid_l), 1)).tolist()

        clf = XGBoostRegimeClassifier(
            n_estimators=cfg.study1_encoders.xgb_n_estimators,
            max_depth=cfg.study1_encoders.xgb_max_depth,
            learning_rate=cfg.study1_encoders.xgb_learning_rate,
            switch_threshold=cfg.study1_encoders.regime_switch_threshold,
            balance_classes=bool(g_cfg["balance_classes"]),
            confirm_bars=int(g_cfg["confirm_bars"]),
            random_state=43,
        )
        clf.fit(sub_states, sub_labels)
        sub_pred_r, _ = clf.predict_regimes_causal(sub_states, initial_regime=REGIME_SIDEWAYS)
        val_pred_r, _ = clf.predict_regimes_causal(val_states, initial_regime=int(sub_pred_r[-1]))

        ens = DynamicRegimeEnsemble(
            state_dim=cfg.study1_encoders.cross_attn_dim,
            experts_per_regime=cfg.study1_ppo.experts_per_regime,
            rolling_sharpe_window_bars=cfg.study1_ppo.rolling_sharpe_window_bars,
            bars_per_year=cfg.evaluation.btc_bars_per_year,
            base_seed=1000,
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
                "include_prev_weight": use_m1,
                "respect_segment_boundaries": use_m1,
                "device": device,
            },
        )
        ens.train_all_pools(
            train_states=sub_states,
            train_regimes=sub_pred_r,
            next_bar_returns=next_ret_f1[sub_tr_idx],
            cost_rates=costs_f1[sub_tr_idx],
            total_rollouts=ppo_rollouts,
        )
        _, val_acts, w_val = ens.rollout_causal_policy(
            states=val_states,
            active_regimes=val_pred_r,
            bar_asset_returns=next_ret_f1[val_idx],
            one_way_cost_rates=costs_f1[val_idx],
            temperature=cfg.study1_ppo.ensemble_temperature,
            num_experts_subset=cfg.study1_ppo.experts_per_regime,
            warmup_states=sub_states,
            warmup_returns=next_ret_f1[sub_tr_idx],
            warmup_costs=costs_f1[sub_tr_idx],
        )
        val_bt = bt.run_decisions(df_val, w_val, phase=2)
        pre_sr = _pre_cost_sharpe(df_val, w_val, cfg.evaluation.btc_bars_per_year)
        val_turnover = float(np.mean(np.abs(np.diff(w_val, prepend=0.0))))
        val_switches = float(np.mean(val_pred_r[1:] != val_pred_r[:-1])) if len(val_pred_r) > 1 else 0.0
        val_r_counts = np.bincount(val_pred_r, minlength=3).tolist()
        act_counts = np.bincount(val_acts, minlength=5).tolist()

        row = {
            "candidate_id": cid,
            "in_selection_pool": in_pool,
            "label_mode": l_key,
            "tau_trend": round(tau_val, 4),
            "markov_prev_weight": use_m1,
            "gate_mode": g_key,
            "balance_classes": bool(g_cfg["balance_classes"]),
            "confirm_bars": int(g_cfg["confirm_bars"]),
            "sub_train_label_counts": lbl_counts.tolist(),
            "sub_train_label_shares": [round(float(x), 4) for x in lbl_shares],
            "sub_train_pred_regime_counts": np.bincount(sub_pred_r, minlength=3).tolist(),
            "val_pred_regime_counts": val_r_counts,
            "val_regime_switches_per_bar": round(val_switches, 4),
            "val_action_counts": act_counts,
            "val_mean_turnover": round(val_turnover, 4),
            "val_pre_cost_sharpe": round(pre_sr, 3),
            "val_sharpe": round(float(val_bt.summary.sharpe), 3),
            "val_sortino": round(float(val_bt.summary.sortino), 3),
            "val_cum_return_pct": round(float(val_bt.summary.cum_return_pct), 2),
            "val_mdd_pct": round(float(val_bt.summary.mdd_pct), 2),
            "elapsed_sec": round(time.perf_counter() - t0, 2),
        }
        print("[D-15 Row]", json.dumps(row, ensure_ascii=False), flush=True)
        rows.append(row)

    # Apply the pre-registered selection rule from DECISIONS.md (commit 84ec746)
    pool_rows = [
        r for r in rows
        if bool(r["in_selection_pool"])
        and min(float(x) for x in r["sub_train_label_shares"]) >= 0.10  # type: ignore[union-attr]
    ]
    best_val_sharpe = max(float(r["val_sharpe"]) for r in pool_rows)  # type: ignore[arg-type]
    near_tied = [r for r in pool_rows if best_val_sharpe - float(r["val_sharpe"]) <= 0.05]  # type: ignore[arg-type]

    label_pref = {"L2_sqrtL_q60": 0, "L1_ols_q60": 1}
    gate_pref = {"G1": 0, "G2": 1, "G0": 2}
    near_tied.sort(
        key=lambda r: (
            float(r["val_mean_turnover"]),  # type: ignore[arg-type]
            label_pref.get(str(r["label_mode"]), 9),
            gate_pref.get(str(r["gate_mode"]), 9),
        )
    )
    winner = near_tied[0] if len(near_tied) > 1 else max(pool_rows, key=lambda r: float(r["val_sharpe"]))  # type: ignore[arg-type]

    summary = {
        "script_path": "scripts/calibrate_study1_d15_fixes.py",
        "preregistered_in_commit": args.preregistered_commit,
        "executed_on_git_state": read_git_state().label,
        "selection_criterion_pre_registered": True,
        "test_fold_accessed": False,
        "sub_train_window_utc": [str(df_all.index[sub_tr_idx[0]]), str(df_all.index[sub_tr_idx[-1]])],
        "val_window_utc": [str(df_all.index[val_idx[0]]), str(df_all.index[val_idx[-1]])],
        "sub_train_bars": int(len(sub_tr_idx)),
        "val_bars": int(len(val_idx)),
        "ppo_rollouts_per_fold": ppo_rollouts,
        "selected_candidate_id": winner["candidate_id"],
        "selected_candidate_summary": winner,
        "candidate_results": rows,
    }
    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[D-15 Calibration] Selected={winner['candidate_id']} -> saved {out_file}", flush=True)


if __name__ == "__main__":
    main()
