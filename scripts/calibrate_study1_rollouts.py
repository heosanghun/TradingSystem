"""
Fold 1 internal validation rollout budget calibration for Study 1 (T-102).

Splits Fold 1's 12-month training window (`2020-10-01 .. 2021-09-30`, 8,741 bars) into:
- Sub-train (first 10 months, `2020-10-01 .. 2021-07-31`, 7,283 bars)
- Validation (last 2 months, `2021-08-01 .. 2021-09-30`, 1,458 bars)

Fold 1 out-of-sample test bars (`2021-10-01` onward) are sliced off before feature
computation and are never accessed. Sweeps on-policy PPO+PER rollout budgets across
`[8, 32, 64, 128, 256, 512]` and writes the summary JSON to
`runs/study1/fold1_val_rollout_calibration.json`.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import time
from pathlib import Path

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
from ts.study1.ppo_expert import ACTION_WEIGHTS, compute_regime_rewards
from ts.study1.regime_classifier import XGBoostRegimeClassifier
from ts.study1.regime_labeling import (
    REGIME_BEAR,
    REGIME_BULL,
    REGIME_SIDEWAYS,
    TrendScanningLabeler,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce Fold 1 internal validation rollout budget calibration (T-102)."
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "runs" / "study1" / "fold1_val_rollout_calibration.json"),
    )
    parser.add_argument(
        "--rollouts-grid",
        type=int,
        nargs="+",
        default=[8, 32, 64, 128, 256, 512],
    )
    parser.add_argument(
        "--keep-rollout-logs",
        action="store_true",
        help="Keep per-expert rollout trajectory JSON files under runs/study1/training_logs/fold1_val_calibration/.",
    )
    args = parser.parse_args()

    cfg = SystemConfig.from_json(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[T-102 Calibration] Device: {device}", flush=True)

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

    sub_tr_mask = fold1_ts < val_cutoff
    val_mask = fold1_ts >= val_cutoff
    sub_tr_idx = fold1_tr_idx[sub_tr_mask]
    val_idx = fold1_tr_idx[val_mask]
    assert np.max(sub_tr_idx) < np.min(val_idx)
    assert np.max(val_idx) < np.min(fold1.test_indices)

    print(
        f"[T-102 Calibration] Sub-train: {len(sub_tr_idx)} bars "
        f"({df_all.index[sub_tr_idx[0]]} .. {df_all.index[sub_tr_idx[-1]]}) | "
        f"Validation (last 2m of Fold 1 train): {len(val_idx)} bars "
        f"({df_all.index[val_idx[0]]} .. {df_all.index[val_idx[-1]]})",
        flush=True,
    )

    # Slice df up to end of Fold 1 train so test period is physically absent
    end_pos = int(val_idx[-1]) + 1
    df_f1 = df_all.iloc[:end_pos].copy()

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
    print(
        f"[T-102 Calibration] Rendered {len(df_f1)} uint8 charts + tech in {time.perf_counter() - t_s:.2f}s",
        flush=True,
    )

    news_path = PROJECT_ROOT / "data" / "news_headlines.parquet"
    if news_path.exists():
        t_s = time.perf_counter()
        news_store = PointInTimeNewsStore.from_file(news_path)
        news_store.df = news_store.df[news_store.df["published_at"] <= df_f1.index[-1]].reset_index(drop=True)
        print(f"[T-102 Calibration] Scoring {len(news_store.df)} Fold 1 headlines with FinBERT...", flush=True)
        news_store.populate_missing_logits_with_finbert(
            model_name=cfg.study3.finbert_model,
            batch_size=32,
            device=device,
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[T-102 Calibration] FinBERT scoring done in {time.perf_counter() - t_s:.2f}s", flush=True)
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

    # Purged Trend Scanning labels inside Sub-train (no leakage into Validation)
    labeler = TrendScanningLabeler(
        horizon_min=cfg.study1_encoders.trend_scan_min_horizon,
        horizon_max=cfg.study1_encoders.trend_scan_max_horizon,
        t_threshold=cfg.study1_encoders.trend_scan_t_threshold,
    )
    sub_tr_labels = labeler.fit_label_train_fold(df_f1["close"].iloc[sub_tr_idx])

    t_s = time.perf_counter()
    torch.manual_seed(42)
    encoder = MultimodalStateEncoder(pretrained_resnet=True, modality_mode="multimodal")
    encoder.pretrain_on_fold_regimes(
        chart_imgs=charts_f1,
        tech_x=tech_f1,
        senti_x=senti_f1,
        regime_labels=sub_tr_labels,
        epochs=4,
        batch_size=32,
        device=device,
        indices=sub_tr_idx,
    )
    sub_tr_states, _ = encoder.encode_dataset(
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
    print(
        f"[T-102 Calibration] Encoder pretrain (4 ep) + encode done in {time.perf_counter() - t_s:.2f}s",
        flush=True,
    )

    clf = XGBoostRegimeClassifier(
        n_estimators=cfg.study1_encoders.xgb_n_estimators,
        max_depth=cfg.study1_encoders.xgb_max_depth,
        learning_rate=cfg.study1_encoders.xgb_learning_rate,
        switch_threshold=cfg.study1_encoders.regime_switch_threshold,
        random_state=43,
    )
    clf.fit(sub_tr_states, sub_tr_labels)
    sub_tr_regimes, _ = clf.predict_regimes_causal(sub_tr_states, initial_regime=REGIME_SIDEWAYS)
    val_regimes, _ = clf.predict_regimes_causal(val_states, initial_regime=int(sub_tr_regimes[-1]))

    bt = CausalPortfolioBacktester(
        initial_capital=cfg.evaluation.initial_capital_usdt,
        bars_per_year=cfg.evaluation.btc_bars_per_year,
    )
    df_val = df_f1.iloc[val_idx].copy()

    rand_turnovers = []
    rand_sharpes = []
    for r_sd in range(20):
        r_rng = np.random.default_rng(100 + r_sd)
        w_rand = ACTION_WEIGHTS[r_rng.integers(0, 5, size=len(val_idx))]
        dw_rand = np.abs(np.diff(w_rand, prepend=0.0))
        rand_turnovers.append(float(np.mean(dw_rand)))
        rand_sharpes.append(float(bt.run_decisions(df_val, w_rand, phase=2).summary.sharpe))

    rollout_grid = [int(r) for r in args.rollouts_grid]
    results = []
    calib_log_dir = PROJECT_ROOT / "runs" / "study1" / "training_logs" / "fold1_val_calibration"
    if calib_log_dir.exists():
        shutil.rmtree(calib_log_dir)

    for r_budget in rollout_grid:
        t0 = time.perf_counter()
        ens = DynamicRegimeEnsemble(
            state_dim=128,
            experts_per_regime=5,
            rolling_sharpe_window_bars=cfg.study1_ppo.rolling_sharpe_window_bars,
            bars_per_year=cfg.evaluation.btc_bars_per_year,
            base_seed=1000,
            ppo_kwargs={"device": device},
        )
        ens.train_all_pools(
            train_states=sub_tr_states,
            train_regimes=sub_tr_regimes,
            next_bar_returns=next_ret_f1[sub_tr_idx],
            cost_rates=costs_f1[sub_tr_idx],
            total_rollouts=r_budget,
            log_dir=calib_log_dir if args.keep_rollout_logs else None,
            fold_id=1,
            tag_prefix=f"calib_r{r_budget}",
        )
        elapsed = time.perf_counter() - t0

        entropies = []
        sampled_turnovers = []
        greedy_turnovers = []
        v_losses = []
        regime_final_rewards = {0: [], 1: [], 2: []}
        for r_id in (REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS):
            for exp in ens.pools[r_id]:
                last_log = exp.last_training_logs_[-1]
                entropies.append(last_log["policy_entropy"])
                sampled_turnovers.append(last_log["mean_turnover"])
                greedy_turnovers.append(last_log["greedy_mean_turnover"])
                v_losses.append(last_log["value_loss"])
                regime_final_rewards[r_id].append(last_log["mean_reward"])

        val_probs, val_acts, w_val = ens.rollout_causal_policy(
            states=val_states,
            active_regimes=val_regimes,
            bar_asset_returns=next_ret_f1[val_idx],
            one_way_cost_rates=costs_f1[val_idx],
            temperature=cfg.study1_ppo.ensemble_temperature,
            num_experts_subset=5,
            warmup_states=sub_tr_states,
            warmup_returns=next_ret_f1[sub_tr_idx],
            warmup_costs=costs_f1[sub_tr_idx],
        )
        val_bt = bt.run_decisions(df_val, w_val, phase=2)
        val_dw = float(np.mean(np.abs(np.diff(w_val, prepend=0.0))))
        ens_prob_ent = float(
            np.mean(-np.sum(val_probs * np.log(np.clip(val_probs, 1e-8, 1.0)), axis=-1))
        )
        act_counts = np.bincount(val_acts, minlength=5)
        act_dist = [round(float(c) / len(val_acts), 4) for c in act_counts]

        val_mean_net_ret_bps = float(np.mean(val_bt.bar_net_returns) * 1e4)
        rew_by_reg = {
            r_id: compute_regime_rewards(r_id, next_ret_f1[val_idx], w_val, costs_f1[val_idx])[0]
            for r_id in (REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS)
        }
        val_active_rew = np.array(
            [rew_by_reg[int(val_regimes[t])][t] for t in range(len(val_idx))], dtype=np.float64
        )

        row = {
            "rollouts": r_budget,
            "train_15experts_sec": round(elapsed, 2),
            "train_final_entropy_mean": round(float(np.mean(entropies)), 4),
            "train_final_sampled_turnover": round(float(np.mean(sampled_turnovers)), 4),
            "train_final_greedy_turnover": round(float(np.mean(greedy_turnovers)), 4),
            "train_final_value_loss": round(float(np.mean(v_losses)), 5),
            "train_regime_rewards": {
                str(k): round(float(np.mean(v)), 5) for k, v in regime_final_rewards.items()
            },
            "val_ens_policy_entropy": round(ens_prob_ent, 4),
            "val_action_dist": act_dist,
            "val_mean_turnover": round(val_dw, 4),
            "val_regime_reward_mean": round(float(np.mean(val_active_rew)), 5),
            "val_mean_net_ret_bps": round(val_mean_net_ret_bps, 4),
            "val_cum_return_pct": round(float(val_bt.summary.cum_return_pct), 2),
            "val_sharpe": round(float(val_bt.summary.sharpe), 3),
            "val_mdd_pct": round(float(val_bt.summary.mdd_pct), 2),
        }
        print("[T-102 Row]", json.dumps(row, ensure_ascii=False), flush=True)
        results.append(row)

    out_summary = {
        "script_path": "scripts/calibrate_study1_rollouts.py",
        "git_commit": read_git_state().label,
        "config_sha256": cfg.config_sha256,
        "sub_train_window_utc": [str(df_all.index[sub_tr_idx[0]]), str(df_all.index[sub_tr_idx[-1]])],
        "val_window_utc": [str(df_all.index[val_idx[0]]), str(df_all.index[val_idx[-1]])],
        "test_fold_accessed": False,
        "selection_criterion_pre_registered": False,
        "uniform_entropy_ln5": round(math.log(5.0), 4),
        "random_baseline_val_turnover": round(float(np.mean(rand_turnovers)), 4),
        "random_baseline_val_sharpe_mean": round(float(np.mean(rand_sharpes)), 3),
        "sub_train_bars": int(len(sub_tr_idx)),
        "val_bars": int(len(val_idx)),
        "grid_results": results,
    }
    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(out_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[T-102 Calibration] Saved summary ->", out_file, flush=True)


if __name__ == "__main__":
    main()
