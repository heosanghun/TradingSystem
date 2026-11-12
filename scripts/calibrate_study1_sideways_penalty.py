"""
Choose the sideways-regime turnover penalty C of Eq. (III.19) on data inside the first
training fold (D-17). The thesis defines the reward but gives no value for C.

Protocol (written to collab/DECISIONS.md before this script was run):
- Only the first training fold (2020-10-01 .. 2021-09-30) is loaded. Test folds are cut off
  before any feature is computed.
- Two validation splits inside that fold, each trained on the months before it:
    split A: train .. 2021-05-31, validate 2021-06-01 .. 2021-07-31
    split B: train .. 2021-07-31, validate 2021-08-01 .. 2021-09-30
- Everything except C is the frozen D-15 configuration.
- For each C in the grid the Phase 2 Sharpe ratio after costs is computed on both
  validation windows. The selected C has the highest mean of the two. If several are
  within 0.05 of the best mean, the one with the lowest mean validation turnover is
  taken, then the smallest C.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
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
from ts.eval.walk_forward import generate_phase2_walk_forward_folds
from ts.ledger import read_git_state
from ts.study1.chart_renderer import CandlestickChartRenderer
from ts.study1.encoders import MultimodalStateEncoder, TechnicalIndicatorCalculator
from ts.study1.ensemble import DynamicRegimeEnsemble
from ts.study1.regime_classifier import XGBoostRegimeClassifier
from ts.study1.regime_labeling import REGIME_SIDEWAYS, TrendScanningLabeler

C_GRID = (0.002, 0.005, 0.01, 0.02, 0.05)
SPLITS = (
    ("A", "2021-06-01T00:00:00Z", "2021-08-01T00:00:00Z"),
    ("B", "2021-08-01T00:00:00Z", "2021-10-01T00:00:00Z"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the sideways turnover penalty C inside training fold 1.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "runs" / "study1" / "fold1_val_sideways_penalty.json"))
    parser.add_argument("--preregistered-commit", type=str, default="")
    parser.add_argument("--c-grid", type=str, default="", help="Comma-separated C values (default: the D-17 stage 1 grid).")
    parser.add_argument("--confirm-grid", type=str, default="",
                        help="Comma-separated regime confirmation lengths in bars (D-17 stage 2). Default: the configured value only.")
    args = parser.parse_args()

    cfg = SystemConfig.from_json(args.config)
    s1e, s1p = cfg.study1_encoders, cfg.study1_ppo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rollouts = int(s1p.default_rollouts_per_fold)
    c_grid = tuple(float(x) for x in args.c_grid.split(",")) if args.c_grid else C_GRID
    k_grid = tuple(int(x) for x in args.confirm_grid.split(",")) if args.confirm_grid else (int(s1e.regime_confirm_bars),)
    candidates = [(c, k) for k in k_grid for c in c_grid]

    df_all, _ = BinanceDataFetcher(cache_dir=PROJECT_ROOT / "data" / "cache").fetch_ohlcv(
        symbol=cfg.data.primary_symbol, interval=cfg.data.bar_interval,
        start_utc=cfg.data.warmup_start, end_utc=cfg.data.eval_end, realign_grid=False)
    folds = generate_phase2_walk_forward_folds(
        pd.DatetimeIndex(df_all.index), first_train_start=cfg.data.first_train_start,
        train_months=cfg.evaluation.walk_forward_train_months, test_months=cfg.evaluation.walk_forward_test_months,
        step_months=cfg.evaluation.walk_forward_step_months, num_folds=cfg.evaluation.walk_forward_num_folds)
    fold1 = folds[0]
    tr_idx = fold1.train_indices
    end_pos = int(tr_idx[-1]) + 1
    assert end_pos <= int(np.min(fold1.test_indices))
    df_f1 = df_all.iloc[:end_pos].copy()     # nothing from 2021-10-01 onward is kept
    del df_all
    ts_tr = df_f1.index[tr_idx]

    charts = CandlestickChartRenderer(window_bars=s1e.candlestick_window_bars, image_size=s1e.image_size).render_series_uint8(df_f1)
    tech = TechnicalIndicatorCalculator(zscore_window=s1e.tech_zscore_window).compute(df_f1).to_numpy(dtype=np.float32)
    news = PointInTimeNewsStore.from_file(PROJECT_ROOT / "data" / "news_headlines.parquet")
    news.df = news.df[news.df["published_at"] <= df_f1.index[-1]].reset_index(drop=True)
    if news.df[["logit_neg", "logit_neu", "logit_pos"]].isna().any().any():
        raise SystemExit("news file has no sentiment logits; run tools/score_news_sentiment_llm.py first")
    senti = news.build_hourly_sentiment_summary(pd.DatetimeIndex(df_f1.index), ewma_span_bars=s1e.senti_ewma_span_bars).to_numpy(dtype=np.float32)

    closes = df_f1["close"].to_numpy(dtype=np.float64)
    next_ret = np.zeros(len(df_f1), dtype=np.float64)
    next_ret[:-1] = closes[1:] / np.maximum(closes[:-1], 1e-8) - 1.0
    costs = compute_phase2_bar_base_cost_rates(df_f1, fee_rate=cfg.evaluation.phase2_fee_rate,
                                               atr_period=cfg.evaluation.phase2_atr_period, kappa=cfg.evaluation.phase2_kappa)
    bt = CausalPortfolioBacktester(initial_capital=cfg.evaluation.initial_capital_usdt, bars_per_year=cfg.evaluation.btc_bars_per_year)
    ppo_kwargs = {
        "hidden_dims": s1p.hidden_dims, "lr_actor": s1p.lr_actor, "lr_critic": s1p.lr_critic, "gamma": s1p.gamma,
        "gae_lambda": s1p.gae_lambda, "clip_eps": s1p.clip_eps, "entropy_coef": s1p.entropy_coef,
        "value_coef": s1p.value_coef, "per_alpha": s1p.per_alpha, "per_beta_start": s1p.per_beta_start,
        "per_beta_end": s1p.per_beta_end, "minibatch_size": s1p.minibatch_size, "ppo_epochs": s1p.ppo_epochs,
        "include_prev_weight": bool(s1p.include_prev_weight), "respect_segment_boundaries": bool(s1p.respect_segment_boundaries),
        "device": device,
    }

    rows: List[Dict[str, object]] = []
    for split_name, val_start, val_end in SPLITS:
        v0, v1 = pd.Timestamp(val_start), pd.Timestamp(val_end)
        sub_idx = tr_idx[ts_tr < v0]
        val_idx = tr_idx[(ts_tr >= v0) & (ts_tr < v1)]
        assert len(sub_idx) and len(val_idx) and np.max(sub_idx) < np.min(val_idx)
        df_val = df_f1.iloc[val_idx].copy()

        labeler = TrendScanningLabeler(horizon_min=s1e.trend_scan_min_horizon, horizon_max=s1e.trend_scan_max_horizon,
                                       trend_quantile=s1e.trend_quantile, unit_root_scale=bool(s1e.trend_scan_unit_root_scale))
        labels = labeler.fit_label_train_fold(df_f1["close"].iloc[sub_idx])
        torch.manual_seed(42)
        encoder = MultimodalStateEncoder(pretrained_resnet=True, modality_mode="multimodal")
        encoder.pretrain_on_fold_regimes(chart_imgs=charts, tech_x=tech, senti_x=senti, regime_labels=labels,
                                         epochs=4, batch_size=32, device=device, indices=sub_idx)
        sub_states, _ = encoder.encode_dataset(charts, tech, senti, batch_size=64, device=device, indices=sub_idx)
        val_states, _ = encoder.encode_dataset(charts, tech, senti, batch_size=64, device=device, indices=val_idx)
        encoder.cpu()
        del encoder
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        clf = XGBoostRegimeClassifier(n_estimators=s1e.xgb_n_estimators, max_depth=s1e.xgb_max_depth,
                                      learning_rate=s1e.xgb_learning_rate, switch_threshold=s1e.regime_switch_threshold,
                                      balance_classes=bool(s1e.xgb_balance_classes), confirm_bars=int(s1e.regime_confirm_bars),
                                      random_state=43)
        clf.fit(sub_states, labels)

        for c_val, k_val in candidates:
            t0 = time.perf_counter()
            # The fitted classifier is the same for every candidate; only the gate differs.
            clf.confirm_bars = int(k_val)
            sub_r, _ = clf.predict_regimes_causal(sub_states, initial_regime=REGIME_SIDEWAYS)
            val_r, _ = clf.predict_regimes_causal(val_states, initial_regime=int(sub_r[-1]))
            ens = DynamicRegimeEnsemble(state_dim=s1e.cross_attn_dim, experts_per_regime=s1p.experts_per_regime,
                                        rolling_sharpe_window_bars=s1p.rolling_sharpe_window_bars,
                                        bars_per_year=cfg.evaluation.btc_bars_per_year, base_seed=1000, ppo_kwargs=dict(ppo_kwargs))
            ens.train_all_pools(train_states=sub_states, train_regimes=sub_r, next_bar_returns=next_ret[sub_idx],
                                cost_rates=costs[sub_idx], total_rollouts=rollouts, sideways_turnover_penalty=float(c_val))
            _, val_acts, w_val = ens.rollout_causal_policy(
                states=val_states, active_regimes=val_r, bar_asset_returns=next_ret[val_idx], one_way_cost_rates=costs[val_idx],
                temperature=s1p.ensemble_temperature, num_experts_subset=s1p.experts_per_regime,
                warmup_states=sub_states, warmup_returns=next_ret[sub_idx], warmup_costs=costs[sub_idx])
            res = bt.run_decisions(df_val, w_val, phase=2)
            gross = res.bar_gross_returns
            row = {
                "split": split_name, "C": c_val, "confirm_bars": int(k_val),
                "val_regime_switches_per_bar": round(float(np.mean(val_r[1:] != val_r[:-1])), 4), "sub_train_bars": int(len(sub_idx)), "val_bars": int(len(val_idx)),
                "val_pred_regime_counts": np.bincount(val_r, minlength=3).tolist(),
                "val_action_counts": np.bincount(val_acts, minlength=5).tolist(),
                "val_mean_turnover": round(float(np.mean(np.abs(np.diff(w_val, prepend=0.0)))), 4),
                "val_sharpe": round(float(res.summary.sharpe), 3),
                "val_pre_cost_sharpe": round(float(np.mean(gross) / max(np.std(gross, ddof=1), 1e-12) * np.sqrt(cfg.evaluation.btc_bars_per_year)), 3),
                "val_cum_return_pct": round(float(res.summary.cum_return_pct), 2),
                "val_mdd_pct": round(float(res.summary.mdd_pct), 2),
                "elapsed_sec": round(time.perf_counter() - t0, 1),
            }
            print("[D-17 Row]", json.dumps(row), flush=True)
            rows.append(row)

    table = []
    for c_val, k_val in candidates:
        rs = [r for r in rows if r["C"] == c_val and r["confirm_bars"] == k_val]
        table.append({"C": c_val, "confirm_bars": k_val,
                      "mean_val_sharpe": round(float(np.mean([r["val_sharpe"] for r in rs])), 3),
                      "mean_val_turnover": round(float(np.mean([r["val_mean_turnover"] for r in rs])), 4),
                      "val_sharpe_by_split": {r["split"]: r["val_sharpe"] for r in rs}})
    best = max(t["mean_val_sharpe"] for t in table)
    tied = [t for t in table if best - t["mean_val_sharpe"] <= 0.05]
    tied.sort(key=lambda t: (t["mean_val_turnover"], t["confirm_bars"], t["C"]))
    selected = tied[0]

    git = read_git_state()
    out = {
        "script_path": "scripts/calibrate_study1_sideways_penalty.py", "preregistered_in_commit": args.preregistered_commit,
        "executed_on_git_state": git.label, "test_fold_accessed": False, "ppo_rollouts_per_fold": rollouts,
        "selection_rule": "highest mean Phase 2 Sharpe over the two validation splits; ties within 0.05 -> lowest mean turnover -> smallest C",
        "grid": [list(c) for c in candidates], "splits": [dict(name=s[0], val_start=s[1], val_end_exclusive=s[2]) for s in SPLITS],
        "summary_by_candidate": table, "selected_C": selected["C"], "selected_confirm_bars": selected["confirm_bars"], "rows": rows,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("[D-17 Selected]", json.dumps(selected), flush=True)


if __name__ == "__main__":
    main()
