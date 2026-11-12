"""Evaluate PPO and parameter-interpolation baselines; keep unapproved budgets provisional."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from ts.baselines.models import (
    BuyAndHoldBaseline,
    MAMLMetaPPOBaseline,
    SinglePPOBaseline,
    StaticEnsembleBaseline,
)
from ts.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, SystemConfig
from ts.data.binance_fetcher import BinanceDataFetcher
from ts.eval.backtester import CausalPortfolioBacktester
from ts.eval.cost_model import compute_phase1_cost_rates, compute_phase2_bar_base_cost_rates
from ts.eval.metrics import compute_crisis_window_loss
from ts.eval.significance import ledoit_wolf_and_block_bootstrap_test
from ts.eval.walk_forward import generate_phase1_kfold_splits, generate_phase2_walk_forward_folds
from ts.ledger import NumberLedger, RunContext
from ts.study1.encoders import TechnicalIndicatorCalculator
from ts.study1.regime_labeling import SMARegimeLabeler, TrendScanningLabeler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Baselines, Crisis Windows, and Ledoit-Wolf Significance.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--ppo-rollouts", type=int, default=8)
    args = parser.parse_args()

    cfg = SystemConfig.from_json(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = PROJECT_ROOT / "data" / "cache"
    out_dir = PROJECT_ROOT / "runs" / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = "runs/baselines/metrics.json"
    script_rel = "scripts/run_baselines_and_eval.py"

    binance = BinanceDataFetcher(cache_dir=cache_dir)
    df_all, _ = binance.fetch_ohlcv(
        symbol=cfg.data.primary_symbol,
        interval=cfg.data.bar_interval,
        start_utc=cfg.data.warmup_start,
        end_utc=cfg.data.eval_end,
        realign_grid=False,
    )

    tech_all = TechnicalIndicatorCalculator(zscore_window=cfg.study1_encoders.tech_zscore_window).compute(df_all).to_numpy(dtype=np.float32)
    closes = df_all["close"].to_numpy(dtype=np.float64)
    next_ret_all = np.zeros(len(df_all), dtype=np.float64)
    next_ret_all[:-1] = (closes[1:] / np.maximum(closes[:-1], 1e-8)) - 1.0

    p1_costs_all = compute_phase1_cost_rates(len(df_all), fee_rate=cfg.evaluation.phase1_fixed_fee_rate)
    p2_costs_all = compute_phase2_bar_base_cost_rates(
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

    p2_idx_list: List[np.ndarray] = []
    p2_w_single: List[np.ndarray] = []
    p2_w_static: List[np.ndarray] = []
    p2_w_maml: List[np.ndarray] = []

    state_dim = tech_all.shape[1]

    for fold in folds:
        tr_idx = fold.train_indices
        te_idx = fold.test_indices
        p2_idx_list.append(te_idx)

        tr_labels = TrendScanningLabeler().fit_label_train_fold(df_all["close"].iloc[tr_idx])

        single = SinglePPOBaseline(state_dim=state_dim, seed=42 + fold.fold_id, device=device)
        single.fit(tech_all[tr_idx], next_ret_all[tr_idx], p2_costs_all[tr_idx], rollouts=args.ppo_rollouts)
        p2_w_single.append(single.predict_weights(tech_all[te_idx]))

        static_ens = StaticEnsembleBaseline(
            state_dim=state_dim, n_agents=cfg.study1_ppo.experts_per_regime, base_seed=200 + 10 * fold.fold_id, device=device
        )
        static_ens.fit(tech_all[tr_idx], next_ret_all[tr_idx], p2_costs_all[tr_idx], rollouts=args.ppo_rollouts)
        p2_w_static.append(static_ens.predict_weights(tech_all[te_idx]))

        maml = MAMLMetaPPOBaseline(state_dim=state_dim, seed=700 + fold.fold_id, device=device)
        maml.fit(tech_all[tr_idx], tr_labels, next_ret_all[tr_idx], p2_costs_all[tr_idx])
        p2_w_maml.append(
            maml.predict_weights_causal(tech_all[te_idx], next_ret_all[te_idx], p2_costs_all[te_idx])
        )

    cat_te_idx = np.concatenate(p2_idx_list)
    eval_df = df_all.iloc[cat_te_idx].copy()
    bt = CausalPortfolioBacktester(
        initial_capital=cfg.evaluation.initial_capital_usdt,
        bars_per_year=cfg.evaluation.btc_bars_per_year,
    )

    bh_w = BuyAndHoldBaseline().predict_weights(len(eval_df))
    res_p2 = {
        "BH": bt.run(eval_df, bh_w, phase=2),
        "SINGLE_PPO": bt.run_decisions(eval_df, np.concatenate(p2_w_single), phase=2),
        "STATIC_ENS": bt.run_decisions(eval_df, np.concatenate(p2_w_static), phase=2),
        "MAML": bt.run_decisions(eval_df, np.concatenate(p2_w_maml), phase=2),
    }

    # Phase 1 K-Fold CV evaluation (`Table 6.6`)
    p1_splits = generate_phase1_kfold_splits(cat_te_idx, n_splits=cfg.evaluation.phase1_kfold_splits)
    idx_to_loc = {int(g): i for i, g in enumerate(cat_te_idx)}
    p1_single_w = np.zeros(len(cat_te_idx), dtype=np.float64)
    p1_static_w = np.zeros(len(cat_te_idx), dtype=np.float64)
    p1_maml_w = np.zeros(len(cat_te_idx), dtype=np.float64)

    for sp_i, (p1_tr, p1_te) in enumerate(p1_splits):
        sma_labels = SMARegimeLabeler().fit_label_train_fold(df_all["close"].iloc[p1_tr])
        s_p1 = SinglePPOBaseline(state_dim=state_dim, seed=10 + sp_i, device=device).fit(
            tech_all[p1_tr], next_ret_all[p1_tr], p1_costs_all[p1_tr], rollouts=args.ppo_rollouts
        )
        st_p1 = StaticEnsembleBaseline(state_dim=state_dim, n_agents=5, base_seed=50 + 10 * sp_i, device=device).fit(
            tech_all[p1_tr], next_ret_all[p1_tr], p1_costs_all[p1_tr], rollouts=args.ppo_rollouts
        )
        m_p1 = MAMLMetaPPOBaseline(state_dim=state_dim, seed=90 + sp_i, device=device).fit(
            tech_all[p1_tr], sma_labels, next_ret_all[p1_tr], p1_costs_all[p1_tr]
        )

        ws = s_p1.predict_weights(tech_all[p1_te])
        wst = st_p1.predict_weights(tech_all[p1_te])
        wm = m_p1.predict_weights_causal(tech_all[p1_te], next_ret_all[p1_te], p1_costs_all[p1_te])
        for j_pos, g_i in enumerate(p1_te):
            loc = idx_to_loc[int(g_i)]
            p1_single_w[loc] = ws[j_pos]
            p1_static_w[loc] = wst[j_pos]
            p1_maml_w[loc] = wm[j_pos]

    res_p1 = {
        "BH": bt.run(eval_df, bh_w, phase=1),
        "SINGLE_PPO": bt.run_decisions(eval_df, p1_single_w, phase=1),
        "STATIC_ENS": bt.run_decisions(eval_df, p1_static_w, phase=1),
        "MAML": bt.run_decisions(eval_df, p1_maml_w, phase=1),
    }

    # Table 6.8 Crisis Windows for Static Ensemble RL (filling Luna blank + re-verifying FTX)
    crisis_static = {}
    for w_name, (c_start, c_end) in cfg.evaluation.crisis_windows.items():
        crisis_static[w_name] = compute_crisis_window_loss(
            res_p2["STATIC_ENS"].equity_curve, c_start, c_end, window_name=w_name
        ).to_dict()

    # Ledoit-Wolf + Circular Block Bootstrap (5,000 reps, block=24) if Study 3 OOS returns exist
    sig_results = {}
    s3_npz_path = PROJECT_ROOT / "runs" / "study3" / "student_oos_predictions.npz"
    if s3_npz_path.exists():
        s3_npz = np.load(s3_npz_path)
        prop_ret = s3_npz["phase2_net_returns"]
        for key, bt_obj in res_p2.items():
            if len(bt_obj.bar_net_returns) == len(prop_ret):
                sig = ledoit_wolf_and_block_bootstrap_test(
                    returns_proposed=prop_ret,
                    returns_baseline=bt_obj.bar_net_returns,
                    baseline_name=key,
                    bars_per_year=cfg.evaluation.btc_bars_per_year,
                    block_size=cfg.evaluation.bootstrap_block_bars,
                    n_bootstrap=cfg.evaluation.bootstrap_replications,
                )
                sig_results[key] = sig.to_dict()

    payload = {
        "config_sha256": cfg.config_sha256,
        "phase1_table_6_6": {k: v.summary.to_dict() for k, v in res_p1.items()},
        "phase2_table_6_7": {k: v.summary.to_dict() for k, v in res_p2.items()},
        "crisis_static_ensemble_table_6_8": crisis_static,
        "ledoit_wolf_significance": sig_results,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    ledger = NumberLedger(run_context=RunContext(stage="baselines", flags={
        "quick_verify": False, "reduced_training_budget": True}))
    # T-102 has not yet established a converged full-training budget. Until that
    # budget is approved, these comparisons remain provisional regardless of the
    # CLI rollout count. Do not infer convergence from an arbitrary number.
    rec_kw = dict(script_path=script_rel, config_sha256=cfg.config_sha256, run_log_path=log_path)

    for ph_num, res_map in [(1, res_p1), (2, res_p2)]:
        tbl = "Table 6.6" if ph_num == 1 else "Table 6.7"
        for m_key, bt_obj in res_map.items():
            sm = bt_obj.summary
            ledger.record(f"T6.{5 + ph_num}_{m_key}_CUMRET", tbl, f"{m_key} Phase {ph_num} CumReturn", sm.cum_return_pct, "{:.2f}", "%", **rec_kw)
            ledger.record(f"T6.{5 + ph_num}_{m_key}_CAGR", tbl, f"{m_key} Phase {ph_num} CAGR", sm.cagr_pct, "{:.2f}", "%", **rec_kw)
            ledger.record(f"T6.{5 + ph_num}_{m_key}_SHARPE", tbl, f"{m_key} Phase {ph_num} Sharpe", sm.sharpe, "{:.2f}", "", **rec_kw)
            ledger.record(f"T6.{5 + ph_num}_{m_key}_SORTINO", tbl, f"{m_key} Phase {ph_num} Sortino", sm.sortino, "{:.2f}", "", **rec_kw)
            ledger.record(f"T6.{5 + ph_num}_{m_key}_MDD", tbl, f"{m_key} Phase {ph_num} MDD", sm.mdd_pct, "{:.2f}", "%", **rec_kw)

    ledger.record("T6.8_LUNA_STATIC_ENS_MDD", "Table 6.8", "Luna/Rate-Hike Static Ensemble Crisis Drawdown", float(crisis_static["luna_rate_hike"]["window_mdd_pct"]), "{:.2f}", "%", **rec_kw)
    ledger.record("T6.8_FTX_STATIC_ENS_MDD", "Table 6.8", "FTX Collapse Static Ensemble Crisis Drawdown", float(crisis_static["ftx_collapse"]["window_mdd_pct"]), "{:.2f}", "%", **rec_kw)

    for b_key, sig_d in sig_results.items():
        ledger.record(f"CH6_SIG_{b_key}_BOOT_P", "Ch.VI Sec 6", f"Ledoit-Wolf Block Bootstrap p-value vs {b_key}", float(sig_d["bootstrap_p_value"]), "{:.4f}", "", **rec_kw)

    print("[Baselines & Eval] Completed and recorded into ledger/numbers.csv.")


if __name__ == "__main__":
    main()
