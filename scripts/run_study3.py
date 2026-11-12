"""
Study 3 Dual-Process Reasoning Distillation and `lambda_3` Sensitivity Sweep.

Measures and records:
- Property 5.1 orthogonal gradient check (`verify_property_5_1_orthogonality`)
- Table 6.6 (Phase 1 Proposed Model) and Table 6.7 (Phase 2 Proposed Model, `lambda_3 = 0.2`)
- Table 6.8: Proposed Model crisis drawdowns in Luna/Rate-Hike and FTX Collapse windows
- Table 6.11: Phase 2 Student Sharpe across `lambda_3 in {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}`
- Table 6.15 Row 4: Teacher-Student action agreement rate on out-of-sample test bars
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from ts.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, SystemConfig
from ts.data.binance_fetcher import BinanceDataFetcher
from ts.data.lob_processor import LOBWindowProcessor
from ts.eval.backtester import CausalPortfolioBacktester
from ts.eval.cost_model import compute_phase1_cost_rates, compute_phase2_bar_base_cost_rates
from ts.eval.metrics import compute_crisis_window_loss
from ts.eval.walk_forward import WalkForwardFold, generate_phase1_kfold_splits, generate_phase2_walk_forward_folds
from ts.ledger import SMOKE_ENV_VAR, NumberLedger, RunContext
from ts.study1.ppo_expert import ACTION_WEIGHTS
from ts.study2.debate import map_continuous_weight_to_discrete_action
from ts.study3.distillation import StudentDistillationTrainer, verify_property_5_1_orthogonality
from ts.study3.teacher_signal import FinBERTReasoningEncoder, build_aligned_distillation_dataset


def _build_bar_lob_windows(
    df_1h: pd.DataFrame,
    df_1m: Optional[pd.DataFrame],
    lob_proc: LOBWindowProcessor,
    *,
    allow_hourly_test_fallback: bool = False,
) -> Tuple[List[List[np.ndarray]], np.ndarray, str]:
    """
    Construct 10 training `(40, 100)` windows per 1h bar and 1 online pre-close `(40, 100)`
    window per 1h bar from 1-minute klines.

    In production runs (`allow_hourly_test_fallback=False`), missing or empty 1-minute data
    raises `RuntimeError` immediately instead of falling back to hourly bars.
    """
    n = len(df_1h)
    per_bar_train: List[List[np.ndarray]] = []
    online_test = np.empty((n, 40, 100), dtype=np.float32)

    if df_1m is None or df_1m.empty:
        if not allow_hourly_test_fallback:
            raise RuntimeError(
                "Study 3 requires 1-minute sub-bar data (df_1m is missing or empty). "
                "Hourly fallback is disabled in production runs."
            )
        mode_tag = "causal_hourly_microstructure_fallback"
        for t in range(n):
            w_t, _ = lob_proc.extract_from_subbar_klines(df_1h, t)
            online_test[t] = w_t
            per_bar_train.append([w_t for _ in range(lob_proc.windows_per_bar)])
        return per_bar_train, online_test, mode_tag

    mode_tag = "subbar_1m_microstructure"
    idx_1m = df_1m.index
    for t in range(n):
        bar_end = df_1h.index[t] + pd.Timedelta(hours=1)
        pos = int(idx_1m.searchsorted(bar_end, side="left")) - 1
        if pos < 0:
            raise RuntimeError(
                f"1-minute sub-bar archive starts at {idx_1m[0]}, after 1h bar close {bar_end}."
            )
        pos = min(pos, len(df_1m) - 1)
        win_latest, _ = lob_proc.extract_from_subbar_klines(df_1m, pos)
        online_test[t] = win_latest

        sub_wins = []
        for k in range(lob_proc.windows_per_bar):
            sub_pos = max(0, pos - (lob_proc.windows_per_bar - 1 - k) * 6)
            w_k, _ = lob_proc.extract_from_subbar_klines(df_1m, sub_pos)
            sub_wins.append(w_k)
        per_bar_train.append(sub_wins)

    return per_bar_train, online_test, mode_tag


def _load_and_validate_teacher_signals(
    *,
    study2_dir: Path,
    total_bars: int,
    required_bar_indices: np.ndarray,
    expected_config_sha256: str,
    allow_test_fallbacks: bool,
) -> Tuple[np.ndarray, List[str], bool, bool]:
    """
    Load Study 2 teacher outputs (`teacher_signals.npz`, `consensus_texts.json`, `metrics.json`)
    and verify completeness, zero failed bars, config hash match, and full fold coverage.
    Returns `(w_final_full, consensus_full, teacher_signals_present, llm_simulated)`.
    """
    t_sig_path = study2_dir / "teacher_signals.npz"
    c_txt_path = study2_dir / "consensus_texts.json"
    s2_metrics_path = study2_dir / "metrics.json"

    if not (t_sig_path.exists() and c_txt_path.exists() and s2_metrics_path.exists()):
        if not allow_test_fallbacks:
            missing = [
                str(p.relative_to(PROJECT_ROOT)) if p.is_relative_to(PROJECT_ROOT) else str(p)
                for p in (t_sig_path, c_txt_path, s2_metrics_path)
                if not p.exists()
            ]
            raise FileNotFoundError(
                f"Study 3 requires completed Study 2 teacher artifacts; missing: {', '.join(missing)}"
            )
        return (
            np.zeros(total_bars, dtype=np.float32),
            ["Test fallback neutral text."] * total_bars,
            False,
            True,
        )

    s2_metrics = json.loads(s2_metrics_path.read_text(encoding="utf-8"))
    s2_status = str(s2_metrics.get("status", "complete"))
    n_failed = int(s2_metrics.get("n_failed_bars", 0))
    s2_cfg_hash = str(s2_metrics.get("config_sha256", ""))
    llm_simulated = bool(s2_metrics.get("is_simulated_llm", False))

    if not allow_test_fallbacks:
        if s2_status != "complete":
            raise RuntimeError(
                f"Study 2 status in {s2_metrics_path} is '{s2_status}' (expected 'complete'); aborting Study 3."
            )
        if n_failed > 0:
            raise RuntimeError(
                f"Study 2 reported n_failed_bars={n_failed} in {s2_metrics_path}; aborting Study 3."
            )
        if s2_cfg_hash and expected_config_sha256 and s2_cfg_hash != expected_config_sha256:
            raise RuntimeError(
                f"Study 2 config_sha256 ({s2_cfg_hash}) does not match current config ({expected_config_sha256})."
            )

    t_npz = np.load(t_sig_path)
    c_list = json.loads(c_txt_path.read_text(encoding="utf-8"))
    bar_indices = np.asarray(t_npz["bar_indices"], dtype=np.int64)
    w_finals = np.asarray(t_npz["w_finals"], dtype=np.float32)

    if len(bar_indices) != len(w_finals) or len(bar_indices) != len(c_list):
        raise RuntimeError(
            f"Teacher signal length mismatch: bar_indices={len(bar_indices)}, "
            f"w_finals={len(w_finals)}, consensus_texts={len(c_list)}."
        )

    present_set = set(int(x) for x in bar_indices.tolist())
    missing_req = [int(i) for i in required_bar_indices.tolist() if int(i) not in present_set]
    if missing_req and not allow_test_fallbacks:
        raise RuntimeError(
            f"Teacher signals do not cover {len(missing_req)} required walk-forward bars "
            f"(first missing bar index: {missing_req[0]})."
        )

    w_final_full = np.zeros(total_bars, dtype=np.float32)
    consensus_full = [""] * total_bars
    for idx_pos, b_i in enumerate(bar_indices):
        bi = int(b_i)
        if 0 <= bi < total_bars:
            w_final_full[bi] = float(w_finals[idx_pos])
            consensus_full[bi] = str(c_list[idx_pos])

    return w_final_full, consensus_full, True, llm_simulated


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Study 3 Distillation and Lambda_3 Sensitivity.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--study2-dir", type=str, default=str(PROJECT_ROOT / "runs" / "study2"))
    parser.add_argument("--no-fetch-1m", action="store_true", help="Load 1m klines strictly from local cache without network calls.")
    parser.add_argument(
        "--allow-test-fallbacks",
        action="store_true",
        help="Test-only mode permitting synthetic/hourly/missing-teacher fallbacks; marks run provisional.",
    )
    args = parser.parse_args()

    if args.allow_test_fallbacks:
        os.environ[SMOKE_ENV_VAR] = "1"

    cfg = SystemConfig.from_json(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = PROJECT_ROOT / "data" / "cache"
    out_dir = PROJECT_ROOT / "runs" / "study3"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = "runs/study3/metrics.json"
    script_rel = "scripts/run_study3.py"

    binance = BinanceDataFetcher(cache_dir=cache_dir)
    df_all, _ = binance.fetch_ohlcv(
        symbol=cfg.data.primary_symbol,
        interval=cfg.data.bar_interval,
        start_utc=cfg.data.warmup_start,
        end_utc=cfg.data.eval_end,
        realign_grid=False,
    )

    folds: List[WalkForwardFold] = generate_phase2_walk_forward_folds(
        pd.DatetimeIndex(df_all.index),
        first_train_start=cfg.data.first_train_start,
        train_months=cfg.evaluation.walk_forward_train_months,
        test_months=cfg.evaluation.walk_forward_test_months,
        step_months=cfg.evaluation.walk_forward_step_months,
        num_folds=cfg.evaluation.walk_forward_num_folds,
    )
    required_bars = np.unique(
        np.concatenate([np.concatenate([f.train_indices, f.test_indices]) for f in folds])
    )

    # Validate Study 2 teacher signals FIRST so missing/stale inputs fail fast
    w_final_full, consensus_full, teacher_present, llm_simulated = _load_and_validate_teacher_signals(
        study2_dir=Path(args.study2_dir),
        total_bars=len(df_all),
        required_bar_indices=required_bars,
        expected_config_sha256=cfg.config_sha256,
        allow_test_fallbacks=args.allow_test_fallbacks,
    )

    # Default path: load 1-minute sub-bars (cache-only if --no-fetch-1m is set)
    df_1m: Optional[pd.DataFrame] = None
    if args.no_fetch_1m:
        cached_1m = sorted(cache_dir.glob(f"{cfg.data.primary_symbol}_1m_*.parquet"))
        if cached_1m:
            df_1m = pd.read_parquet(cached_1m[0])
        elif not args.allow_test_fallbacks:
            raise FileNotFoundError(
                f"No cached 1-minute sub-bar parquet found in {cache_dir} for {cfg.data.primary_symbol}."
            )
    else:
        try:
            df_1m, _ = binance.fetch_ohlcv(
                symbol=cfg.data.primary_symbol,
                interval="1m",
                start_utc=cfg.data.warmup_start,
                end_utc=cfg.data.eval_end,
                realign_grid=False,
            )
        except Exception as exc:
            if not args.allow_test_fallbacks:
                raise RuntimeError(f"Failed to acquire 1-minute sub-bar data for Study 3: {exc}") from exc
            df_1m = None

    lob_proc = LOBWindowProcessor(
        seq_len=cfg.study3.lob_seq_len,
        num_features=cfg.study3.lob_input_dim,
        windows_per_bar=cfg.data.lob_windows_per_bar,
    )
    per_bar_train_wins, online_wins_all, lob_mode = _build_bar_lob_windows(
        df_all,
        df_1m,
        lob_proc,
        allow_hourly_test_fallback=args.allow_test_fallbacks,
    )

    finbert = FinBERTReasoningEncoder(
        model_name=cfg.study3.finbert_model,
        device=device,
    )

    closes = df_all["close"].to_numpy(dtype=np.float64)
    next_ret_all = np.zeros(len(df_all), dtype=np.float64)
    next_ret_all[:-1] = (closes[1:] / np.maximum(closes[:-1], 1e-8)) - 1.0
    p2_costs_all = compute_phase2_bar_base_cost_rates(
        df_all,
        fee_rate=cfg.evaluation.phase2_fee_rate,
        atr_period=cfg.evaluation.phase2_atr_period,
        kappa=cfg.evaluation.phase2_kappa,
    )

    epochs = int(args.epochs) if args.epochs is not None else int(cfg.study3.epochs)
    lambda3_grid = list(cfg.evaluation.sensitivity_lambda3_grid)
    oos_actions_by_l3: Dict[float, List[np.ndarray]] = {l3: [] for l3 in lambda3_grid}
    oos_indices_list: List[np.ndarray] = []
    prop_5_1_checks: List[Dict[str, float]] = []

    for fold in folds:
        tr_idx = fold.train_indices
        te_idx = fold.test_indices
        oos_indices_list.append(te_idx)

        pack = build_aligned_distillation_dataset(
            per_bar_lob_windows=[per_bar_train_wins[i] for i in tr_idx],
            w_final_per_bar=w_final_full[tr_idx],
            consensus_texts_per_bar=[consensus_full[i] for i in tr_idx],
            next_bar_returns=next_ret_all[tr_idx],
            one_way_cost_rates=p2_costs_all[tr_idx],
            finbert_encoder=finbert,
            sigma_w=cfg.study3.teacher_sigma_w,
            tau=cfg.study3.distill_temperature_tau,
        )

        te_lob_t = torch.as_tensor(online_wins_all[te_idx], dtype=torch.float32, device=device)

        for l3 in lambda3_grid:
            trainer = StudentDistillationTrainer(
                lambda_reasoning=l3,
                tau=cfg.study3.distill_temperature_tau,
                ppo_clip=cfg.study3.student_ppo_clip,
                c_v=cfg.study3.student_value_coef,
                c_e=cfg.study3.student_entropy_coef,
                lr=cfg.study3.learning_rate,
                batch_size=cfg.study3.batch_size,
                epochs=epochs,
                max_grad_norm=cfg.study3.max_grad_norm,
                seed=42 + fold.fold_id,
                device=device,
            )
            student, _ = trainer.fit(pack)
            acts_t, _ = student.act_online(te_lob_t)
            oos_actions_by_l3[l3].append(acts_t.cpu().numpy())

            if abs(l3 - cfg.study3.lambda_reasoning) < 1e-6 and fold.fold_id == 1:
                with torch.no_grad():
                    _, _, h_p = student(te_lob_t[:32])
                e_ref = torch.as_tensor(pack.teacher_reasoning[: len(h_p)], dtype=torch.float32, device=device)
                prop_5_1_checks.append(verify_property_5_1_orthogonality(h_p, e_ref))

    cat_te_idx = np.concatenate(oos_indices_list)
    eval_df = df_all.iloc[cat_te_idx].copy()
    bt = CausalPortfolioBacktester(
        initial_capital=cfg.evaluation.initial_capital_usdt,
        bars_per_year=cfg.evaluation.btc_bars_per_year,
    )

    table_6_11 = {}
    for l3 in lambda3_grid:
        acts = np.concatenate(oos_actions_by_l3[l3])
        res = bt.run_decisions(eval_df, ACTION_WEIGHTS[acts], phase=2)
        table_6_11[f"{l3:.1f}"] = res.summary.to_dict()

    primary_acts = np.concatenate(oos_actions_by_l3[cfg.study3.lambda_reasoning])
    primary_weights = ACTION_WEIGHTS[primary_acts]
    res_p2 = bt.run_decisions(eval_df, primary_weights, phase=2)

    crisis_proposed = {}
    for w_name, (c_start, c_end) in cfg.evaluation.crisis_windows.items():
        crisis_proposed[w_name] = compute_crisis_window_loss(
            res_p2.equity_curve, c_start, c_end, window_name=w_name
        ).to_dict()

    teacher_disc_oos = np.array(
        [map_continuous_weight_to_discrete_action(float(w)) for w in w_final_full[cat_te_idx]],
        dtype=np.int64,
    )
    agreement_pct = float(np.mean(primary_acts == teacher_disc_oos)) * 100.0

    # Phase 1 Proposed Model (5-fold CV over evaluation period with fixed 0.05% fee, Table 6.6)
    p1_splits = generate_phase1_kfold_splits(cat_te_idx, n_splits=cfg.evaluation.phase1_kfold_splits)
    p1_pred_weights = np.zeros(len(cat_te_idx), dtype=np.float64)
    idx_to_local = {int(g_i): loc_i for loc_i, g_i in enumerate(cat_te_idx)}
    p1_costs_all = compute_phase1_cost_rates(len(df_all), fee_rate=cfg.evaluation.phase1_fixed_fee_rate)

    for split_i, (p1_tr, p1_te) in enumerate(p1_splits):
        p1_pack = build_aligned_distillation_dataset(
            per_bar_lob_windows=[per_bar_train_wins[i] for i in p1_tr],
            w_final_per_bar=w_final_full[p1_tr],
            consensus_texts_per_bar=[consensus_full[i] for i in p1_tr],
            next_bar_returns=next_ret_all[p1_tr],
            one_way_cost_rates=p1_costs_all[p1_tr],
            finbert_encoder=finbert,
            sigma_w=cfg.study3.teacher_sigma_w,
            tau=cfg.study3.distill_temperature_tau,
        )
        p1_trainer = StudentDistillationTrainer(
            lambda_reasoning=cfg.study3.lambda_reasoning,
            epochs=epochs,
            seed=100 + split_i,
            device=device,
        )
        s_p1, _ = p1_trainer.fit(p1_pack)
        acts_p1, _ = s_p1.act_online(torch.as_tensor(online_wins_all[p1_te], dtype=torch.float32, device=device))
        acts_p1_np = acts_p1.cpu().numpy()
        for k_pos, g_idx in enumerate(p1_te):
            p1_pred_weights[idx_to_local[int(g_idx)]] = ACTION_WEIGHTS[acts_p1_np[k_pos]]

    res_p1 = bt.run_decisions(eval_df, p1_pred_weights, phase=1)

    np.savez_compressed(
        out_dir / "student_oos_predictions.npz",
        oos_bar_indices=cat_te_idx,
        student_actions=primary_acts,
        student_weights=primary_weights,
        phase2_net_returns=res_p2.bar_net_returns,
        phase1_net_returns=res_p1.bar_net_returns,
    )

    payload = {
        "config_sha256": cfg.config_sha256,
        "lob_input_mode": lob_mode,
        "teacher_signals_present": teacher_present,
        "llm_simulated": llm_simulated,
        "property_5_1_orthogonality": prop_5_1_checks[0] if prop_5_1_checks else {},
        "proposed_phase1_table_6_6": res_p1.summary.to_dict(),
        "proposed_phase2_table_6_7": res_p2.summary.to_dict(),
        "crisis_windows_table_6_8": crisis_proposed,
        "table_6_11_lambda3_sensitivity": table_6_11,
        "table_6_15_teacher_student_agreement_pct": agreement_pct,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    run_ctx = RunContext(
        stage="study3",
        flags={
            "llm_simulated": llm_simulated,
            "teacher_signals_present": teacher_present,
            "lob_input_mode": lob_mode,
            "finbert_offline_hash": False,
            "quick_verify": bool(args.allow_test_fallbacks),
            "reduced_training_budget": bool(epochs < cfg.study3.epochs),
        },
    )
    ledger = NumberLedger(run_context=run_ctx)
    rec_kw = dict(script_path=script_rel, config_sha256=cfg.config_sha256, run_log_path=log_path)

    for ph_num, summ in [(1, res_p1.summary), (2, res_p2.summary)]:
        tbl = "Table 6.6" if ph_num == 1 else "Table 6.7"
        ledger.record(f"T6.{5 + ph_num}_PROPOSED_CUMRET", tbl, f"Proposed Phase {ph_num} CumReturn", summ.cum_return_pct, "{:.2f}", "%", **rec_kw)
        ledger.record(f"T6.{5 + ph_num}_PROPOSED_CAGR", tbl, f"Proposed Phase {ph_num} CAGR", summ.cagr_pct, "{:.2f}", "%", **rec_kw)
        ledger.record(f"T6.{5 + ph_num}_PROPOSED_SHARPE", tbl, f"Proposed Phase {ph_num} Sharpe", summ.sharpe, "{:.2f}", "", **rec_kw)
        ledger.record(f"T6.{5 + ph_num}_PROPOSED_SORTINO", tbl, f"Proposed Phase {ph_num} Sortino", summ.sortino, "{:.2f}", "", **rec_kw)
        ledger.record(f"T6.{5 + ph_num}_PROPOSED_MDD", tbl, f"Proposed Phase {ph_num} MDD", summ.mdd_pct, "{:.2f}", "%", **rec_kw)

    ledger.record("T6.8_LUNA_PROPOSED_MDD", "Table 6.8", "Luna/Rate-Hike Proposed Crisis Drawdown", float(crisis_proposed["luna_rate_hike"]["window_mdd_pct"]), "{:.2f}", "%", **rec_kw)
    ledger.record("T6.8_FTX_PROPOSED_MDD", "Table 6.8", "FTX Collapse Proposed Crisis Drawdown", float(crisis_proposed["ftx_collapse"]["window_mdd_pct"]), "{:.2f}", "%", **rec_kw)

    for l3_key, d_l3 in table_6_11.items():
        clean_l3 = l3_key.replace(".", "_")
        ledger.record(f"T6.11_LAMBDA3_{clean_l3}_SHARPE", "Table 6.11", f"Lambda_3={l3_key} Phase 2 Sharpe", float(d_l3["sharpe"]), "{:.2f}", "", **rec_kw)

    ledger.record("T6.15_TEACHER_STUDENT_AGREE_PCT", "Table 6.15", "Teacher-Student Action Agreement Rate", agreement_pct, "{:.2f}", "%", **rec_kw)
    print("[Study 3] Completed and recorded ledger entries.")


if __name__ == "__main__":
    main()
