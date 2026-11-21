"""
Choose the task term of the student's distillation loss and its execution rule on data
inside the first training fold (D-20).

Background. The student's task term is a PPO surrogate on sampled actions. Its reward charges
turnover between unrelated samples, so it does not teach the student to keep a position. The
alternative is the turnover-regularised Sharpe loss of Lim, Zohren & Roberts (2019): the
net-of-cost Sharpe ratio of the expected position over contiguous bars. With it, the position
can be executed directly or through a no-trade band around the held weight (Martin 2012;
de Lataillade et al. 2012).

Protocol (written to the decision log, now docs/DECISIONS.md, before this script was run):
- Only the first training fold (2020-10-01 .. 2021-09-30) is used.
- Three validation splits inside it, each with students trained on the months before it:
    split C: train .. 2021-03-31, validate 2021-04-01 .. 2021-05-31
    split A: train .. 2021-05-31, validate 2021-06-01 .. 2021-07-31
    split B: train .. 2021-07-31, validate 2021-08-01 .. 2021-09-30
- Candidates:
    T0  PPO task loss, confidence gate 0.8                      (current setting)
    T1  net-of-cost Sharpe task loss, expected weight executed
    T2  net-of-cost Sharpe task loss, no-trade band 0.25 on the five action levels
- Selection: T1 and T2 are eligible only if they trade in at least one validation window.
  If no eligible candidate has a mean Phase 2 Sharpe more than 0.05 above T0, T0 is kept.
  Otherwise the highest mean wins; within 0.05, the lower mean turnover.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from scripts.run_study3 import _build_bar_lob_windows, _load_and_validate_teacher_signals
from ts.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, SystemConfig
from ts.data.binance_fetcher import BinanceDataFetcher
from ts.data.lob_processor import LOBWindowProcessor
from ts.eval.backtester import CausalPortfolioBacktester
from ts.eval.cost_model import compute_phase2_bar_base_cost_rates
from ts.eval.walk_forward import generate_phase2_walk_forward_folds
from ts.ledger import read_git_state
from ts.study1.ppo_expert import ACTION_WEIGHTS
from ts.study3.distillation import StudentDistillationTrainer
from ts.study3.student_model import execute_student_weights
from ts.study3.teacher_signal import FinBERTReasoningEncoder, build_aligned_distillation_dataset

SPLITS = (
    ("C", "2021-04-01T00:00:00Z", "2021-06-01T00:00:00Z"),
    ("A", "2021-06-01T00:00:00Z", "2021-08-01T00:00:00Z"),
    ("B", "2021-08-01T00:00:00Z", "2021-10-01T00:00:00Z"),
)
BAND = 0.25
CURRENT_THETA = 0.8
# name -> (task objective, execution mode)
CANDIDATES = {
    "T0": ("ppo", "gate"),
    "T1": ("net_sharpe", "expected"),
    "T2": ("net_sharpe", "expected_band"),
}
MARGIN = 0.05


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the student's task objective and execution rule inside training fold 1.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--study2-dir", type=str, default=str(PROJECT_ROOT / "runs" / "study2"))
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "runs" / "study3" / "fold1_val_task_objective.json"))
    parser.add_argument("--preregistered-commit", type=str, default="")
    args = parser.parse_args()

    cfg = SystemConfig.from_json(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    binance = BinanceDataFetcher(cache_dir=PROJECT_ROOT / "data" / "cache")
    df_all, _ = binance.fetch_ohlcv(symbol=cfg.data.primary_symbol, interval=cfg.data.bar_interval,
                                    start_utc=cfg.data.warmup_start, end_utc=cfg.data.eval_end, realign_grid=False)
    folds = generate_phase2_walk_forward_folds(
        pd.DatetimeIndex(df_all.index), first_train_start=cfg.data.first_train_start,
        train_months=cfg.evaluation.walk_forward_train_months, test_months=cfg.evaluation.walk_forward_test_months,
        step_months=cfg.evaluation.walk_forward_step_months, num_folds=cfg.evaluation.walk_forward_num_folds)
    tr_idx = folds[0].train_indices
    ts_tr = df_all.index[tr_idx]
    assert int(tr_idx[-1]) < int(np.min(folds[0].test_indices))

    w_final_full, consensus_full, _present, _sim = _load_and_validate_teacher_signals(
        study2_dir=Path(args.study2_dir), total_bars=len(df_all), required_bar_indices=np.unique(tr_idx),
        expected_config_sha256=cfg.config_sha256, allow_test_fallbacks=False)
    df_1m, _ = binance.fetch_ohlcv(symbol=cfg.data.primary_symbol, interval="1m",
                                   start_utc=cfg.data.warmup_start, end_utc=cfg.data.eval_end, realign_grid=False)
    lob_proc = LOBWindowProcessor(seq_len=cfg.study3.lob_seq_len, num_features=cfg.study3.lob_input_dim,
                                  windows_per_bar=cfg.data.lob_windows_per_bar)
    per_bar_train_wins, online_wins_all, lob_mode = _build_bar_lob_windows(df_all, df_1m, lob_proc, allow_hourly_test_fallback=False)
    finbert = FinBERTReasoningEncoder(model_name=cfg.study3.finbert_model, device=device)

    closes = df_all["close"].to_numpy(dtype=np.float64)
    next_ret = np.zeros(len(df_all), dtype=np.float64)
    next_ret[:-1] = closes[1:] / np.maximum(closes[:-1], 1e-8) - 1.0
    costs = compute_phase2_bar_base_cost_rates(df_all, fee_rate=cfg.evaluation.phase2_fee_rate,
                                               atr_period=cfg.evaluation.phase2_atr_period, kappa=cfg.evaluation.phase2_kappa)
    bpy = cfg.evaluation.btc_bars_per_year
    bt = CausalPortfolioBacktester(initial_capital=cfg.evaluation.initial_capital_usdt, bars_per_year=bpy)

    rows: List[Dict[str, object]] = []
    for split_name, v_start, v_end in SPLITS:
        v0, v1 = pd.Timestamp(v_start), pd.Timestamp(v_end)
        sub_idx = tr_idx[ts_tr < v0]
        val_idx = tr_idx[(ts_tr >= v0) & (ts_tr < v1)]
        assert len(sub_idx) and len(val_idx) and np.max(sub_idx) < np.min(val_idx)
        pack = build_aligned_distillation_dataset(
            per_bar_lob_windows=[per_bar_train_wins[i] for i in sub_idx], w_final_per_bar=w_final_full[sub_idx],
            consensus_texts_per_bar=[consensus_full[i] for i in sub_idx], next_bar_returns=next_ret[sub_idx],
            one_way_cost_rates=costs[sub_idx], finbert_encoder=finbert,
            sigma_w=cfg.study3.teacher_sigma_w, tau=cfg.study3.distill_temperature_tau)
        df_val = df_all.iloc[val_idx].copy()
        val_x = torch.as_tensor(online_wins_all[val_idx], dtype=torch.float32, device=device)
        probs_by_objective: Dict[str, np.ndarray] = {}
        for objective in ("ppo", "net_sharpe"):
            trainer = StudentDistillationTrainer(
                lambda_reasoning=cfg.study3.lambda_reasoning, tau=cfg.study3.distill_temperature_tau,
                ppo_clip=cfg.study3.student_ppo_clip, c_v=cfg.study3.student_value_coef, c_e=cfg.study3.student_entropy_coef,
                lr=cfg.study3.learning_rate, batch_size=cfg.study3.batch_size, epochs=int(cfg.study3.epochs),
                max_grad_norm=cfg.study3.max_grad_norm, seed=43, device=device,
                task_objective=objective, bars_per_year=bpy)
            student, history = trainer.fit(pack)
            student.eval()
            with torch.no_grad():
                _, probs_t = student.act_online(val_x)
            probs_by_objective[objective] = probs_t.cpu().numpy()
            print("[D-20 Train]", json.dumps({"split": split_name, "objective": objective,
                                              "first_epoch": history[0], "last_epoch": history[-1]}), flush=True)
        for name, (objective, mode) in CANDIDATES.items():
            probs = probs_by_objective[objective]
            w = execute_student_weights(probs, ACTION_WEIGHTS, mode, CURRENT_THETA, 1, BAND)
            res = bt.run_decisions(df_val, w, phase=2)
            gross = res.bar_gross_returns
            row = {
                "split": split_name, "candidate": name, "objective": objective, "execution": mode,
                "sub_train_bars": int(len(sub_idx)), "val_bars": int(len(val_idx)),
                "val_mean_turnover": round(float(np.mean(np.abs(np.diff(w, prepend=0.0)))), 5),
                "val_position_changes": int(np.sum(np.abs(np.diff(w, prepend=0.0)) > 1e-9)),
                "val_mean_abs_weight": round(float(np.mean(np.abs(w))), 4),
                "val_mean_weight": round(float(np.mean(w)), 4),
                "val_sharpe": round(float(res.summary.sharpe), 3),
                "val_pre_cost_sharpe": round(float(np.mean(gross) / max(np.std(gross, ddof=1), 1e-12) * np.sqrt(bpy)), 3),
                "val_cum_return_pct": round(float(res.summary.cum_return_pct), 2),
                "val_mdd_pct": round(float(res.summary.mdd_pct), 2),
                "mean_top_probability": round(float(np.mean(probs.max(axis=1))), 4),
            }
            print("[D-20 Row]", json.dumps(row), flush=True)
            rows.append(row)

    table = {}
    for name in CANDIDATES:
        rs = [r for r in rows if r["candidate"] == name]
        table[name] = {
            "candidate": name, "objective": CANDIDATES[name][0], "execution": CANDIDATES[name][1],
            "mean_val_sharpe": round(float(np.mean([r["val_sharpe"] for r in rs])), 3),
            "mean_val_turnover": round(float(np.mean([r["val_mean_turnover"] for r in rs])), 5),
            "trades_in_any_split": bool(any(r["val_mean_turnover"] > 0 for r in rs)),
            "val_sharpe_by_split": {r["split"]: r["val_sharpe"] for r in rs},
        }
    base = table["T0"]
    better = [t for k, t in table.items() if k != "T0" and t["trades_in_any_split"]
              and t["mean_val_sharpe"] > base["mean_val_sharpe"] + MARGIN]
    if better:
        top = max(t["mean_val_sharpe"] for t in better)
        tied = [t for t in better if top - t["mean_val_sharpe"] <= MARGIN]
        tied.sort(key=lambda t: (t["mean_val_turnover"], t["candidate"]))
        selected = tied[0]
    else:
        selected = base
    out = {
        "script_path": "scripts/calibrate_study3_task_objective.py", "preregistered_in_commit": args.preregistered_commit,
        "executed_on_git_state": read_git_state().label, "test_fold_accessed": False, "lob_input_mode": lob_mode,
        "selection_rule": "T1/T2 eligible only if they trade in a validation split; keep T0 unless an eligible candidate "
                          "exceeds T0's mean Phase 2 Sharpe by more than 0.05; then highest mean, ties within 0.05 -> lower mean turnover",
        "fixed_values": {"net_sharpe_seq_len": 64, "band": BAND, "current_theta": CURRENT_THETA, "sharpe_annualised": True},
        "splits": [list(x) for x in SPLITS], "summary_by_candidate": list(table.values()),
        "selected_candidate": selected["candidate"], "selected_objective": selected["objective"],
        "selected_execution": selected["execution"], "rows": rows,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("[D-20 Selected]", json.dumps(selected), flush=True)


if __name__ == "__main__":
    main()
