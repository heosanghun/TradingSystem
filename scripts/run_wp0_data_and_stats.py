"""
WP0 — Real Market Data Ingestion, Table 6.2 Statistics, Stationarity Tests, Buy & Hold,
and 10,000-Run Student Inference Latency Benchmark.

Usage (inside `.venv` sandbox):
  c:\\Project\\TradingSystem\\.venv\\Scripts\\python.exe scripts/run_wp0_data_and_stats.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ts.baselines.models import BuyAndHoldBaseline
from ts.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, SystemConfig
from ts.data.binance_fetcher import BinanceDataFetcher
from ts.data.macro_fetcher import MacroAssetFetcher
from ts.eval.backtester import CausalPortfolioBacktester
from ts.eval.latency_benchmark import benchmark_student_and_teacher_latency
from ts.eval.metrics import compute_crisis_window_loss, compute_distribution_and_stationarity_stats
from ts.ledger import NumberLedger


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WP0 empirical data & latency verification.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--latency-runs", type=int, default=None)
    args = parser.parse_args()

    cfg = SystemConfig.from_json(args.config)
    cache_dir = PROJECT_ROOT / "data" / "cache"
    out_dir = PROJECT_ROOT / "runs" / "wp0"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = "runs/wp0/metrics.json"
    script_rel = "scripts/run_wp0_data_and_stats.py"

    ledger = NumberLedger()
    binance = BinanceDataFetcher(cache_dir=cache_dir)
    macro = MacroAssetFetcher(cache_dir=cache_dir)

    # 1. Fetch full BTCUSDT 1h history (warmup + train + 27m eval)
    full_btc_df, full_gap_report = binance.fetch_ohlcv(
        symbol=cfg.data.primary_symbol,
        interval=cfg.data.bar_interval,
        start_utc=cfg.data.warmup_start,
        end_utc=cfg.data.eval_end,
        realign_grid=False,
    )
    eval_btc_df, eval_gap_report = binance.fetch_ohlcv(
        symbol=cfg.data.primary_symbol,
        interval=cfg.data.bar_interval,
        start_utc=cfg.data.eval_start,
        end_utc=cfg.data.eval_end,
        realign_grid=False,
    )

    btc_stats = compute_distribution_and_stationarity_stats(eval_btc_df["close"], symbol="BTCUSDT")

    # 2. Fetch US control assets SPY & GLD for Table 6.2 comparison
    control_stats = {}
    for sym in cfg.data.control_symbols:
        ctrl_df = macro.fetch_us_control_asset(
            ticker=sym,
            start_utc=cfg.data.eval_start,
            end_utc=cfg.data.eval_end,
            prefer_hourly=True,
        )
        st = compute_distribution_and_stationarity_stats(ctrl_df["close"], symbol=sym)
        st_dict = st.to_dict()
        st_dict["bar_resolution"] = ctrl_df.attrs.get("resolution", "unknown")
        control_stats[sym] = st_dict

    # 3. Buy & Hold Benchmark on exact evaluation period (Phase 1, Phase 2, & Table 6.8 Crisis Windows)
    bh = BuyAndHoldBaseline()
    bh_w = bh.predict_weights(len(eval_btc_df))
    bt_engine = CausalPortfolioBacktester(
        initial_capital=cfg.evaluation.initial_capital_usdt,
        bars_per_year=cfg.evaluation.btc_bars_per_year,
        risk_free_annual=cfg.evaluation.risk_free_rate_annual,
        phase1_fee_rate=cfg.evaluation.phase1_fixed_fee_rate,
        phase2_fee_rate=cfg.evaluation.phase2_fee_rate,
        phase2_atr_period=cfg.evaluation.phase2_atr_period,
        phase2_kappa=cfg.evaluation.phase2_kappa,
        phase2_eta=cfg.evaluation.phase2_eta,
    )
    bh_p1 = bt_engine.run(eval_btc_df, bh_w, phase=1)
    bh_p2 = bt_engine.run(eval_btc_df, bh_w, phase=2)

    crisis_bh = {}
    for w_name, (c_start, c_end) in cfg.evaluation.crisis_windows.items():
        crisis_bh[w_name] = compute_crisis_window_loss(
            bh_p2.equity_curve, c_start, c_end, window_name=w_name
        ).to_dict()

    # 4. Table 6.14 Student 1D-CNN 10,000-run GPU/CPU latency benchmark
    n_lat = args.latency_runs or cfg.evaluation.latency_benchmark_runs
    latency_report = benchmark_student_and_teacher_latency(
        student_model=None,
        teacher_step_fn=None,
        n_student_runs=n_lat,
    )

    payload = {
        "config_sha256": cfg.config_sha256,
        "full_range_gap_report": full_gap_report,
        "eval_range_gap_report": eval_gap_report,
        "table_6_2": {
            "BTCUSDT": btc_stats.to_dict(),
            **control_stats,
        },
        "buy_and_hold": {
            "phase1": bh_p1.summary.to_dict(),
            "phase2": bh_p2.summary.to_dict(),
            "crisis_windows": crisis_bh,
        },
        "table_6_14_student_latency": latency_report.student_online.to_dict(),
    }

    (out_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Record verified figures into `ledger/numbers.csv`
    rec_kw = dict(script_path=script_rel, config_sha256=cfg.config_sha256, run_log_path=log_path)
    ledger.record("T6.1_BTC_OBSERVED_BARS", "Table 6.1", "BTC observed 1h bars (2021.10-2023.12)", eval_gap_report["observed_bars"], "{:.0f}", "bars", **rec_kw)
    ledger.record("T6.1_BTC_MISSING_BARS", "Table 6.1", "BTC missing maintenance 1h bars", eval_gap_report["missing_bars"], "{:.0f}", "bars", **rec_kw)

    for sym_key, st_d in [("BTC", btc_stats.to_dict()), ("SPY", control_stats["SPY"]), ("GLD", control_stats["GLD"])]:
        ledger.record(f"T6.2_{sym_key}_N", "Table 6.2", f"{sym_key} log-return sample count", float(st_d["n_bars"]), "{:.0f}", "bars", **rec_kw)
        ledger.record(f"T6.2_{sym_key}_MEAN", "Table 6.2", f"{sym_key} log-return mean", float(st_d["mean"]), "{:.6f}", "", **rec_kw)
        ledger.record(f"T6.2_{sym_key}_STD", "Table 6.2", f"{sym_key} log-return std", float(st_d["std"]), "{:.4f}", "", **rec_kw)
        ledger.record(f"T6.2_{sym_key}_MIN", "Table 6.2", f"{sym_key} log-return min", float(st_d["min_val"]), "{:.4f}", "", **rec_kw)
        ledger.record(f"T6.2_{sym_key}_MAX", "Table 6.2", f"{sym_key} log-return max", float(st_d["max_val"]), "{:.4f}", "", **rec_kw)
        ledger.record(f"T6.2_{sym_key}_KURT", "Table 6.2", f"{sym_key} Pearson kurtosis", float(st_d["pearson_kurtosis"]), "{:.2f}", "", **rec_kw)

    ledger.record("CH6_ADF_BTC_PRICE_STAT", "Ch.VI Sec 2.2", "BTC price ADF statistic", btc_stats.adf_price_stat, "{:.2f}", "", **rec_kw)
    ledger.record("CH6_ADF_BTC_PRICE_P", "Ch.VI Sec 2.2", "BTC price ADF p-value", btc_stats.adf_price_pvalue, "{:.4f}", "", **rec_kw)
    ledger.record("CH6_ADF_BTC_RET_P", "Ch.VI Sec 2.2", "BTC log-return ADF p-value", btc_stats.adf_ret_pvalue, "{:.4e}", "", **rec_kw)
    ledger.record("CH6_LB_BTC_R2_P", "Ch.VI Sec 2.2", "BTC squared-return Ljung-Box p-value", btc_stats.ljung_box_r2_pvalue, "{:.4e}", "", **rec_kw)

    for ph_num, summ in [(1, bh_p1.summary), (2, bh_p2.summary)]:
        tbl = "Table 6.6" if ph_num == 1 else "Table 6.7"
        ledger.record(f"T6.{5 + ph_num}_BH_CUMRET", tbl, f"Buy & Hold Phase {ph_num} CumReturn", summ.cum_return_pct, "{:.2f}", "%", **rec_kw)
        ledger.record(f"T6.{5 + ph_num}_BH_CAGR", tbl, f"Buy & Hold Phase {ph_num} CAGR", summ.cagr_pct, "{:.2f}", "%", **rec_kw)
        ledger.record(f"T6.{5 + ph_num}_BH_SHARPE", tbl, f"Buy & Hold Phase {ph_num} Sharpe", summ.sharpe, "{:.2f}", "", **rec_kw)
        ledger.record(f"T6.{5 + ph_num}_BH_SORTINO", tbl, f"Buy & Hold Phase {ph_num} Sortino", summ.sortino, "{:.2f}", "", **rec_kw)
        ledger.record(f"T6.{5 + ph_num}_BH_MDD", tbl, f"Buy & Hold Phase {ph_num} MDD", summ.mdd_pct, "{:.2f}", "%", **rec_kw)

    ledger.record("T6.8_LUNA_BH_MDD", "Table 6.8", "Luna/Rate-Hike B&H Crisis Drawdown", float(crisis_bh["luna_rate_hike"]["window_mdd_pct"]), "{:.2f}", "%", **rec_kw)
    ledger.record("T6.8_FTX_BH_MDD", "Table 6.8", "FTX Collapse B&H Crisis Drawdown", float(crisis_bh["ftx_collapse"]["window_mdd_pct"]), "{:.2f}", "%", **rec_kw)

    ledger.record("T6.14_STUDENT_MEAN_MS", "Table 6.14", "Student 1D-CNN online latency mean", latency_report.student_online.mean_ms, "{:.3f}", "ms", **rec_kw)
    ledger.record("T6.14_STUDENT_MEDIAN_MS", "Table 6.14", "Student 1D-CNN online latency median", latency_report.student_online.median_ms, "{:.3f}", "ms", **rec_kw)
    ledger.record("T6.14_STUDENT_P99_MS", "Table 6.14", "Student 1D-CNN online latency p99", latency_report.student_online.p99_ms, "{:.3f}", "ms", **rec_kw)

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
