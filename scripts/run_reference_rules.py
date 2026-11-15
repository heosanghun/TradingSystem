"""
Reference rules under the thesis evaluation protocol (no learning involved).

Purpose: show what fixed, textbook rules earn on the evaluation period when they are
scored exactly like the models: decisions at the close of bar t are held over bar t+1
(`CausalPortfolioBacktester.run_decisions`) and charged the Phase 2 cost of Eq. (VI.8).
The table answers one question: at this cost level, how slow does a strategy have to be
before a gross signal survives?

The rules use only past prices:
  - long when the close is above its N-bar simple moving average, otherwise cash
  - sign of the N-bar price change (long or short)
for N in 24, 72, 168, 720 bars. Moving averages and price changes are computed on the full
series including the warm-up before the evaluation period, so the first evaluation bars
already have a full window.

These rules are not tuned and are not candidates for the proposed model. They were first
computed as a diagnosis after the model results were known; they are reported as context.

Output: runs/reference_rules/metrics.json and ledger IDs CH6_REF_*.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from ts.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, SystemConfig
from ts.data.binance_fetcher import BinanceDataFetcher
from ts.eval.backtester import CausalPortfolioBacktester
from ts.eval.walk_forward import generate_phase2_walk_forward_folds
from ts.ledger import NumberLedger, RunContext

WINDOWS = (24, 72, 168, 720)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score fixed reference rules under the Phase 2 protocol.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()

    import pandas as pd

    cfg = SystemConfig.from_json(args.config)
    df_all, _ = BinanceDataFetcher(cache_dir=PROJECT_ROOT / "data" / "cache").fetch_ohlcv(
        symbol=cfg.data.primary_symbol, interval=cfg.data.bar_interval,
        start_utc=cfg.data.warmup_start, end_utc=cfg.data.eval_end, realign_grid=False)
    folds = generate_phase2_walk_forward_folds(
        pd.DatetimeIndex(df_all.index), first_train_start=cfg.data.first_train_start,
        train_months=cfg.evaluation.walk_forward_train_months, test_months=cfg.evaluation.walk_forward_test_months,
        step_months=cfg.evaluation.walk_forward_step_months, num_folds=cfg.evaluation.walk_forward_num_folds)
    idx = np.concatenate([f.test_indices for f in folds])
    eval_df = df_all.iloc[idx].copy()
    close_all = df_all["close"]
    close_eval = close_all.iloc[idx].to_numpy(dtype=np.float64)

    bt = CausalPortfolioBacktester(initial_capital=cfg.evaluation.initial_capital_usdt,
                                   bars_per_year=cfg.evaluation.btc_bars_per_year)
    bars_per_year = cfg.evaluation.btc_bars_per_year

    rules = {"ALWAYS_LONG": np.ones(len(idx), dtype=np.float64)}
    for n in WINDOWS:
        sma = close_all.rolling(n).mean().iloc[idx].to_numpy(dtype=np.float64)
        change = (close_all / close_all.shift(n) - 1.0).iloc[idx].to_numpy(dtype=np.float64)
        rules[f"SMA{n}_LONG_OR_CASH"] = (close_eval > sma).astype(np.float64)
        rules[f"MOM{n}_LONG_SHORT"] = np.sign(change)

    results = {}
    for name, weights in rules.items():
        res = bt.run_decisions(eval_df, weights, phase=2)
        gross = res.bar_gross_returns
        turnover_total = float(np.abs(np.diff(res.weights, prepend=0.0)).sum())
        results[name] = {
            "sharpe_after_costs": float(res.summary.sharpe),
            "sharpe_before_costs": float(np.mean(gross) / max(np.std(gross, ddof=1), 1e-12) * np.sqrt(bars_per_year)),
            "cum_return_pct": float(res.summary.cum_return_pct),
            "mdd_pct": float(res.summary.mdd_pct),
            "turnover_per_bar": float(np.mean(np.abs(np.diff(weights)))),
            "mean_cost_per_unit_turnover_pct": float(100.0 * res.bar_costs.sum() / turnover_total) if turnover_total > 0 else 0.0,
        }

    out_dir = PROJECT_ROOT / "runs" / "reference_rules"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"config_sha256": cfg.config_sha256, "n_bars": int(len(idx)), "windows": list(WINDOWS), "rules": results}
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    ledger = NumberLedger(run_context=RunContext(stage="wp0"))
    rec_kw = dict(script_path="scripts/run_reference_rules.py", config_sha256=cfg.config_sha256,
                  run_log_path="runs/reference_rules/metrics.json")
    for name, r in results.items():
        ledger.record(f"CH6_REF_{name}_SHARPE", "Ch.VI reference rules", f"{name} Phase 2 Sharpe after costs", r["sharpe_after_costs"], "{:.2f}", "", **rec_kw)
        ledger.record(f"CH6_REF_{name}_GROSS_SHARPE", "Ch.VI reference rules", f"{name} Sharpe before costs", r["sharpe_before_costs"], "{:.2f}", "", **rec_kw)
        ledger.record(f"CH6_REF_{name}_TURNOVER", "Ch.VI reference rules", f"{name} turnover per bar", r["turnover_per_bar"], "{:.4f}", "", **rec_kw)
        ledger.record(f"CH6_REF_{name}_CUMRET", "Ch.VI reference rules", f"{name} Phase 2 cumulative return", r["cum_return_pct"], "{:.2f}", "%", **rec_kw)
    for name, r in results.items():
        print(f"{name:22s} after {r['sharpe_after_costs']:6.2f}  before {r['sharpe_before_costs']:6.2f}  "
              f"cum {r['cum_return_pct']:7.2f}%  turnover {r['turnover_per_bar']:.4f}")


if __name__ == "__main__":
    main()
