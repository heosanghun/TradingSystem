"""Independent pandas check of close-decision, next-bar portfolio accounting.

This diagnostic does not train models or write the thesis number ledger.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def reference_phase1(ohlcv: pd.DataFrame, decisions: np.ndarray,
                     capital: float = 1_000_000.0, fee: float = 0.0005) -> pd.DataFrame:
    """Hold the previous close decision; start flat and charge absolute turnover."""
    if len(ohlcv) != len(decisions) or ohlcv.empty:
        raise ValueError("Nonempty bars and one decision per bar are required")
    if not np.isfinite(decisions).all():
        raise ValueError("Decisions must be finite")
    close = ohlcv["close"].astype(float)
    returns = close.div(close.shift()).sub(1)
    returns.iloc[0] = close.iloc[0] / float(ohlcv["open"].iloc[0]) - 1
    held = pd.Series(decisions, index=ohlcv.index).clip(-1, 1).shift(fill_value=0)
    turnover = held.sub(held.shift(fill_value=0)).abs()
    net = held.mul(returns).sub(turnover.mul(fee))
    if (net <= -1).any():
        raise ValueError("Audit is limited to solvent trajectories")
    return pd.DataFrame({"held_weight": held, "net_return": net,
                         "equity": capital * net.add(1).cumprod()})


def run_audit(ohlcv: pd.DataFrame) -> dict:
    """Compare three random policies and one last-hour momentum policy."""
    # Import only after the independently specified reference implementation.
    from ts.eval.backtester import CausalPortfolioBacktester
    rng = np.random.default_rng(261112)
    policies = {f"random_{i}": rng.choice([-1., -.5, 0., .5, 1.], len(ohlcv))
                for i in range(3)}
    policies["previous_hour_up"] = np.where(ohlcv.close.pct_change().fillna(0) > 0, 1., 0.)
    report = {}
    for name, decisions in policies.items():
        expected = reference_phase1(ohlcv, decisions)
        observed = CausalPortfolioBacktester().run_decisions(ohlcv, decisions, phase=1)
        net_error = float(np.max(np.abs(expected.net_return.to_numpy() - observed.bar_net_returns)))
        equity_error = float(np.max(np.abs(expected.equity.to_numpy() - observed.equity_curve.to_numpy())))
        cumulative_error = abs(float(expected.equity.iloc[-1] / 1_000_000 - 1)
                               - float(observed.summary.cum_return_pct / 100))
        passed = net_error <= 1e-9 and cumulative_error <= 1e-9
        report[name] = dict(max_net_return_error=net_error, max_equity_error=equity_error,
                            cumulative_return_error=cumulative_error, passed=passed)
        if not passed:
            raise AssertionError(f"Independent audit mismatch: {name}: {report[name]}")
    return {"phase": 1, "bars": len(ohlcv), "cases": report,
            "is_diagnostic": True, "ledger_written": False}
