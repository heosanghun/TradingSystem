"""
Causal 1-Hour Portfolio Backtester (Chapter VI Sec 3.2–3.3).

Executes target portfolio weights `w_t ∈ [-1.0, +1.0]` computed at bar close `t`
over the subsequent bar's asset return `r_{t+1} = P_{t+1} / P_t - 1` (or aligned
out-of-sample return series), deducting exact Phase 1 (`0.05%`) or Phase 2
(`ATR + Almgren-Chriss`, Eq. VI.8) transaction costs on turnover `|w_t - w_{t-1}|`.

Strictly zero post-hoc scaling, inversion, blending, or return capping!
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ts.eval.cost_model import compute_phase2_step_cost_rate, compute_wilder_atr
from ts.eval.metrics import PerformanceSummary, compute_performance_summary


@dataclass
class BacktestResult:
    timestamps: pd.DatetimeIndex
    equity_curve: pd.Series
    bar_net_returns: np.ndarray
    bar_gross_returns: np.ndarray
    bar_costs: np.ndarray
    weights: np.ndarray
    summary: PerformanceSummary


class CausalPortfolioBacktester:
    """Simulates a 1-hour rebalanced portfolio under Phase 1 or Phase 2 cost regimes."""

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        bars_per_year: int = 8760,
        risk_free_annual: float = 0.0,
        phase1_fee_rate: float = 0.0005,
        phase2_fee_rate: float = 0.0005,
        phase2_atr_period: int = 14,
        phase2_kappa: float = 0.1,
        phase2_eta: float = 0.05,
    ) -> None:
        self.initial_capital = float(initial_capital)
        self.bars_per_year = int(bars_per_year)
        self.risk_free_annual = float(risk_free_annual)
        self.phase1_fee_rate = float(phase1_fee_rate)
        self.phase2_fee_rate = float(phase2_fee_rate)
        self.phase2_atr_period = int(phase2_atr_period)
        self.phase2_kappa = float(phase2_kappa)
        self.phase2_eta = float(phase2_eta)

    def run(
        self,
        ohlcv_df: pd.DataFrame,
        target_weights: np.ndarray,
        phase: int = 2,
        precomputed_atr: Optional[np.ndarray] = None,
        initial_prev_weight: float = 0.0,
    ) -> BacktestResult:
        """
        Simulate portfolio evolution across `ohlcv_df`.

        Convention:
        - `ohlcv_df` has `N` bars in the evaluation window (`t = 0 .. N-1`).
        - Bar return realized during bar `t` is `r_t = close[t] / close[t-1] - 1`
          (for `t = 0`, `close[0] / open[0] - 1`), and `target_weights[t]` is the position
          held across bar `t` (decided strictly from information available prior to bar `t`'s return!).
        """
        n = len(ohlcv_df)
        if len(target_weights) != n:
            raise ValueError(f"target_weights length ({len(target_weights)}) != ohlcv_df length ({n})")

        closes = ohlcv_df["close"].to_numpy(dtype=np.float64)
        opens = ohlcv_df["open"].to_numpy(dtype=np.float64)
        highs = ohlcv_df["high"].to_numpy(dtype=np.float64)
        lows = ohlcv_df["low"].to_numpy(dtype=np.float64)
        vols = (
            ohlcv_df["volume"].to_numpy(dtype=np.float64)
            if "volume" in ohlcv_df.columns
            else np.full(n, 1e6, dtype=np.float64)
        )

        asset_ret = np.empty(n, dtype=np.float64)
        if n > 0:
            asset_ret[0] = (closes[0] / max(opens[0], 1e-8)) - 1.0
        if n > 1:
            asset_ret[1:] = (closes[1:] / np.maximum(closes[:-1], 1e-8)) - 1.0

        atr = (
            np.asarray(precomputed_atr, dtype=np.float64)
            if precomputed_atr is not None
            else compute_wilder_atr(highs, lows, closes, period=self.phase2_atr_period)
        )

        w = np.clip(np.asarray(target_weights, dtype=np.float64), -1.0, 1.0)
        equity = np.empty(n, dtype=np.float64)
        net_ret = np.empty(n, dtype=np.float64)
        gross_ret = np.empty(n, dtype=np.float64)
        bar_costs = np.empty(n, dtype=np.float64)

        cur_eq = self.initial_capital
        prev_w = float(initial_prev_weight)

        for t in range(n):
            wt = float(w[t])
            dw = abs(wt - prev_w)
            if phase == 1:
                s_t = self.phase1_fee_rate if dw > 1e-12 else 0.0
            else:
                # Entry/rebalance cost uses price available at start of bar `t` (`opens[t]`) and prior ATR
                atr_ref = float(atr[t - 1]) if t > 0 else float(atr[0])
                vol_ref = float(vols[t - 1]) if t > 0 else float(vols[0])
                s_t = compute_phase2_step_cost_rate(
                    price_t=float(opens[t]),
                    atr14_t=atr_ref,
                    bar_base_vol_t=vol_ref,
                    abs_delta_w=dw,
                    prev_equity=cur_eq,
                    fee_rate=self.phase2_fee_rate,
                    kappa=self.phase2_kappa,
                    eta=self.phase2_eta,
                )

            g_r = wt * float(asset_ret[t])
            c_r = s_t * dw
            n_r = max(g_r - c_r, -0.9999)  # limited liability protection

            cur_eq = cur_eq * (1.0 + n_r)
            equity[t] = cur_eq
            gross_ret[t] = g_r
            bar_costs[t] = c_r
            net_ret[t] = n_r
            prev_w = wt

        eq_with_v0 = np.concatenate([[self.initial_capital], equity])
        summary = compute_performance_summary(
            equity_curve=eq_with_v0,
            bar_net_returns=net_ret,
            weights=w,
            bars_per_year=self.bars_per_year,
            risk_free_annual=self.risk_free_annual,
        )

        eq_series = pd.Series(equity, index=ohlcv_df.index, name="equity")
        return BacktestResult(
            timestamps=pd.DatetimeIndex(ohlcv_df.index),
            equity_curve=eq_series,
            bar_net_returns=net_ret,
            bar_gross_returns=gross_ret,
            bar_costs=bar_costs,
            weights=w,
            summary=summary,
        )
