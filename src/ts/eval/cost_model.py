"""
Transaction Cost & Dynamic Slippage Model (Chapter VI Sec 3.3, Eqs. VI.6–VI.8).

Implements:
- Phase 1: Fixed one-way fee rate `c_fee = 0.0005` (`0.05%`)
- Phase 2: ATR(14) volatility slippage + Almgren-Chriss temporary market impact:
    `TR_t = max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)`
    `ATR_t = ((n - 1) * ATR_{t-1} + TR_t) / n`   (`n = 14`, Wilder smoothing)
    `ν_t = ATR_t / P_t`
    `s_t = c_fee + κ * ν_t + η * (Q_t / V_t)`
  with `c_fee = 0.0005`, `κ = 0.1`, `η = 0.05`, where `Q_t = |Δw_t| * Equity_{t-1} / P_t`
  is the traded BTC base volume and `V_t` is the bar's observed BTC base volume.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_wilder_atr(
    high: np.ndarray | pd.Series,
    low: np.ndarray | pd.Series,
    close: np.ndarray | pd.Series,
    period: int = 14,
) -> np.ndarray:
    """Exact Wilder ATR(`period`) matching Eq. (VI.6)."""
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    n = len(c)
    if n == 0:
        return np.array([], dtype=np.float64)

    prev_c = np.empty_like(c)
    prev_c[0] = c[0]
    prev_c[1:] = c[:-1]

    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = np.empty(n, dtype=np.float64)
    atr[0] = tr[0]
    alpha = 1.0 / float(period)
    for t in range(1, n):
        atr[t] = (1.0 - alpha) * atr[t - 1] + alpha * tr[t]
    return atr


def compute_phase1_cost_rates(num_bars: int, fee_rate: float = 0.0005) -> np.ndarray:
    """Fixed one-way transaction cost array (`0.05%`) for Phase 1."""
    return np.full(int(num_bars), float(fee_rate), dtype=np.float64)


def compute_phase2_bar_base_cost_rates(
    ohlcv_df: pd.DataFrame,
    fee_rate: float = 0.0005,
    atr_period: int = 14,
    kappa: float = 0.1,
) -> np.ndarray:
    """
    Precompute equity-independent bar cost floor `c_fee + κ * (ATR_14,t / P_t)`
    used inside RL training rollouts; full equity-dependent Almgren-Chriss impact
    `η * (Q_t / V_t)` is additionally applied during Phase 2 portfolio backtesting.
    """
    c = ohlcv_df["close"].to_numpy(dtype=np.float64)
    atr = compute_wilder_atr(
        ohlcv_df["high"].to_numpy(dtype=np.float64),
        ohlcv_df["low"].to_numpy(dtype=np.float64),
        c,
        period=atr_period,
    )
    nu_t = atr / np.maximum(c, 1e-8)
    return float(fee_rate) + float(kappa) * nu_t


def compute_phase2_step_cost_rate(
    price_t: float,
    atr14_t: float,
    bar_base_vol_t: float,
    abs_delta_w: float,
    prev_equity: float,
    fee_rate: float = 0.0005,
    kappa: float = 0.1,
    eta: float = 0.05,
    max_impact_cap: float = 0.02,
) -> float:
    """
    Exact Eq. (VI.8) one-way dynamic transaction cost & slippage rate `s_t` at bar `t`:
      `s_t = c_fee + κ * (ATR_14,t / P_t) + η * (Q_t / V_t)`
    where `Q_t = |Δw_t| * prev_equity / P_t` is traded BTC base volume.
    """
    if abs_delta_w <= 1e-12:
        return 0.0
    p = max(float(price_t), 1e-8)
    nu_t = max(float(atr14_t), 0.0) / p
    q_btc = (float(abs_delta_w) * max(float(prev_equity), 0.0)) / p

    if bar_base_vol_t > 1e-8:
        impact = min(float(eta) * (q_btc / float(bar_base_vol_t)), float(max_impact_cap))
    else:
        # Maintenance gap bar with zero traded exchange volume
        impact = float(max_impact_cap)

    return float(fee_rate) + float(kappa) * nu_t + impact
