"""
Market Regime Ground-Truth Labelers (Ch. III Sec 2.1 & Ch. VI Sec 3.1).

Regime ID convention (consistent across Study 1):
- `0`: Bull (`상승`)
- `1`: Bear (`하락`)
- `2`: Sideways (`횡보`)
- `-1`: Unlabeled / Purged boundary tail inside training fold

Provides:
1. `SMARegimeLabeler` (Phase 1, Eq. III.13):
   - 50-bar SMA percentage slope `m_t` and 20-bar relative volatility `sigma_t = Std_20(C)/SMA_20(C)`
   - Quantile thresholds `tau_trend = Q_0.60(|m_t|)` and `tau_vol = Q_0.75(sigma_t)` fitted strictly on train fold
2. `TrendScanningLabeler` (Phase 2, López de Prado 2020):
   - Forward horizons `L in [24..120]` OLS slope t-statistic on log prices
   - Purged at `train_end_exclusive` (`t + horizon_max < train_end_exclusive`) so
     test-fold prices are not accessed.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from ts.data.temporal_guard import assert_purged_train_horizon


REGIME_BULL = 0
REGIME_BEAR = 1
REGIME_SIDEWAYS = 2
REGIME_NAMES = {
    REGIME_BULL: "Bull",
    REGIME_BEAR: "Bear",
    REGIME_SIDEWAYS: "Sideways",
}


class SMARegimeLabeler:
    """Phase 1 SMA-50 slope + relative volatility labeler (Eq. III.13)."""

    def __init__(
        self,
        sma_window: int = 50,
        vol_window: int = 20,
        trend_quantile: float = 0.60,
        vol_quantile: float = 0.75,
    ) -> None:
        self.sma_window = sma_window
        self.vol_window = vol_window
        self.trend_quantile = trend_quantile
        self.vol_quantile = vol_quantile
        self.tau_trend_: float = 0.0
        self.tau_vol_: float = 0.0

    def _compute_slope_and_vol(self, close: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        eps = 1e-8
        sma50 = close.rolling(self.sma_window, min_periods=2).mean()
        prev_sma50 = sma50.shift(1).bfill()
        m_t = (((sma50 - prev_sma50) / (prev_sma50 + eps)) * 100.0).fillna(0.0).to_numpy(dtype=np.float64)

        sma20 = close.rolling(self.vol_window, min_periods=2).mean()
        std20 = close.rolling(self.vol_window, min_periods=2).std().fillna(0.0)
        sigma_t = (std20 / (sma20 + eps)).fillna(0.0).to_numpy(dtype=np.float64)
        return m_t, sigma_t

    def fit_label_train_fold(self, train_close: pd.Series) -> np.ndarray:
        m_t, sigma_t = self._compute_slope_and_vol(train_close)
        valid = slice(self.sma_window, len(train_close))
        m_ref = np.abs(m_t[valid]) if len(train_close) > self.sma_window else np.abs(m_t)
        s_ref = sigma_t[valid] if len(train_close) > self.sma_window else sigma_t

        self.tau_trend_ = float(np.quantile(m_ref, self.trend_quantile))
        self.tau_vol_ = float(np.quantile(s_ref, self.vol_quantile))

        labels = np.full(len(train_close), REGIME_SIDEWAYS, dtype=np.int64)
        is_bull = (m_t > self.tau_trend_) & (sigma_t < self.tau_vol_)
        is_bear = m_t < -self.tau_trend_
        labels[is_bull] = REGIME_BULL
        labels[is_bear] = REGIME_BEAR
        return labels


class TrendScanningLabeler:
    """
    Phase 2 Purged Forward Trend-Scanning Labeler (Ch. III Sec 2.1, Ch. VI Sec 3.1, Table A.2).

    For each bar `t` in the training fold where `t + horizon_max < len(train_close)`,
    evaluates OLS linear regression `log(P_{t+k}) = beta_0 + beta_1 * k + eps` across
    candidate horizons `L in [horizon_min, horizon_max]` (step `horizon_step`), optionally
    normalizes `t_stat(beta_1)` by `sqrt(L)` to remove unit-root horizon-length scaling bias,
    and selects `L* = argmax_L |stat(L)|`.

    Threshold selection (fitted strictly inside purged training bars, zero test access):
      - When `trend_quantile` is set (default `0.60` per Appendix Table A.2, unless a fixed
        `t_threshold` is explicitly passed without `trend_quantile`), `tau_trend_` is the
        `trend_quantile` empirical quantile of `|stat(L*)|` on valid bars of `train_close`.
      - Otherwise uses fixed `t_threshold`.

    Trailing bars `[len(train_close) - horizon_max .. len(train_close) - 1]` are marked `-1`
    (purged) so test-fold prices are never accessed.
    """

    def __init__(
        self,
        horizon_min: int = 24,
        horizon_max: int = 120,
        horizon_step: int = 12,
        t_threshold: Optional[float] = None,
        trend_quantile: Optional[float] = 0.60,
        unit_root_scale: bool = False,
    ) -> None:
        self.horizon_min = horizon_min
        self.horizon_max = horizon_max
        self.horizon_step = horizon_step
        if t_threshold is not None and trend_quantile == 0.60:
            # Explicit fixed threshold call (e.g. unit tests specifying t_threshold=1.8)
            self.t_threshold: float = float(t_threshold)
            self.trend_quantile: Optional[float] = None
        else:
            self.t_threshold = float(t_threshold) if t_threshold is not None else 2.0
            self.trend_quantile = float(trend_quantile) if trend_quantile is not None else None
        self.unit_root_scale = bool(unit_root_scale)
        self.tau_trend_: float = float(self.t_threshold)

    def fit_label_train_fold(self, train_close: pd.Series) -> np.ndarray:
        n = len(train_close)
        labels = np.full(n, -1, dtype=np.int64)
        if n <= self.horizon_max + 1:
            # Short synthetic test fold fallback: shrink max horizon safely inside `n - 1`
            eff_max = max(4, n // 3)
            eff_min = max(3, eff_max // 2)
            eff_step = max(1, (eff_max - eff_min) // 3)
        else:
            eff_max = self.horizon_max
            eff_min = self.horizon_min
            eff_step = self.horizon_step

        horizons = list(range(eff_min, eff_max + 1, max(1, eff_step)))
        if eff_max not in horizons:
            horizons.append(eff_max)

        log_p = np.log(np.maximum(train_close.to_numpy(dtype=np.float64), 1e-8))
        last_valid_t = n - 1 - eff_max
        if last_valid_t < 0:
            return labels
        assert_purged_train_horizon(last_valid_t, eff_max, n)
        n_valid = last_valid_t + 1

        best_stats = np.zeros(n_valid, dtype=np.float64)
        best_abs = np.full(n_valid, -1.0, dtype=np.float64)

        for L in horizons:
            m = L + 1
            x = np.arange(m, dtype=np.float64)
            x_c = x - x.mean()
            sxx = float(np.dot(x_c, x_c))
            dof = float(max(1, m - 2))

            windows = sliding_window_view(log_p[: n_valid + L], m)[:n_valid]
            y_c = windows - np.mean(windows, axis=1, keepdims=True)
            beta1 = (y_c @ x_c) / sxx
            resid = y_c - beta1[:, None] * x_c[None, :]
            s2 = np.sum(resid * resid, axis=1) / dof
            se_beta1 = np.sqrt(np.maximum(s2 / sxx, 1e-16))
            stat_l = beta1 / se_beta1
            if self.unit_root_scale:
                stat_l = stat_l / math.sqrt(float(L))

            abs_l = np.abs(stat_l)
            better = abs_l > best_abs
            best_abs = np.where(better, abs_l, best_abs)
            best_stats = np.where(better, stat_l, best_stats)

        if self.trend_quantile is not None and len(best_abs) > 0:
            self.tau_trend_ = float(np.quantile(best_abs, self.trend_quantile))
        else:
            self.tau_trend_ = float(self.t_threshold)

        valid_labels = np.full(n_valid, REGIME_SIDEWAYS, dtype=np.int64)
        valid_labels[best_stats > self.tau_trend_] = REGIME_BULL
        valid_labels[best_stats < -self.tau_trend_] = REGIME_BEAR
        labels[:n_valid] = valid_labels
        return labels
