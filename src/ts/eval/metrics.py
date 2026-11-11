"""
Chapter VI Performance, Crisis Drawdown, Distribution & Stationarity Metrics (Eqs. VI.1–VI.5).

Implements:
- Table 6.2 & Eqs. (VI.1)–(VI.2): Log-return sample statistics (`n`, mean, std, min, max,
  Pearson kurtosis where Normal = 3.0), ADF test on price `P_t` and log return `r_t`,
  Ljung-Box Q test on squared returns `r_t^2`, and Jarque-Bera normality test.
- Tables 6.6–6.7 & Eqs. (VI.3)–(VI.5): Cumulative Return, CAGR, Annualized Sharpe,
  Annualized Sortino (downside semi-deviation over full `n`), and Maximum Drawdown (MDD).
- Table 6.8: Crisis sub-period drawdown (`window_mdd_pct`) & start-to-end return (`period_return_pct`)
  for Luna/Rate-Hike (`2022.05~2022.06`) and FTX Collapse (`2022.11.07~11.09`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller


@dataclass
class DistributionStats:
    symbol: str
    n_bars: int
    mean: float
    std: float
    min_val: float
    max_val: float
    pearson_kurtosis: float
    skewness: float
    adf_price_stat: float
    adf_price_pvalue: float
    adf_ret_stat: float
    adf_ret_pvalue: float
    ljung_box_r2_stat: float
    ljung_box_r2_pvalue: float
    jarque_bera_stat: float
    jarque_bera_pvalue: float

    def to_dict(self) -> Dict[str, float | int | str]:
        return asdict(self)


@dataclass
class PerformanceSummary:
    cum_return_pct: float
    cagr_pct: float
    sharpe: float
    sortino: float
    mdd_pct: float
    win_rate_pct: float
    avg_turnover: float
    total_bars: int
    final_equity: float

    def to_dict(self) -> Dict[str, float | int]:
        return asdict(self)


@dataclass
class CrisisLossMetrics:
    window_name: str
    start_utc: str
    end_utc: str
    n_bars: int
    window_mdd_pct: float
    period_return_pct: float

    def to_dict(self) -> Dict[str, float | int | str]:
        return asdict(self)


def compute_distribution_and_stationarity_stats(
    close_series: pd.Series,
    symbol: str = "BTCUSDT",
    ljung_box_lags: int = 24,
) -> DistributionStats:
    """
    Computes exact Table 6.2 statistics plus Eq. (VI.1) ADF & Eq. (VI.2) Ljung-Box tests.
    Note: Reports Pearson kurtosis (`fisher=False`, Normal = 3.0) to match Thesis Ch. VI Sec 2.2 (`K = 15.42 vs 3`).
    """
    prices = close_series.dropna().to_numpy(dtype=np.float64)
    if len(prices) < 30:
        raise ValueError(f"Need at least 30 price observations for {symbol}, got {len(prices)}")

    log_ret = np.diff(np.log(np.maximum(prices, 1e-12)))
    n = int(len(log_ret))

    mean_r = float(np.mean(log_ret))
    std_r = float(np.std(log_ret, ddof=1))
    min_r = float(np.min(log_ret))
    max_r = float(np.max(log_ret))
    kurt_pearson = float(stats.kurtosis(log_ret, fisher=False, bias=False))
    skew_r = float(stats.skew(log_ret, bias=False))

    # Eq. VI.1: ADF with constant + trend ("ct") on level price, constant ("c") on log returns
    adf_p = adfuller(prices, regression="ct", maxlag=24, autolag="AIC")
    adf_r = adfuller(log_ret, regression="c", maxlag=24, autolag="AIC")

    # Eq. VI.2: Ljung-Box Q test on squared returns r_t^2
    lb_df = acorr_ljungbox(log_ret ** 2, lags=[min(ljung_box_lags, max(1, n // 5))], return_df=True)
    lb_stat = float(lb_df["lb_stat"].iloc[-1])
    lb_pval = float(lb_df["lb_pvalue"].iloc[-1])

    jb_stat, jb_pval = stats.jarque_bera(log_ret)

    return DistributionStats(
        symbol=symbol,
        n_bars=n,
        mean=mean_r,
        std=std_r,
        min_val=min_r,
        max_val=max_r,
        pearson_kurtosis=kurt_pearson,
        skewness=skew_r,
        adf_price_stat=float(adf_p[0]),
        adf_price_pvalue=float(adf_p[1]),
        adf_ret_stat=float(adf_r[0]),
        adf_ret_pvalue=float(adf_r[1]),
        ljung_box_r2_stat=lb_stat,
        ljung_box_r2_pvalue=lb_pval,
        jarque_bera_stat=float(jb_stat),
        jarque_bera_pvalue=float(jb_pval),
    )


def compute_performance_summary(
    equity_curve: np.ndarray,
    bar_net_returns: np.ndarray,
    weights: np.ndarray,
    bars_per_year: int = 8760,
    risk_free_annual: float = 0.0,
) -> PerformanceSummary:
    """
    Exact Eqs. (VI.3)–(VI.5) on a causal portfolio simulation trajectory.
    - `equity_curve`: length `n + 1` (starts at `V_0`) or length `n`
    - `bar_net_returns`: length `n` net arithmetic bar returns `R_p(t)`
    """
    r = np.asarray(bar_net_returns, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    eq = np.asarray(equity_curve, dtype=np.float64)
    n = len(r)
    if n == 0:
        return PerformanceSummary(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)

    v0 = float(eq[0])
    vt = float(eq[-1])
    cum_ret = (vt / max(v0, 1e-12) - 1.0) * 100.0

    years = max(float(n) / float(bars_per_year), 1e-8)
    ratio_v = max(vt / max(v0, 1e-12), 1e-12)
    cagr = (ratio_v ** (1.0 / years) - 1.0) * 100.0

    rf_bar = float(risk_free_annual) / float(bars_per_year)
    excess = r - rf_bar
    mu_ex = float(np.mean(excess))
    sd_ex = float(np.std(excess, ddof=1)) if n > 1 else 0.0
    sqrt_ann = float(np.sqrt(bars_per_year))

    sharpe = (mu_ex / sd_ex) * sqrt_ann if sd_ex > 1e-12 else 0.0

    # Eq. VI.4: Downside deviation over all n observations `sqrt((1/n) Σ min(0, R_p - R_f)^2)`
    downside_sq = np.minimum(0.0, excess) ** 2
    dd_bar = float(np.sqrt(np.mean(downside_sq)))
    sortino = (mu_ex / dd_bar) * sqrt_ann if dd_bar > 1e-12 else 0.0

    # Eq. VI.5: Maximum Drawdown (%)
    running_peak = np.maximum.accumulate(eq)
    drawdowns = (eq - running_peak) / np.maximum(running_peak, 1e-12)
    mdd_pct = float(np.min(drawdowns)) * 100.0

    active_mask = np.abs(w) > 1e-8
    if np.any(active_mask):
        win_rate = float(np.mean(r[active_mask] > 0.0)) * 100.0
    else:
        win_rate = 0.0

    prev_w = np.empty_like(w)
    prev_w[0] = 0.0
    prev_w[1:] = w[:-1]
    avg_turnover = float(np.mean(np.abs(w - prev_w)))

    return PerformanceSummary(
        cum_return_pct=cum_ret,
        cagr_pct=cagr,
        sharpe=sharpe,
        sortino=sortino,
        mdd_pct=mdd_pct,
        win_rate_pct=win_rate,
        avg_turnover=avg_turnover,
        total_bars=n,
        final_equity=vt,
    )


def compute_crisis_window_loss(
    equity_series: pd.Series,
    start_utc: str,
    end_utc: str,
    window_name: str = "",
) -> CrisisLossMetrics:
    """
    Computes crisis sub-period loss for Table 6.8 (`루나·금리 인상`, `FTX 파산`).
    Records BOTH:
      - `window_mdd_pct`: Peak-to-trough drawdown (%) inside `[start_utc, end_utc]`
      - `period_return_pct`: Cumulative start-to-end return (%) across `[start_utc, end_utc]`
    """
    s_ts = pd.Timestamp(start_utc).tz_convert("UTC") if pd.Timestamp(start_utc).tzinfo else pd.Timestamp(start_utc, tz="UTC")
    e_ts = pd.Timestamp(end_utc).tz_convert("UTC") if pd.Timestamp(end_utc).tzinfo else pd.Timestamp(end_utc, tz="UTC")

    sub = equity_series.loc[(equity_series.index >= s_ts) & (equity_series.index <= e_ts)]
    if len(sub) < 2:
        return CrisisLossMetrics(
            window_name=window_name,
            start_utc=str(s_ts),
            end_utc=str(e_ts),
            n_bars=len(sub),
            window_mdd_pct=0.0,
            period_return_pct=0.0,
        )

    vals = sub.to_numpy(dtype=np.float64)
    peaks = np.maximum.accumulate(vals)
    dds = (vals - peaks) / np.maximum(peaks, 1e-12)
    win_mdd = float(np.min(dds)) * 100.0
    per_ret = (float(vals[-1]) / max(float(vals[0]), 1e-12) - 1.0) * 100.0

    return CrisisLossMetrics(
        window_name=window_name,
        start_utc=str(s_ts),
        end_utc=str(e_ts),
        n_bars=int(len(vals)),
        window_mdd_pct=win_mdd,
        period_return_pct=per_ret,
    )
