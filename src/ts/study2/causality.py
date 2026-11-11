"""
Pairwise Granger Causality & Gaussian-KDE Normalized Transfer Entropy (Ch. IV Sec 2.1–2.2).

Implements Eqs. (IV.1)–(IV.4) & Table A.3:
1. Bivariate VAR Granger F-test (`p = q = 24`, `p < 0.05`, Eq. IV.1)
2. Lag-summarized Gaussian KDE Transfer Entropy (`k = 24`, Silverman bandwidth
   `h = 1.06 * sigma_hat * n^(-1/5)`, normalized by `H(Y_t | Y_past)` into `[0, 1]`, Eqs. IV.2–IV.4).
   Note (`구현계획_261111.md` Sec 4 #8): Direct 49-D KDE on `n = 1,440` hourly samples suffers
   from severe curse-of-dimensionality; we project `Y_{t-1:t-24}` and `X_{t-1:t-24}` onto
   standardized lag-block summaries `(lag_1, mean_lags_2_to_k)` so Silverman Gaussian KDE
   remains statistically consistent and tractable across ~3,540 asset pairs.
3. Candidate OR-rule: `(p_GC < 0.05) OR (TE_tilde > Q_0.90)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class CandidateEdge:
    edge_id: str
    source: str
    target: str
    granger_f: float
    granger_p: float
    te_norm: float
    sign_corr: float  # sign of lagged cross-correlation (+1 positive, -1 danger/inverse)


def granger_var_ftest(x: np.ndarray, y: np.ndarray, lag: int = 24) -> Tuple[float, float]:
    """
    Exact OLS SSR F-test for `X -> Y` under Eq. (IV.1):
      Restricted:   Y_t = c + Σ_{i=1}^p α_i Y_{t-i} + e_r
      Unrestricted: Y_t = c + Σ_{i=1}^p α_i Y_{t-i} + Σ_{j=1}^p β_j X_{t-j} + e_u
    Returns `(f_stat, p_value)`.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    n_total = len(y_arr)
    eff_lag = min(lag, max(1, (n_total - 5) // 3))
    n = n_total - eff_lag
    if n <= 2 * eff_lag + 2:
        return 0.0, 1.0

    if np.std(x_arr) < 1e-12 or np.std(y_arr) < 1e-12:
        return 0.0, 1.0

    target_y = y_arr[eff_lag:]
    y_lags = np.column_stack([y_arr[eff_lag - k : n_total - k] for k in range(1, eff_lag + 1)])
    x_lags = np.column_stack([x_arr[eff_lag - k : n_total - k] for k in range(1, eff_lag + 1)])

    ones = np.ones((n, 1), dtype=np.float64)
    design_r = np.hstack([ones, y_lags])
    design_u = np.hstack([ones, y_lags, x_lags])

    beta_r, _, _, _ = np.linalg.lstsq(design_r, target_y, rcond=None)
    resid_r = target_y - design_r @ beta_r
    ssr_r = float(np.dot(resid_r, resid_r))

    beta_u, _, _, _ = np.linalg.lstsq(design_u, target_y, rcond=None)
    resid_u = target_y - design_u @ beta_u
    ssr_u = float(np.dot(resid_u, resid_u))

    df_num = eff_lag
    df_den = n - (2 * eff_lag + 1)
    if df_den <= 0 or ssr_u <= 1e-16:
        return 0.0, 1.0

    f_stat = max(0.0, ((ssr_r - ssr_u) / df_num) / (ssr_u / df_den))
    p_val = float(stats.f.sf(f_stat, df_num, df_den))
    return f_stat, p_val


def _silverman_kde_entropy_1d_or_nd(samples_nd: np.ndarray, eval_sub: int = 256) -> float:
    """
    Differential entropy estimate `-E[log p_hat(z)]` using isotropic product Gaussian kernel
    with componentwise Silverman bandwidth `h_d = 1.06 * sigma_d * n^(-1/5)` (Eq. IV.3).
    Evaluates on a deterministic strided subsample of size `<= eval_sub` for O(N * M) speed.
    """
    z = np.asarray(samples_nd, dtype=np.float64)
    if z.ndim == 1:
        z = z[:, None]
    n, d = z.shape
    if n < 5:
        return 0.0

    std = np.std(z, axis=0, ddof=1)
    std = np.where(std < 1e-8, 1.0, std)
    h = 1.06 * std * (n ** (-0.2))  # Eq. IV.3 Silverman rule

    if n > eval_sub:
        idx = np.linspace(0, n - 1, num=eval_sub, dtype=int)
        q = z[idx]
    else:
        q = z

    # Scaled pairwise squared distance: (M, N)
    z_scaled = z / h[None, :]
    q_scaled = q / h[None, :]
    sq_dist = (
        np.sum(q_scaled ** 2, axis=1, keepdims=True)
        + np.sum(z_scaled ** 2, axis=1)[None, :]
        - 2.0 * (q_scaled @ z_scaled.T)
    )
    sq_dist = np.maximum(sq_dist, 0.0)

    log_norm = -0.5 * d * np.log(2.0 * np.pi) - float(np.sum(np.log(h))) - np.log(n)
    # log-sum-exp across N reference points
    max_term = np.max(-0.5 * sq_dist, axis=1, keepdims=True)
    log_p = log_norm + max_term.squeeze(1) + np.log(
        np.sum(np.exp(-0.5 * sq_dist - max_term), axis=1) + 1e-16
    )
    return float(-np.mean(log_p))


def normalized_transfer_entropy_kde(x: np.ndarray, y: np.ndarray, lag: int = 24) -> float:
    """
    Implements Eqs. (IV.2)–(IV.4):
      TE_{X->Y} = H(Y_t | Y_{t-1:t-k}) - H(Y_t | Y_{t-1:t-k}, X_{t-1:t-k})
      TE_tilde  = TE_{X->Y} / H(Y_t | Y_{t-1:t-k}) ∈ [0, 1]
    Uses standardized lag-summary coordinates `y_past = 0.5*y_{t-1} + 0.5*mean(y_{t-2:t-k})`
    and `x_past = 0.5*x_{t-1} + 0.5*mean(x_{t-2:t-k})` with Silverman Gaussian KDE.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    n_total = len(y_arr)
    eff_lag = min(lag, max(1, n_total // 4))
    if n_total <= eff_lag + 8 or np.std(x_arr) < 1e-12 or np.std(y_arr) < 1e-12:
        return 0.0

    # Standardize within window
    xs = (x_arr - np.mean(x_arr)) / (np.std(x_arr) + 1e-8)
    ys = (y_arr - np.mean(y_arr)) / (np.std(y_arr) + 1e-8)

    yt = ys[eff_lag:]
    if eff_lag == 1:
        y_past = ys[:-1]
        x_past = xs[:-1]
    else:
        y_lags = np.column_stack([ys[eff_lag - k : n_total - k] for k in range(1, eff_lag + 1)])
        x_lags = np.column_stack([xs[eff_lag - k : n_total - k] for k in range(1, eff_lag + 1)])
        y_past = 0.5 * y_lags[:, 0] + 0.5 * np.mean(y_lags[:, 1:], axis=1)
        x_past = 0.5 * x_lags[:, 0] + 0.5 * np.mean(x_lags[:, 1:], axis=1)

    # H(Y_t | Y_past) = H(Y_t, Y_past) - H(Y_past)
    h_yp = _silverman_kde_entropy_1d_or_nd(y_past)
    h_yt_yp = _silverman_kde_entropy_1d_or_nd(np.column_stack([yt, y_past]))
    h_cond_y = h_yt_yp - h_yp

    # H(Y_t | Y_past, X_past) = H(Y_t, Y_past, X_past) - H(Y_past, X_past)
    h_yp_xp = _silverman_kde_entropy_1d_or_nd(np.column_stack([y_past, x_past]))
    h_yt_yp_xp = _silverman_kde_entropy_1d_or_nd(np.column_stack([yt, y_past, x_past]))
    h_cond_yx = h_yt_yp_xp - h_yp_xp

    te = max(0.0, h_cond_y - h_cond_yx)
    # Shift differential conditional entropy to positive scale via Gaussian entropy bound if needed
    denom = max(abs(h_cond_y), te + 1e-6, 0.5)
    te_norm = float(np.clip(te / denom, 0.0, 1.0))
    return te_norm


def discover_candidate_edges(
    returns_window_df: pd.DataFrame,
    granger_lag: int = 24,
    granger_p_threshold: float = 0.05,
    te_lag: int = 24,
    te_top_quantile: float = 0.90,
) -> List[CandidateEdge]:
    """
    Evaluate all ordered asset pairs `(X -> Y, X != Y)` on the 60-day rolling 1h return window
    and retain candidates passing `(granger_p < 0.05) OR (te_norm > Q_0.90)`.
    """
    cols = list(returns_window_df.columns)
    mat = returns_window_df.to_numpy(dtype=np.float64)
    m = len(cols)
    if m < 2:
        return []

    raw_records: List[Tuple[str, str, float, float, float, float]] = []
    for i in range(m):
        x = mat[:, i]
        if np.std(x) < 1e-12:
            continue
        for j in range(m):
            if i == j:
                continue
            y = mat[:, j]
            if np.std(y) < 1e-12:
                continue
            f_stat, p_val = granger_var_ftest(x, y, lag=granger_lag)
            te_n = normalized_transfer_entropy_kde(x, y, lag=te_lag)
            # Lag-1 cross-correlation sign distinguishes positive momentum vs tail-risk contagion
            corr = float(np.corrcoef(x[:-1], y[1:])[0, 1]) if len(x) > 2 else 0.0
            if np.isnan(corr):
                corr = 0.0
            raw_records.append((cols[i], cols[j], f_stat, p_val, te_n, corr))

    if not raw_records:
        return []

    all_te = np.array([r[4] for r in raw_records], dtype=np.float64)
    theta_te = float(np.quantile(all_te, te_top_quantile))

    candidates: List[CandidateEdge] = []
    edge_counter = 1
    for src, dst, f_stat, p_val, te_n, corr in raw_records:
        if (p_val < granger_p_threshold) or (te_n > theta_te and te_n > 0.0):
            candidates.append(
                CandidateEdge(
                    edge_id=f"E{edge_counter:02d}",
                    source=src,
                    target=dst,
                    granger_f=f_stat,
                    granger_p=p_val,
                    te_norm=te_n,
                    sign_corr=corr,
                )
            )
            edge_counter += 1

    return candidates
