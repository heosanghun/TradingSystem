"""Pairwise Granger F-tests and a lag-summary KDE information-flow proxy.

The transfer-entropy implementation compresses each 24-lag history into one
weighted summary, estimates densities in at most three dimensions, and rescales
using max(abs(conditional entropy), TE + 1e-6, 0.5). It is an approximation, not
an estimator of the full 49-dimensional conditional distribution. No consistency
claim for that original estimand is made. Candidate selection remains the
configured Granger p-value OR upper TE quantile rule; it does not correct for
multiple comparisons and can retain many edges even for independent noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict
import hashlib
import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import lstsq, qr
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits


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
    OLS residual-sum-of-squares F-test for `X -> Y` under Eq. (IV.1):
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
    sq_dist = cdist(q_scaled, z_scaled, metric="sqeuclidean")

    log_norm = -0.5 * d * np.log(2.0 * np.pi) - float(np.sum(np.log(h))) - np.log(n)
    # log-sum-exp across N reference points
    max_term = np.max(-0.5 * sq_dist, axis=1, keepdims=True)
    log_p = log_norm + max_term.squeeze(1) + np.log(
        np.sum(np.exp(-0.5 * sq_dist - max_term), axis=1) + 1e-16
    )
    return float(-np.mean(log_p))


def normalized_transfer_entropy_kde(x: np.ndarray, y: np.ndarray, lag: int = 24) -> float:
    """
    Lag-summary approximation motivated by Eqs. (IV.2)–(IV.4):
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


@threadpool_limits.wrap(limits=1, user_api="blas")
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

    if not np.isfinite(mat).all():
        raise ValueError("Candidate discovery requires finite aligned returns")
    # Prepare each target's restricted regression and TE conditional entropy once.
    # Full-rank regressions use Frisch-Waugh-Lovell residualization. Degenerate
    # targets use the original pairwise least-squares path for matching semantics.
    n_total = len(mat)
    gc_lag = min(granger_lag, max(1, (n_total - 5) // 3))
    gc_n = n_total - gc_lag
    gc_ready = gc_n > 2 * gc_lag + 2
    lagged = [np.column_stack([mat[gc_lag-k:n_total-k, j]
                              for k in range(1, gc_lag+1)]) for j in range(m)] if gc_ready else []
    restricted = []
    if gc_ready:
        for j in range(m):
            design = np.column_stack([np.ones(gc_n), lagged[j]])
            q, r = qr(design, mode="economic")
            if np.linalg.matrix_rank(r) != design.shape[1]:
                restricted.append(None)
            else:
                target = mat[gc_lag:, j]
                residual = target - q @ (q.T @ target)
                restricted.append((q, residual, float(residual @ residual)))
    te_eff_lag = min(te_lag, max(1, n_total // 4))
    te_ready = n_total > te_eff_lag + 8
    te_prepared = []
    te_pair_entropy = {}
    if te_ready:
        for j in range(m):
            series = mat[:, j]
            zs = (series - np.mean(series)) / (np.std(series) + 1e-8)
            future = zs[te_eff_lag:]
            past = zs[:-1] if te_eff_lag == 1 else (
                .5 * zs[te_eff_lag-1:-1] + .5 * np.mean(np.column_stack([
                    zs[te_eff_lag-k:n_total-k] for k in range(2, te_eff_lag+1)]), axis=1))
            # The query subsample is drawn from reference observations, so every
            # density sum contains its own unit kernel. Products are safe from
            # all-zero underflow and reuse the same per-coordinate bandwidths.
            def coordinate_kernel(values):
                std = float(np.std(values, ddof=1))
                h = 1.06 * (std if std >= 1e-8 else 1.) * len(values) ** -.2
                query = values[np.linspace(0, len(values)-1, 256, dtype=int)] if len(values) > 256 else values
                kernel = np.exp(-.5 * cdist((query/h)[:, None], (values/h)[:, None], "sqeuclidean"))
                norm = -.5 * np.log(2*np.pi) - np.log(h)
                return kernel, norm
            kf, nf = coordinate_kernel(future)
            kp, npast = coordinate_kernel(past)
            joint = kf * kp
            log_n = np.log(len(future))
            h_past = -npast + log_n - np.mean(np.log(kp.sum(axis=1)))
            h_joint = -nf - npast + log_n - np.mean(np.log(joint.sum(axis=1)))
            te_prepared.append((kp, joint, npast, nf, h_joint - h_past, log_n))

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
            target_fit = restricted[j] if gc_ready else None
            if target_fit is None:
                f_stat, p_val = granger_var_ftest(x, y, lag=granger_lag)
            else:
                q, yr, ssr_r = target_fit
                xr = lagged[i] - q @ (q.T @ lagged[i])
                reference_scale = np.linalg.norm(lagged[i])
                residual_scale = np.linalg.norm(xr)
                if residual_scale <= np.finfo(float).eps * max(xr.shape) * reference_scale:
                    rank = -1
                else:
                    beta, _, rank, _ = lstsq(xr, yr, lapack_driver="gelsy")
                if rank < gc_lag:
                    f_stat, p_val = granger_var_ftest(x, y, lag=granger_lag)
                else:
                    ru = yr - xr @ beta
                    ssr_u = float(ru @ ru)
                    df_den = gc_n - (2 * gc_lag + 1)
                    f_stat = max(0., ((ssr_r - ssr_u) / gc_lag) / (ssr_u / df_den)) if ssr_u > 1e-16 else 0.
                    p_val = float(stats.f.sf(f_stat, gc_lag, df_den))
            if te_ready:
                kp, joint, npast, nf, cond, log_n = te_prepared[j]
                xp, _, nx, _, _, _ = te_prepared[i]
                pair_key = (min(i, j), max(i, j))
                if pair_key not in te_pair_entropy:
                    sums = np.einsum("ij,ij->i", kp, xp)
                    te_pair_entropy[pair_key] = -npast - nx + log_n - np.mean(np.log(sums))
                joint_sums = np.einsum("ij,ij->i", joint, xp)
                h_all = -nf - npast - nx + log_n - np.mean(np.log(joint_sums))
                cond_xy = h_all - te_pair_entropy[pair_key]
                te = max(0., cond - cond_xy)
                te_n = float(np.clip(te / max(abs(cond), te + 1e-6, .5), 0., 1.))
            else:
                te_n = 0.
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


def discover_candidate_edges_cached(returns_window_df: pd.DataFrame, cache_path: Path,
                                    **parameters) -> List[CandidateEdge]:
    """Reuse a daily result only when input values, clock, columns and settings match."""
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(returns_window_df, index=True).values.tobytes())
    digest.update(json.dumps(list(returns_window_df.columns)).encode())
    digest.update(json.dumps(parameters, sort_keys=True).encode())
    digest.update(b"candidate-discovery-v2")
    key = digest.hexdigest()
    path = Path(cache_path)
    if path.exists():
        frame = pd.read_parquet(path)
        if frame.attrs.get("input_sha256") == key:
            return [CandidateEdge(**row) for row in frame.to_dict(orient="records")]
        logging.getLogger(__name__).info("Recomputing mismatched candidate cache %s", path)
    edges = discover_candidate_edges(returns_window_df, **parameters)
    frame = pd.DataFrame([asdict(e) for e in edges], columns=list(CandidateEdge.__dataclass_fields__))
    frame.attrs["input_sha256"] = key
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)
    return edges
