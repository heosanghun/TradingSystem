"""
Robust Sharpe Ratio Statistical Significance Tests (Chapter VI Sec 6, p. 72).

Implements genuine Ledoit & Wolf (2008) "Robust performance hypothesis testing with the Sharpe ratio"
(*Journal of Empirical Finance*, 15(5), 850–859):
1. Asymptotic HAC (Parzen / Newey-West kernel) delta-method standard error for `ΔSR = SR_1 - SR_2`.
2. Studentized Circular Block Bootstrap (`block_size = 24` bars = 1 day, `B = 5,000` replications)
   for `H_0: SR_proposed == SR_baseline` on dependent, heavy-tailed hourly financial returns.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np
from scipy import stats


@dataclass
class SharpeSignificanceResult:
    baseline_name: str
    sr_proposed_ann: float
    sr_baseline_ann: float
    delta_sr_ann: float
    hac_z_stat: float
    hac_p_value: float
    bootstrap_p_value: float
    ci95_delta_sr_low: float
    ci95_delta_sr_high: float
    block_size: int
    n_bootstrap: int

    def to_dict(self) -> Dict[str, float | int | str]:
        return asdict(self)


def _compute_sr_and_grad(r1: np.ndarray, r2: np.ndarray) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Delta-method moment function for `f(μ_1, μ_2, γ_1, γ_2) = μ_1 / sqrt(γ_1 - μ_1^2) - μ_2 / sqrt(γ_2 - μ_2^2)`
    from Ledoit & Wolf (2008), Section 3.1.
    """
    mu1 = float(np.mean(r1))
    mu2 = float(np.mean(r2))
    g1 = float(np.mean(r1 ** 2))
    g2 = float(np.mean(r2 ** 2))

    var1 = max(g1 - mu1 ** 2, 1e-16)
    var2 = max(g2 - mu2 ** 2, 1e-16)
    sd1 = float(np.sqrt(var1))
    sd2 = float(np.sqrt(var2))

    sr1 = mu1 / sd1
    sr2 = mu2 / sd2
    diff = sr1 - sr2

    # Gradient ∇f(ν) = [ γ_1 / var_1^{3/2}, -γ_2 / var_2^{3/2}, -0.5 μ_1 / var_1^{3/2}, +0.5 μ_2 / var_2^{3/2} ]
    grad = np.array(
        [
            g1 / (var1 ** 1.5),
            -g2 / (var2 ** 1.5),
            -0.5 * mu1 / (var1 ** 1.5),
            0.5 * mu2 / (var2 ** 1.5),
        ],
        dtype=np.float64,
    )
    y_t = np.column_stack([r1 - mu1, r2 - mu2, r1 ** 2 - g1, r2 ** 2 - g2])
    return sr1, sr2, diff, grad, y_t


def _newey_west_se(grad: np.ndarray, y_t: np.ndarray, max_lag: int = 24) -> float:
    """HAC standard error `s(ΔSR) = sqrt((∇f^T Ψ_hat ∇f) / T)` using Bartlett/Newey-West weights."""
    t_len = len(y_t)
    if t_len < 4:
        return 1e-6
    u_t = y_t @ grad  # scalar score series `∇f^T y_t` of length T! (O(T) exact equivalence to 4x4 matrix!)
    gamma_0 = float(np.dot(u_t, u_t)) / t_len
    l_max = min(max_lag, t_len - 1)
    lrv = gamma_0
    for lag in range(1, l_max + 1):
        w = 1.0 - lag / (l_max + 1.0)
        gamma_k = float(np.dot(u_t[lag:], u_t[:-lag])) / t_len
        lrv += 2.0 * w * gamma_k
    lrv = max(lrv, 1e-18)
    return float(np.sqrt(lrv / t_len))


def ledoit_wolf_and_block_bootstrap_test(
    returns_proposed: np.ndarray,
    returns_baseline: np.ndarray,
    baseline_name: str = "Baseline",
    bars_per_year: int = 8760,
    block_size: int = 24,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> SharpeSignificanceResult:
    """
    Runs both the asymptotic HAC Ledoit-Wolf z-test and the Circular Block Bootstrap test
    (`block_size = 24`, `n_bootstrap = 5,000`) specified in Chapter VI Sec 6 (p. 72).
    """
    r1 = np.asarray(returns_proposed, dtype=np.float64)
    r2 = np.asarray(returns_baseline, dtype=np.float64)
    if len(r1) != len(r2):
        raise ValueError(f"Return series length mismatch: {len(r1)} vs {len(r2)}")

    n = len(r1)
    sqrt_ann = float(np.sqrt(bars_per_year))

    sr1, sr2, diff_bar, grad, y_t = _compute_sr_and_grad(r1, r2)
    se_bar = _newey_west_se(grad, y_t, max_lag=block_size)
    z_stat = diff_bar / max(se_bar, 1e-12)
    hac_pval = float(2.0 * (1.0 - stats.norm.cdf(abs(z_stat))))

    # Circular Block Bootstrap (Politis & Romano; Ledoit & Wolf 2008 Sec 3.2)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n, size=(n_bootstrap, n_blocks))
    offsets = np.arange(block_size, dtype=np.int64)[None, :]

    boot_diffs_ann = np.empty(n_bootstrap, dtype=np.float64)
    exceed_count = 0

    for b in range(n_bootstrap):
        idx = (starts[b, :, None] + offsets).ravel()[:n] % n
        b1 = r1[idx]
        b2 = r2[idx]
        _, _, b_diff, b_grad, b_y = _compute_sr_and_grad(b1, b2)
        b_se = _newey_west_se(b_grad, b_y, max_lag=block_size)
        # Studentized circular block bootstrap statistic `|ΔSR* - ΔSR| / se* >= |ΔSR| / se`
        z_star = abs(b_diff - diff_bar) / max(b_se, 1e-12)
        if z_star >= abs(z_stat):
            exceed_count += 1
        boot_diffs_ann[b] = b_diff * sqrt_ann

    boot_pval = float((exceed_count + 1) / (n_bootstrap + 1))
    ci_low, ci_high = np.percentile(boot_diffs_ann, [2.5, 97.5])

    return SharpeSignificanceResult(
        baseline_name=baseline_name,
        sr_proposed_ann=sr1 * sqrt_ann,
        sr_baseline_ann=sr2 * sqrt_ann,
        delta_sr_ann=diff_bar * sqrt_ann,
        hac_z_stat=float(z_stat),
        hac_p_value=hac_pval,
        bootstrap_p_value=boot_pval,
        ci95_delta_sr_low=float(ci_low),
        ci95_delta_sr_high=float(ci_high),
        block_size=block_size,
        n_bootstrap=n_bootstrap,
    )
