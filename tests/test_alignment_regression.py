"""
Regression tests for decision-time alignment (collab T-001).

A weight decided at the close of bar `t` must earn the return of bar `t + 1`, never of bar `t`.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ts.eval.backtester import CausalPortfolioBacktester

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _random_walk_ohlcv(n: int = 2000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ret = rng.standard_normal(n) * 0.005
    close = 30_000.0 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[30_000.0], close[:-1]])
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    idx = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": np.full(n, 1e4)},
        index=idx,
    )


def test_run_decisions_holds_weight_on_the_next_bar() -> None:
    df = _random_walk_ohlcv()
    close = df["close"].to_numpy()
    bar_ret = np.zeros(len(df))
    bar_ret[1:] = close[1:] / close[:-1] - 1.0

    rng = np.random.default_rng(0)
    decisions = rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=len(df))

    res = CausalPortfolioBacktester().run_decisions(df, decisions, phase=1)

    assert res.weights[0] == 0.0
    np.testing.assert_array_equal(res.weights[1:], decisions[:-1])
    np.testing.assert_allclose(res.bar_gross_returns[1:], decisions[:-1] * bar_ret[1:], atol=1e-12)


def test_same_bar_sign_rule_has_no_edge_on_a_random_walk() -> None:
    """
    `sign(return of bar t)` decided at the close of bar `t` is pure hindsight if it is paid
    bar `t`'s own return. On a random walk it must not make money once aligned correctly.
    """
    df = _random_walk_ohlcv(n=5000, seed=11)
    close = df["close"].to_numpy()
    bar_ret = np.zeros(len(df))
    bar_ret[1:] = close[1:] / close[:-1] - 1.0
    decisions = np.sign(bar_ret)

    bt = CausalPortfolioBacktester()
    aligned = bt.run_decisions(df, decisions, phase=1)
    leaked = bt.run(df, decisions, phase=1)

    # The misaligned call earns |r_t| on every bar; the aligned call has no systematic edge.
    assert float(np.mean(leaked.bar_gross_returns)) > 0.002
    assert abs(float(np.mean(aligned.bar_gross_returns))) < 0.0005


def test_run_scripts_do_not_backtest_model_weights_without_the_shift() -> None:
    """Only Buy & Hold (held from the first bar) may call `bt.run` directly in run scripts."""
    pattern = re.compile(r"bt\.run\(([^)]*)\)")
    offenders = []
    for name in ("run_study1.py", "run_study3.py", "run_baselines_and_eval.py"):
        text = (PROJECT_ROOT / "scripts" / name).read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            for m in pattern.finditer(line):
                if "bh_w" not in m.group(1):
                    offenders.append(f"{name}:{line_no}: {line.strip()}")
    assert not offenders, "Model weights passed to bt.run without decision shift:\n" + "\n".join(offenders)
