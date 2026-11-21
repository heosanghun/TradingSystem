"""Read-only examiner replay for P61. Optional retraining uses a temporary directory."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ts.config import PROJECT_ROOT
from ts.eval.backtester import CausalPortfolioBacktester


ARCHIVE = PROJECT_ROOT / "runs/baselines/trend_entry/P61_seed42"
CACHE = PROJECT_ROOT / "data/cache"
PERIODS = {
    "paper": ("2021-10-01", "2023-12-31", CACHE / "btcusdt_1h_20200801_20231231.parquet"),
    "reused": ("2024-01-01", "2026-10-31", CACHE / "p01_sources/confirmed_ohlcv.parquet"),
}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_metrics(returns):
    r = np.asarray(returns, dtype=float)
    equity = np.r_[1., np.cumprod(1. + r)]
    annual_mean = r.mean() * np.sqrt(8760.)
    return {
        "sortino": annual_mean / np.sqrt(np.mean(np.minimum(r, 0.) ** 2)),
        "sharpe": annual_mean / np.std(r, ddof=1),
        "cum_return_pct": (equity[-1] - 1.) * 100.,
        "cagr_pct": (equity[-1] ** (8760. / len(r)) - 1.) * 100.,
        "mdd_pct": np.min(equity / np.maximum.accumulate(equity) - 1.) * 100.,
    }


def rule_from_closes():
    """Reconstruct the two entry states one day at a time, without the strategy module."""
    old = pd.read_csv(CACHE / "p05_sources/BTCUSDT_1h.csv", usecols=["timestamp_utc", "close"],
                      index_col="timestamp_utc", parse_dates=True).close.loc[:"2020-07-31"]
    sources = [pd.read_parquet(path, columns=["close"]).close for _, _, path in PERIODS.values()]
    hourly = pd.concat([old, *sources])
    daily = hourly[hourly.index.hour == 23].asfreq("D")
    risk = np.log(daily).diff().rolling(30).std()
    result = pd.Series(0., index=daily.index)
    slow = base = strong = recovering = waiting = False
    for i in range(200, len(daily)):
        history = daily.iloc[i-200:i].to_numpy()
        price = daily.iat[i]
        if price > history[-55:].max():
            slow = True
        if price < history[-20:].min():
            slow = False
        in_trend = price > daily.iloc[i-199:i+1].mean() and price > history[-30]
        base = in_trend and (slow or base)
        strong = base and (strong or np.log(price/history[-7]) > 1.645*np.sqrt(7)*risk.iat[i])
        if price < history[-20:].min():
            waiting = True
        elif waiting and price > history[-10:].max():
            waiting = False
            if price > daily.iloc[i-49:i+1].mean():
                recovering = True
        if price < history[-5:].min():
            recovering = False
        if strong or recovering:
            result.iat[i] = min(1., .25 / max(1e-6, np.sqrt(365)*risk.iat[i]))
    return result


def replay(archive):
    info = json.loads((archive / "metrics.json").read_text())
    for path, field in [
        (Path(__file__).with_name("trend_entry_student.py"), "source_sha256"),
        (Path(__file__).with_name("momentum_condition_student.py"), "training_source_sha256"),
        (archive / "frozen.pt", "weights_sha256"),
    ]:
        if file_hash(path) != info[field]:
            raise ValueError(f"Digest mismatch: {path}")
    rule = rule_from_closes()
    for period, (start, end, path) in PERIODS.items():
        bars = pd.read_parquet(path).loc[start:end]
        decisions = pd.read_parquet(archive / f"{period}_decisions.parquet")
        returns = pd.read_parquet(archive / f"{period}_returns.parquet")
        for frame in (decisions, returns):
            pd.testing.assert_index_equal(frame.index, bars.index)
            if (not bars.index.is_unique or not bars.index.is_monotonic_increasing
                    or str(bars.index.tz) != "UTC" or not np.isfinite(frame).all().all()):
                raise ValueError(f"Invalid clock or values: {period}")
        missing = pd.date_range(bars.index[0], bars.index[-1], freq="h").difference(bars.index)
        print(f"{period}: {len(bars)} bars, missing inside span: {missing.tolist()}")
        dates = bars.index[bars.index.hour == 23]
        expected_rule = rule.loc[dates].reindex(bars.index).ffill().fillna(0.)
        np.testing.assert_allclose(expected_rule, decisions.rule, rtol=0., atol=1e-10)
        np.testing.assert_array_equal(decisions.student.loc[expected_rule.eq(0.)], 0.)
        for name in ("student", "rule", "untrained"):
            d = decisions[name]
            if not d.between(0., 1.).all() or d.diff().fillna(0.).loc[d.index.hour != 23].ne(0.).any():
                raise ValueError(f"Invalid spot decisions: {period}/{name}")
            result = CausalPortfolioBacktester().run_decisions(bars, d.to_numpy(), phase=2)
            np.testing.assert_array_equal(result.weights, d.shift(1, fill_value=0.))
            np.testing.assert_allclose(result.bar_net_returns, returns[name], rtol=0., atol=1e-12)
            numbers = independent_metrics(result.bar_net_returns)
            for key, value in numbers.items():
                np.testing.assert_allclose(value, info[period][name][key], rtol=0., atol=1e-8)
            print(name, " ".join(f"{key}={value:.6f}" for key, value in numbers.items()))
    return info


def retrain():
    import torch

    if not (CACHE / "p05_sources/BTCUSDT_1h.csv").is_file():
        raise FileNotFoundError("Offline replay needs data/cache/p05_sources/BTCUSDT_1h.csv")
    runner = (
        "import sys; from pathlib import Path; from ts.baselines import trend_entry_student as m; "
        "m.OUT=Path(sys.argv.pop(1)); m.main()"
    )
    with tempfile.TemporaryDirectory(prefix="p61_examiner_") as tmp:
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        subprocess.run([sys.executable, "-c", runner, tmp, "--policies", "P61", "--seed", "42"],
                       cwd=PROJECT_ROOT, env=env, check=True)
        fresh = Path(tmp) / ARCHIVE.name
        frozen = torch.load(ARCHIVE / "frozen.pt", map_location="cpu", weights_only=True)
        regenerated = torch.load(fresh / "frozen.pt", map_location="cpu", weights_only=True)
        if frozen.keys() != regenerated.keys():
            raise ValueError("Checkpoint fields differ")
        for key in frozen:
            torch.testing.assert_close(frozen[key], regenerated[key], rtol=0., atol=0.)
        for period in PERIODS:
            for kind in ("decisions", "returns"):
                name = f"{period}_{kind}.parquet"
                pd.testing.assert_frame_equal(pd.read_parquet(ARCHIVE / name),
                                              pd.read_parquet(fresh / name), check_exact=True)
        info = replay(fresh)
        original = json.loads((ARCHIVE / "metrics.json").read_text())
        if info["folds"] != original["folds"]:
            raise ValueError("Fold training or prediction counts differ")
        print(f"RETRAIN PASS: {len(frozen)} identical tensors and 4 identical hourly tables")


def sensitivity():
    from ts.baselines.selection_audit import intervals

    archive_audit = json.loads((ARCHIVE / "audit.json").read_text())
    archive_bands = json.loads((ARCHIVE / "selection_audit.json").read_text())
    folds = json.loads((ARCHIVE / "metrics.json").read_text())["folds"]
    for period, (start, end, path) in PERIODS.items():
        bars = pd.read_parquet(path).loc[start:end]
        d = pd.read_parquet(ARCHIVE / f"{period}_decisions.parquet").student
        r = pd.read_parquet(ARCHIVE / f"{period}_returns.parquet").student
        bt = CausalPortfolioBacktester()
        expensive = CausalPortfolioBacktester(phase2_fee_rate=.001, phase2_kappa=.2, phase2_eta=.1)
        quarterly = r.groupby([r.index.year, r.index.quarter])
        best = quarterly.apply(lambda s: (1. + s).prod()).idxmax()
        keep = (r.index.year != best[0]) | (r.index.quarter != best[1])
        cases = {
            "double_cost": expensive.run_decisions(bars, d.to_numpy(), phase=2).bar_net_returns,
            "extra_hour": bt.run_decisions(bars, d.shift(1, fill_value=0.).to_numpy(), phase=2).bar_net_returns,
            "sortino_without_best_quarter": r[keep],
        }
        if period == "paper":
            cash = d.copy()
            for fold, successor in zip(folds, folds[1:]):
                if fold["n_train"] == 0:
                    left, right = [pd.Timestamp(f["test_from"]).normalize() + pd.Timedelta(hours=23)
                                   for f in (fold, successor)]
                    cash.loc[(cash.index >= left) & (cash.index < right)] = 0.
            cases["cash_in_empty_training_fold"] = bt.run_decisions(bars, cash.to_numpy(), phase=2).bar_net_returns
        for case, values in cases.items():
            value = independent_metrics(values)["sortino"]
            saved = archive_audit[period][case]
            expected = saved["sortino"] if isinstance(saved, dict) else saved
            np.testing.assert_allclose(value, expected, atol=1e-8, rtol=0.)
            print(period, case, f"{value:.6f}")
        book = {}
        for i in (28, *range(31, 63)):
            s = pd.read_parquet(ARCHIVE.parent / f"P{i}_seed42" / f"{period}_returns.parquet").student
            pd.testing.assert_index_equal(s.index, r.index)
            book[f"P{i}"] = s
        bands = intervals(pd.DataFrame(book))
        for key, band in bands.items():
            np.testing.assert_allclose(band, archive_bands[period][key], atol=1e-8, rtol=0.)
            print(period, key, band)
    print("SENSITIVITY PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--sensitivity", action="store_true")
    args = parser.parse_args()
    replay(ARCHIVE)
    print("REPLAY PASS")
    if args.retrain:
        retrain()
    if args.sensitivity:
        sensitivity()
