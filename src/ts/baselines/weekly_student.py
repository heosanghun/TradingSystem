"""Research candidate P-01. Run with python -m ts.baselines.weekly_student.

Daily supervised samples, seven-day labels, weekly decisions, hourly accounting.
All dates are UTC candle open times. The last feature is known at candle close.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ts.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, SystemConfig
from ts.eval.backtester import CausalPortfolioBacktester
from ts.eval.cost_model import compute_phase2_bar_base_cost_rates
from ts.eval.metrics import compute_performance_summary
from ts.eval.walk_forward import generate_phase2_walk_forward_folds
from ts.study1.ppo_expert import ACTION_WEIGHTS
from ts.study3.distillation import ReasoningDistillationLoss
from ts.study3.student_model import Student1DCNN
from ts.study3.teacher_signal import FinBERTReasoningEncoder, compute_teacher_soft_logits_and_probs

WEEK = pd.Timedelta(days=7)
OUT = PROJECT_ROOT / "runs/baselines/weekly_student"


def weekly_inputs(bars):
    """Calendar-aligned features and a risk estimate available at the decision."""
    if not bars.index.is_monotonic_increasing or not bars.index.is_unique or str(bars.index.tz) != "UTC":
        raise ValueError("Expected unique ascending UTC candles")
    close = bars.close
    hourly = np.log(close).diff()
    sigma = hourly.rolling("30D", min_periods=24 * 21).std().clip(lower=1e-6)
    features = {}
    for days in (1, 7, 30, 90):
        lag = close.shift(freq=pd.Timedelta(days=days)).reindex(close.index)
        features[f"momentum_{days}d"] = np.log(close / lag) / (sigma * np.sqrt(24 * days))
    features["range"] = np.log(bars.high / bars.low) / sigma
    features["location"] = (2 * close - bars.high - bars.low) / (bars.high - bars.low).replace(0, np.nan)
    log_vol = np.log1p(bars.volume)
    features["volume"] = (log_vol - log_vol.rolling("7D", min_periods=72).mean()) / log_vol.rolling("7D", min_periods=72).std().clip(lower=1e-6)
    features["volatility_ratio"] = hourly.rolling("1D", min_periods=18).std() / sigma
    return pd.DataFrame(features, index=bars.index).clip(-8, 8), sigma


def decision_windows(features):
    daily = np.flatnonzero(features.index.hour == 23)
    daily = daily[daily >= 23]
    offsets = np.arange(-23, 1)
    rows = daily[:, None] + offsets
    windows = features.to_numpy(dtype=np.float32)[rows].transpose(0, 2, 1)
    valid = np.isfinite(windows).all(axis=(1, 2))
    valid &= (features.index[daily] - features.index[daily - 23]) == pd.Timedelta(hours=23)
    return daily[valid], np.ascontiguousarray(windows[valid])


def train_rows(times, start, test_start, valid_labels):
    return np.flatnonzero((times >= start) & (times + WEEK < test_start) & valid_labels)


def fit_student(x, labels, logits, reasons):
    torch.manual_seed(42)
    model = Student1DCNN(input_dim=x.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0001)
    loss_fn = ReasoningDistillationLoss()
    zero = torch.zeros(())
    for _ in range(50):
        for batch in torch.randperm(len(x)).split(64):
            z, _, projection = model(x[batch])
            task = F.cross_entropy(z, labels[batch])
            loss, *_ = loss_fn(z, logits[batch], projection, reasons[batch], task, zero, zero)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
    return model.eval()


def decisions_at(index, times, values):
    return pd.Series(values, index=times).reindex(index).ffill().fillna(0).to_numpy()


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    torch.set_num_threads(2)
    cfg = SystemConfig.from_json(DEFAULT_CONFIG_PATH)
    archive = PROJECT_ROOT / "data/cache/btcusdt_1h_20200801_20231231.parquet"
    source = PROJECT_ROOT / "runs/study2"
    prices = pd.read_parquet(archive)
    archive_index = prices.index
    features, sigma = weekly_inputs(prices)
    day, windows = decision_windows(features)
    time = archive_index[day]
    teacher_path = source / "teacher_signals.npz"
    texts_path = source / "consensus_texts.json"
    meta = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
    if (meta.get("status") != "complete" or meta.get("n_failed_bars") != 0
            or meta.get("is_simulated_llm") is not False or meta.get("config_sha256") != cfg.config_sha256):
        raise ValueError("Incomplete, simulated or incompatible teacher run")
    with np.load(teacher_path) as signal:
        key = pd.Index(signal["bar_indices"])
        teacher_pos = key.get_indexer(day)
        if np.any(teacher_pos < 0):
            raise ValueError("Missing teacher observations")
        teacher_logits, _ = compute_teacher_soft_logits_and_probs(signal["w_finals"][teacher_pos])
    texts = json.loads(texts_path.read_text(encoding="utf-8"))
    if len(texts) != len(key):
        raise ValueError("Teacher text/decision length mismatch")
    # FinBERT is frozen. Identical texts need only one forward pass.
    codes, unique = pd.factorize(np.asarray([texts[i] for i in teacher_pos], dtype=object))
    embedding = FinBERTReasoningEncoder().encode_texts(unique.tolist())[codes]
    close_now = prices.close.iloc[day].to_numpy()
    close_later = prices.close.reindex(time + WEEK).to_numpy()
    move = close_later / close_now - 1
    cost = compute_phase2_bar_base_cost_rates(prices)[day]
    scale = sigma.iloc[day].to_numpy() * np.sqrt(168)
    target = np.sign(move) * np.maximum(np.abs(move) - 2 * cost, 0) / scale
    label = np.abs(target[:, None].clip(-1, 1) - ACTION_WEIGHTS).argmin(axis=1)
    tensors = [torch.from_numpy(np.ascontiguousarray(a)) for a in (windows, label.astype(np.int64), teacher_logits, embedding)]

    folds = generate_phase2_walk_forward_folds(archive_index)
    if len(folds) != 9:
        raise ValueError("Incomplete development window")
    pred_times, predicted, fold_info = [], [], []
    for fold in folds:
        rows = train_rows(time, fold.train_start, fold.test_start, np.isfinite(target))
        weekly = np.flatnonzero((time >= fold.test_start) & (time <= fold.test_end) & (time.day_of_week == 6))
        model = fit_student(*(v[rows] for v in tensors))
        with torch.inference_mode():
            actions, _ = model.act_online(tensors[0][weekly])
        pred_times.extend(time[weekly])
        predicted.extend(ACTION_WEIGHTS[actions.numpy()])
        fold_info.append({"fold": fold.fold_id, "train_examples": len(rows),
                          "last_train_label_bar": str(time[rows[-1]] + WEEK),
                          "test_start": str(fold.test_start), "decisions": len(weekly)})
    idx = np.concatenate([fold.test_indices for fold in folds])
    bars = prices.iloc[idx]
    old_path = PROJECT_ROOT / "runs/study3/student_oos_predictions.npz"
    with np.load(old_path) as old:
        np.testing.assert_array_equal(old["oos_bar_indices"], idx)
        original = old["student_weights"]
        saved_returns = old["phase2_net_returns"]
    weekly_times = pd.DatetimeIndex(pred_times)
    # Fixed reference always uses all observed Sunday decision bars.
    sundays = bars.index[(bars.index.day_of_week == 6) & (bars.index.hour == 23)]
    month_back = prices.close.reindex(sundays - pd.Timedelta(days=30)).to_numpy()
    if not np.isfinite(month_back).all():
        raise ValueError("Missing reference lookback price")
    decisions = {
        "weekly_student": decisions_at(bars.index, weekly_times, predicted),
        "original_student": original, "always_long": np.ones(len(idx)),
        "weekly_30d_trend": decisions_at(bars.index, sundays, (prices.close.loc[sundays].to_numpy() > month_back).astype(float)),
    }
    bt = CausalPortfolioBacktester()
    results, series = {}, {}
    for name, weights in decisions.items():
        result = bt.run_decisions(bars, weights, phase=2)
        series[name] = result.bar_net_returns
        results[name] = {**result.summary.to_dict(), "cash_fraction": float(np.mean(result.weights == 0)),
                         "position_changes": int(np.count_nonzero(np.diff(result.weights, prepend=0)))}
        for fold, info in zip(folds, fold_info):
            sub = (bars.index >= fold.test_start) & (bars.index <= fold.test_end)
            r = result.bar_net_returns[sub]
            eq = np.r_[1., np.cumprod(1 + r)]
            info[name + "_sharpe"] = compute_performance_summary(eq, r, result.weights[sub]).sharpe
    np.testing.assert_allclose(series["original_student"], saved_returns, atol=1e-12, rtol=0)
    primary = results["weekly_student"]
    gate = primary["mdd_pct"] >= -24 and all(primary["sharpe"] > results[name]["sharpe"] for name in ("original_student", "always_long"))
    OUT.mkdir(parents=True, exist_ok=True)
    output = {"stage": "development_replay", "eligible_for_confirmation": bool(gate),
              "start": str(bars.index[0]), "end": str(bars.index[-1]), "config_sha256": cfg.config_sha256,
              "sha256": {p.name: digest(p) for p in [Path(__file__), archive, old_path, teacher_path, texts_path]},
              "folds": fold_info, "strategies": results}
    pd.DataFrame(series, index=bars.index).to_parquet(OUT / "hourly_returns.parquet")
    pd.DataFrame(decisions, index=bars.index).to_parquet(OUT / "decisions.parquet")
    if gate:
        cutoff = pd.Timestamp("2024-01-01", tz="UTC")
        final_rows = train_rows(time, cutoff - pd.DateOffset(years=1), cutoff, np.isfinite(target))
        final_model = fit_student(*(v[final_rows] for v in tensors))
        torch.save(final_model.state_dict(), OUT / "frozen_student.pt")
        output["frozen_model"] = {"sha256": digest(OUT / "frozen_student.pt"),
                                  "training_examples": len(final_rows),
                                  "last_label_bar": str(time[final_rows[-1]] + WEEK)}
    (OUT / "metrics.json").write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
