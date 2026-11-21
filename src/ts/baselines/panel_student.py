"""P-04 spot policy trained across assets, with a BTC-only ablation."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ts.baselines.breakout_student import TARGET, daily_inputs, predict, teacher_examples, windows
from ts.baselines.weekly_confirmation import paired_sharpe_interval
from ts.baselines.weekly_student import OUT as PRIOR, decisions_at, digest, train_rows
from ts.config import PROJECT_ROOT
from ts.eval.backtester import CausalPortfolioBacktester
from ts.eval.cost_model import compute_phase2_bar_base_cost_rates
from ts.eval.walk_forward import generate_phase2_walk_forward_folds
from ts.study3.distillation import ReasoningDistillationLoss
from ts.study3.student_model import Student1DCNN

OUT = PROJECT_ROOT / "runs/baselines/panel_student"
ASSETS = ("eth", "bnb", "xrp", "ltc")


def utility_actions(bars, times):
    close = bars.close
    prices = np.column_stack([close.reindex(times + pd.Timedelta(days=n)) for n in range(8)])
    changes = prices[:, 1:] / prices[:, :-1] - 1.
    fees = pd.Series(compute_phase2_bar_base_cost_rates(bars), index=bars.index).loc[times].to_numpy()
    spot = np.array([0., .5, 1.])
    utility = ((changes.sum(axis=1) - 2 * fees)[:, None] * spot
               - 2 * np.square(np.minimum(changes, 0.)).sum(axis=1)[:, None] * spot**2)
    return torch.from_numpy(2 + np.argmax(utility, axis=1)), np.isfinite(utility).all(axis=1)


def sample_asset(bars):
    times, x = windows(daily_inputs(bars)[0])
    y, valid = utility_actions(bars, times)
    return times, x, y, valid


def fit_panel(pre_x, pre_y, btc_x, btc_y, teacher_logits, reasons):
    torch.manual_seed(42)
    model = Student1DCNN(input_dim=btc_x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
    distill = ReasoningDistillationLoss()
    zero = torch.zeros(())
    for features, labels, use_teacher in ((pre_x, pre_y, False), (btc_x, btc_y, True)):
        for _ in range(10):
            for batch in torch.randperm(len(features)).split(64):
                logits, _, projection = model(features[batch])
                loss = F.cross_entropy(logits, labels[batch])
                if use_teacher:
                    loss, *_ = distill(logits, teacher_logits[batch], projection, reasons[batch], loss, zero, zero)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
                optimizer.step()
    return model.eval()


def main():
    torch.set_num_threads(2)
    cache = PROJECT_ROOT / "data/cache"
    base = cache / "btcusdt_1h_20200801_20231231.parquet"
    extra = cache / "p01_sources/confirmed_ohlcv.parquet"
    hashes = json.loads((PRIOR / "metrics.json").read_text())["sha256"]
    if digest(base) != hashes[base.name] or digest(extra) != json.loads((PRIOR / "confirmation.json").read_text())["data_sha256"]:
        raise ValueError("BTC data differ from P-01")
    hist, later = pd.read_parquet(base), pd.read_parquet(extra)
    btc = sample_asset(hist)
    times, btc_x, labels, valid = btc
    # Study 2 policy and text are available only on BTC; auxiliary assets use realized rewards.
    _, logits, reasons = teacher_examples(times, hist, (labels.numpy() - 2) * .5,
                                          PROJECT_ROOT / "runs/study2", hashes)
    all_times, all_x = windows(daily_inputs(pd.concat([hist[later.columns], later]))[0])
    paths = [cache / f"{a}usdt_1h_20200725_20231231.parquet" for a in ASSETS]
    panel = [sample_asset(pd.read_parquet(path)) for path in paths]
    folds = generate_phase2_walk_forward_folds(hist.index)
    if len(folds) != 9:
        raise ValueError("Expected nine phase 2 folds")
    spans = [(f.train_start, f.test_start, f.test_end) for f in folds]
    spans.append((later.index[0] - pd.DateOffset(years=1), later.index[0], later.index[-1]))
    result = {"folds": [], "frozen_model_sha256": {}, "inputs_sha256":
              {p.name: digest(p) for p in [base, extra, *paths, Path(__file__)]}}
    OUT.mkdir(parents=True, exist_ok=True)
    predictions = {"P04": [], "btc_only": []}
    for fold, (start, cutoff, end) in enumerate(spans):
        b = train_rows(times, start, cutoff, valid)
        test = np.flatnonzero((all_times >= cutoff) & (all_times <= end) & (all_times.day_of_week == 6))
        samples_x, samples_y = [btc_x[b]], [labels[b]]
        last = []
        for asset_times, features, actions, keep in panel:
            rows = train_rows(asset_times, start, cutoff, keep)
            samples_x.append(features[rows])
            samples_y.append(actions[rows])
            last.append(asset_times[rows[-1]])
        result["folds"].append({"test_start": str(cutoff), "n_btc": len(b),
                                "n_all_assets": sum(map(len, samples_x)),
                                "last_label": str(max([times[b[-1]], *last]) + pd.Timedelta(days=7))})
        for name, pre_x, pre_y in (("P04", torch.cat(samples_x), torch.cat(samples_y)),
                                   ("btc_only", btc_x[b], labels[b])):
            model = fit_panel(pre_x, pre_y, btc_x[b], labels[b], logits[b], reasons[b])
            predictions[name].append(pd.Series(predict(model, all_x[test]), index=all_times[test]))
            if fold == len(folds):
                path = OUT / f"{name}.pt"
                torch.save(model.state_dict(), path)
                result["frozen_model_sha256"][name] = digest(path)
    paper = hist.iloc[np.concatenate([f.test_indices for f in folds])]
    bt = CausalPortfolioBacktester()
    for stage, bars, prior_file in (("paper", paper, "hourly_returns.parquet"),
                                    ("reused", later, "confirmation_hourly_returns.parquet")):
        prior = pd.read_parquet(PRIOR / prior_file)
        if not prior.index.equals(bars.index):
            raise ValueError("Different reference calendar")
        returns, weights, metrics = {}, {}, {}
        for name, segments in predictions.items():
            series = pd.concat(segments[:-1]) if stage == "paper" else segments[-1]
            dates = bars.index[(bars.index.day_of_week == 6) & (bars.index.hour == 23)]
            if not dates.equals(series.index):
                raise ValueError("Missing weekly actions")
            weights[name] = decisions_at(bars.index, dates, series)
            simulation = bt.run_decisions(bars, weights[name], phase=2)
            returns[name] = simulation.bar_net_returns
            metrics[name] = simulation.summary.to_dict()
        result[stage] = {"results": metrics, "paired_sr_diff_ci_vs_p01":
                         paired_sharpe_interval(returns["P04"], prior["weekly_student"]),
                         "paired_sr_diff_ci_vs_btc_only": paired_sharpe_interval(returns["P04"], returns["btc_only"])}
        pd.DataFrame(returns, index=bars.index).to_parquet(OUT / f"{stage}_returns.parquet")
        pd.DataFrame(weights, index=bars.index).to_parquet(OUT / f"{stage}_decisions.parquet")
        print(stage, pd.DataFrame(metrics).T[list(TARGET)].round(3))
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
