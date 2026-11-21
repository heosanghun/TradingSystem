"""P-06 continuous spot student of the historical P-05 breakout teacher."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from ts.baselines.breakout_student import TARGET, daily_inputs, windows, teacher_examples
from ts.baselines.early_breakout_audit import policy_inputs
from ts.baselines.weekly_confirmation import paired_sharpe_interval
from ts.baselines.weekly_student import OUT as PRIOR, decisions_at, digest
from ts.config import PROJECT_ROOT
from ts.eval.backtester import CausalPortfolioBacktester
from ts.eval.walk_forward import generate_phase2_walk_forward_folds
from ts.study3.distillation import ReasoningDistillationLoss
from ts.study3.student_model import Student1DCNN

OUT = PROJECT_ROOT / "runs/baselines/risk_student"


class RiskStudent(Student1DCNN):
    def __init__(self, input_dim):
        super().__init__(input_dim=input_dim)
        self.position = nn.Sequential(nn.Linear(131, 16), nn.ReLU(), nn.Linear(16, 1))
        nn.init.normal_(self.position[-1].weight, std=.01)
        nn.init.constant_(self.position[-1].bias, .25)

    def weight(self, hidden, today):
        return self.position(torch.cat([hidden, today], dim=1)).squeeze(-1).clamp(0., 1.)


def fit(x, current, target, logits, reasons, seed=42):
    torch.manual_seed(seed)
    model = RiskStudent(x.shape[1])
    optim = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
    loss_fn = ReasoningDistillationLoss()
    zero = torch.zeros(())
    for _ in range(50):
        for rows in torch.randperm(len(x)).split(64):
            h = model.encode_hidden(x[rows])
            position = model.weight(h, current[rows])
            tracking = F.smooth_l1_loss(position, target[rows], beta=.1)
            total, *_ = loss_fn(model.actor(h), logits[rows], model.projection_head(h),
                                reasons[rows], tracking, zero, zero)
            optim.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
            optim.step()
    return model.eval()


def main():
    torch.set_num_threads(2)
    gate = json.loads((PROJECT_ROOT / "runs/baselines/early_breakout/metrics.json").read_text())
    if not gate["pass_all"]:
        raise ValueError("The historical teacher gate did not pass")
    base = PROJECT_ROOT / "data/cache/btcusdt_1h_20200801_20231231.parquet"
    extra = PROJECT_ROOT / "data/cache/p01_sources/confirmed_ohlcv.parquet"
    hashes = json.loads((PRIOR / "metrics.json").read_text())["sha256"]
    if digest(base) != hashes[base.name] or digest(extra) != json.loads((PRIOR / "confirmation.json").read_text())["data_sha256"]:
        raise ValueError("Unexpected BTC inputs")
    hist, later = pd.read_parquet(base), pd.read_parquet(extra)
    prices = pd.concat([hist[later.columns], later])
    times, x = windows(daily_inputs(prices)[0])
    states = policy_inputs(prices).loc[times]
    current = torch.tensor(states.to_numpy(), dtype=torch.float32)
    target = (states.fast + states.slow) * states.budget / 2
    y = torch.tensor(target.to_numpy(), dtype=torch.float32)
    known = times <= hist.index[-1]
    _, logits, reasons = teacher_examples(times[known], hist, target.to_numpy()[known],
                                          PROJECT_ROOT / "runs/study2", hashes)
    folds = generate_phase2_walk_forward_folds(hist.index)
    if len(folds) != 9:
        raise ValueError("Incomplete nine-fold experiment")
    spans = [(f.train_start, f.test_start, f.test_end) for f in folds]
    spans.append((later.index[0] - pd.DateOffset(years=1), later.index[0], later.index[-1]))
    OUT.mkdir(parents=True, exist_ok=True)
    predictions, audit = [], []
    for i, (start, cutoff, end) in enumerate(spans):
        train = np.flatnonzero((times >= start) & (times < cutoff) & known)
        test = np.flatnonzero((times >= cutoff) & (times <= end))
        model = fit(x[train], current[train], y[train], logits[train], reasons[train])
        with torch.inference_mode():
            weight = model.weight(model.encode_hidden(x[test]), current[test]).numpy()
        predictions.append(pd.Series(weight, index=times[test]))
        audit.append({"test_from": str(cutoff), "n_train": len(train), "last_train": str(times[train[-1]])})
        if i == len(folds):
            torch.save(model.state_dict(), OUT / "frozen_student.pt")
    paper = hist.iloc[np.concatenate([f.test_indices for f in folds])]
    results = {"folds": audit, "frozen_model_sha256": digest(OUT / "frozen_student.pt"),
               "inputs_sha256": {p.name: digest(p) for p in (base, extra, Path(__file__))}}
    for stage, bars, daily, prior_file in (("paper", paper, pd.concat(predictions[:-1]), "hourly_returns.parquet"),
                                          ("reused", later, predictions[-1], "confirmation_hourly_returns.parquet")):
        if not daily.index.equals(bars.index[bars.index.hour == 23]):
            raise ValueError("Missing daily predictions")
        weights = {"P05": decisions_at(bars.index, daily.index, target.loc[daily.index]),
                   "P06": decisions_at(bars.index, daily.index, daily)}
        metrics, returns = {}, {}
        for name, w in weights.items():
            res = CausalPortfolioBacktester().run_decisions(bars, w, phase=2)
            metrics[name] = res.summary.to_dict()
            returns[name] = res.bar_net_returns
        prior = pd.read_parquet(PRIOR / prior_file)["weekly_student"]
        if not prior.index.equals(bars.index):
            raise ValueError("P-01 dates differ")
        results[stage] = {"results": metrics, "daily_weight_mae": float((daily-target.loc[daily.index]).abs().mean()),
                          "paired_sr_diff_ci_vs_p01": paired_sharpe_interval(returns["P06"], prior),
                          "paired_sr_diff_ci_vs_teacher": paired_sharpe_interval(returns["P06"], returns["P05"])}
        pd.DataFrame(returns, index=bars.index).to_parquet(OUT / f"{stage}_returns.parquet")
        pd.DataFrame(weights, index=bars.index).to_parquet(OUT / f"{stage}_decisions.parquet")
        print(stage, pd.DataFrame(metrics).T[list(TARGET)].round(3))
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
