"""Distillation ablation for P-61 (diagnostic; outputs go to a temp folder, not the project).

Same entry days, same labels, same risk budget, same seed and the same training loop as
risk_student.fit. Only the teacher terms of the loss change:
  full        0.4 tracking + 0.4 policy KL + 0.2 reasoning cosine   (P-61 as archived)
  econ_only   0.4 tracking
  no_reason   0.4 tracking + 0.4 policy KL
  shuffled    full weights, but teacher logits and reasoning vectors permuted across days
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

import ts.baselines.momentum_condition_student as mcs
from ts.baselines.momentum_condition_student import make_data, train_and_evaluate
from ts.baselines.risk_student import RiskStudent
from ts.baselines.trend_entry_student import entry_signals, day_volume
from ts.config import PROJECT_ROOT
from ts.study3.distillation import ReasoningDistillationLoss

WEIGHTS = {"full": (.4, .4, .2), "econ_only": (.4, 0., 0.), "no_reason": (.4, .4, 0.), "shuffled": (.4, .4, .2)}


def make_fit(wt, wp, wr):
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
                _, tsk, pol, rsn = loss_fn(model.actor(h), logits[rows], model.projection_head(h),
                                           reasons[rows], tracking, zero, zero)
                total = wt * tsk + wp * pol + wr * rsn
                optim.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
                optim.step()
        return model.eval()
    return fit


def main():
    seeds = [int(s) for s in sys.argv[1:]] or [42]
    torch.set_num_threads(2)
    data = make_data()
    bars = pd.concat([data["hist"][data["later"].columns], data["later"]])
    full = pd.concat([data["old"], bars])
    signals = entry_signals(data["close"], day_volume(full)).loc[data["times"]]
    active = signals["P61"].to_numpy()
    y, budget = data["budget_y"], data["budget"]
    out_root = Path(tempfile.mkdtemp(prefix="distillation_ablation_"))
    rows = []
    for seed in seeds:
        for name, (wt, wp, wr) in WEIGHTS.items():
            d = dict(data)
            if name == "shuffled":
                perm = torch.from_numpy(np.random.default_rng(1000 + seed).permutation(len(data["logits"])))
                d["logits"], d["reasons"] = data["logits"][perm], data["reasons"][perm]
            mcs.fit = make_fit(wt, wp, wr)
            out = out_root / f"{name}_seed{seed}"
            train_and_evaluate(d, active, y, budget.where(signals["P61"], 0.), seed, out, {"variant": name})
            m = json.loads((out / "metrics.json").read_text())
            for stage in ("paper", "reused"):
                s = m[stage]["student"]
                rows.append(dict(seed=seed, variant=name, stage=stage, sortino=s["sortino"], sharpe=s["sharpe"],
                                 cum=s["cum_return_pct"], cagr=s["cagr_pct"], mdd=s["mdd_pct"]))
    df = pd.DataFrame(rows)
    df.to_csv(out_root / "ablation.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))
    print(df.groupby(["stage", "variant"])[["sortino", "sharpe", "cum", "mdd"]].mean().round(3))
    print("OUT", out_root)


if __name__ == "__main__":
    main()
