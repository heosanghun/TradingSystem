"""
Export the 128-dimensional hidden vectors of the trend-and-recovery spot student (research
strategy P-61) for the latent-space figure.

One model is used for every point, so that all points live in the same hidden space:
the student trained, exactly as in `momentum_condition_student.train_and_evaluate`, on the
entry days of 2021 (training window 2021-01-01 .. 2021-12-31, labels ending before
2022-01-01). It then encodes the daily input window of every day from 2022-01-01 on. None
of those days was used to train it.

Check: the same reconstruction applied to the last training window (2023) must give a
checkpoint identical to the archived `runs/baselines/trend_entry/P61_seed42/frozen.pt`.

Output: runs/baselines/trend_entry/P61_seed42/single_model_hidden.npz with
  timestamps_ns  (N,)      decision time (UTC 23:00 bar), nanoseconds
  hidden         (N, 128)  `encode_hidden` output
  weight         (N,)      the position this model would hold if entry were allowed
  entry_allowed  (N,)      P61 entry state on that day
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from ts.baselines.cycle_student import training_indices
from ts.baselines.momentum_condition_student import make_data
from ts.baselines.risk_student import fit
from ts.baselines.trend_entry_student import day_volume, entry_signals
from ts.config import PROJECT_ROOT

ARCHIVE = PROJECT_ROOT / "runs" / "baselines" / "trend_entry" / "P61_seed42"


def train_window(data, allowed, y, start, cutoff, seed):
    t = data["times"]
    start, cutoff = (pd.Timestamp(x) for x in (start, cutoff))
    start, cutoff = (x if x.tzinfo else x.tz_localize("UTC") for x in (start, cutoff))
    rows = training_indices(t, start, cutoff,
                            allowed & data["known"], y, True)
    model = fit(data["x"][rows], data["state"][rows], torch.tensor(y[rows], dtype=torch.float32),
                data["logits"][rows], data["reasons"][rows], seed=seed)
    return model, len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-start", default="2021-01-01")
    parser.add_argument("--train-cutoff", default="2022-01-01")
    parser.add_argument("--encode-from", default="2022-01-01")
    args = parser.parse_args()
    torch.set_num_threads(2)

    data = make_data()
    bars = pd.concat([data["hist"][data["later"].columns], data["later"]])
    full = pd.concat([data["old"], bars])
    allowed = entry_signals(data["close"], day_volume(full)).loc[data["times"]]["P61"].to_numpy()
    y = data["budget_y"]

    # Reconstruction check against the archived final checkpoint (training window 2023).
    later0 = data["later"].index[0]
    final, n_final = train_window(data, allowed, y, later0 - pd.DateOffset(years=1), later0, args.seed)
    archived = torch.load(ARCHIVE / "frozen.pt", map_location="cpu", weights_only=True)
    same = all(torch.equal(archived[k], v) for k, v in final.state_dict().items()) and archived.keys() == final.state_dict().keys()
    print(f"reconstruction check (2023 window, {n_final} rows): identical to frozen.pt = {same}")
    if not same:
        raise SystemExit("training reconstruction differs from the archived model")

    model, n_rows = train_window(data, allowed, y, args.train_start, args.train_cutoff, args.seed)
    t = data["times"]
    keep = np.flatnonzero(t >= pd.Timestamp(args.encode_from, tz="UTC"))
    with torch.inference_mode():
        hidden = model.encode_hidden(data["x"][keep])
        weight = model.weight(hidden, data["state"][keep])
    out = ARCHIVE / "single_model_hidden.npz"
    np.savez_compressed(out, timestamps_ns=t[keep].as_unit("ns").asi8, hidden=hidden.numpy().astype(np.float32),
                        weight=weight.numpy().astype(np.float32), entry_allowed=allowed[keep])
    print(f"model trained on {n_rows} entry days from {args.train_start} to before {args.train_cutoff}; "
          f"encoded {len(keep)} days from {t[keep][0].date()} to {t[keep][-1].date()} -> {out}")


if __name__ == "__main__":
    main()
