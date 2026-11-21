"""Within-round simultaneous bootstrap bands for the P31-P62 comparisons."""
import json

import numpy as np
import pandas as pd

from ts.baselines.trend_entry_student import OUT


def ratio(sums, count):
    mean, downside = np.split(sums / count, 2, axis=-1)
    return np.sqrt(8760) * np.divide(mean, np.sqrt(downside), out=np.zeros_like(mean), where=downside > 0)


def intervals(book):
    values = book.to_numpy()
    length, width = values.shape
    moments = np.column_stack([values, np.minimum(values, 0.)**2])
    blocks, tail = divmod(length, 168)
    prefix = np.vstack([np.zeros(2 * width), np.cumsum(np.vstack([moments, moments[:168]]), axis=0)])
    start = np.arange(length)
    weeks = prefix[start + 168] - prefix[start]
    partial = prefix[start + tail] - prefix[start]
    rng = np.random.default_rng(42)
    draws = []
    for _ in range(100):
        sums = weeks[rng.integers(length, size=(50, blocks))].sum(axis=1) + partial[rng.integers(length, size=50)]
        draws.append(ratio(sums, length))
    samples = np.vstack(draws)
    estimate = ratio(moments.sum(axis=0), length)
    def band(point, reps):
        se = reps.std(axis=0, ddof=1)
        if np.any(se == 0):
            raise ValueError("Degenerate comparison")
        critical = np.quantile(np.max(np.abs((reps - point) / se), axis=1), .95)
        i = book.columns[1:].get_loc("P61")
        return [float(point[i] - critical * se[i]), float(point[i] + critical * se[i])]
    return {"P61_simultaneous": band(estimate[1:], samples[:, 1:]),
            "P61_minus_P28_simultaneous": band(estimate[1:] - estimate[0], samples[:, 1:] - samples[:, [0]])}


def main():
    summary = {}
    for stage in ("paper", "reused"):
        paths = {f"P{i}": OUT / f"P{i}_seed42" / f"{stage}_returns.parquet" for i in (28, *range(31, 63))}
        series = {name: pd.read_parquet(path).student for name, path in paths.items()}
        if any(not item.index.equals(series["P28"].index) for item in series.values()):
            raise ValueError("Unequal evaluation periods")
        summary[stage] = intervals(pd.DataFrame(series))
    file = OUT / "P61_seed42" / "selection_audit.json"
    file.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
