"""Verify the published hourly decisions, net returns and five reported metrics (stdlib)."""
import csv
import gzip
import hashlib
import json
import math
import statistics
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ACTORS = ("student", "rule", "untrained")
PERIODS = {
    "paper": (2021, 2023, "2021-10-01T00:00:00+00:00", "2023-12-31T23:00:00+00:00", 19727),
    "reused": (2024, 2026, "2024-01-01T00:00:00+00:00", "2026-10-31T23:00:00+00:00", 24840),
}
MISSING_HOUR = datetime.fromisoformat("2023-03-24T13:00:00+00:00")


def metrics(returns):
    equity = peak = 1.0
    drawdown = 0.0
    for value in returns:
        equity *= 1 + value
        peak = max(peak, equity)
        drawdown = min(drawdown, equity / peak - 1)
    n = len(returns)
    numerator = statistics.mean(returns) * math.sqrt(8760)
    downside = math.sqrt(math.fsum(min(r, 0) ** 2 for r in returns) / n)
    return {
        "sortino": numerator / downside,
        "sharpe": numerator / statistics.stdev(returns),
        "cum_return_pct": 100 * (equity - 1),
        "cagr_pct": 100 * (equity ** (8760 / n) - 1),
        "mdd_pct": 100 * drawdown,
    }


def check_period(period, reference):
    year_from, year_to, start, end, expected_count = PERIODS[period]
    expected = datetime.fromisoformat(start)
    previous = dict.fromkeys(ACTORS, 0.0)
    returns = {actor: [] for actor in ACTORS}
    for year in range(year_from, year_to + 1):
        path = ROOT / "hourly" / f"{period}_{year}.csv.gz"
        with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                clock = datetime.fromisoformat(row["utc"])
                if expected == MISSING_HOUR:
                    expected += timedelta(hours=1)
                if clock != expected or clock.utcoffset() != timedelta(0):
                    raise ValueError(f"{path.name}: unexpected timestamp {clock}")
                for actor in ACTORS:
                    weight = float(row[f"decision_{actor}"])
                    net = float(row[f"net_{actor}"])
                    if not 0 <= weight <= 1 or not math.isfinite(net) or net <= -1:
                        raise ValueError(f"Invalid observation at {clock} for {actor}")
                    if clock.hour != 23 and weight != previous[actor]:
                        raise ValueError(f"Unexpected decision change at {clock}")
                    previous[actor] = weight
                    returns[actor].append(net)
                expected += timedelta(hours=1)
    if expected - timedelta(hours=1) != datetime.fromisoformat(end):
        raise ValueError(f"Incomplete end of {period}")
    for actor, observations in returns.items():
        if len(observations) != expected_count:
            raise ValueError(f"Wrong observation count in {period}")
        values = metrics(observations)
        for field, value in values.items():
            if not math.isclose(value, reference[actor][field], rel_tol=0, abs_tol=1e-8):
                raise ValueError(f"{period}/{actor}/{field}: {value} != {reference[actor][field]}")
        print(period, actor, len(observations), *(f"{k}={v:.6f}" for k, v in values.items()))


def main():
    for line in (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        data = (ROOT / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Digest mismatch: {name}")
    reference = json.loads((ROOT / "results/metrics.json").read_text(encoding="utf-8"))
    for period in PERIODS:
        check_period(period, reference[period])
    print("PASS: archive digests, hourly clocks, daily decision times and all 30 metric comparisons")


if __name__ == "__main__":
    main()
