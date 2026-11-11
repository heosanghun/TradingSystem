"""
Provenance Number Ledger (`ledger/numbers.csv`).

Every empirical figure cited in the thesis (Tables 6.2–6.15, Abstract, Ch. I,
Ch. III–VII) maps to a row in `ledger/numbers.csv` recording:
  number_id, table_ref, metric_name, value, formatted_value, unit,
  script_path, config_sha256, git_commit, run_log_path, recorded_at_utc
"""

from __future__ import annotations

import csv
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

from ts.config import PROJECT_ROOT


DEFAULT_LEDGER_PATH = PROJECT_ROOT / "ledger" / "numbers.csv"

LEDGER_COLUMNS = (
    "number_id",
    "table_ref",
    "metric_name",
    "value",
    "formatted_value",
    "unit",
    "script_path",
    "config_sha256",
    "git_commit",
    "run_log_path",
    "recorded_at_utc",
)


def get_git_commit_short(cwd: Path = PROJECT_ROOT) -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return "uncommitted"


@dataclass
class LedgerEntry:
    number_id: str
    table_ref: str
    metric_name: str
    value: float
    formatted_value: str
    unit: str
    script_path: str
    config_sha256: str
    git_commit: str
    run_log_path: str
    recorded_at_utc: str = ""

    def __post_init__(self) -> None:
        if not self.recorded_at_utc:
            self.recorded_at_utc = datetime.now(timezone.utc).isoformat()


class NumberLedger:
    """Upsert-capable provenance ledger backed by `ledger/numbers.csv`."""

    def __init__(self, ledger_path: Union[str, Path] = DEFAULT_LEDGER_PATH) -> None:
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

    def read_all(self) -> Dict[str, Dict[str, str]]:
        if not self.ledger_path.exists():
            return {}
        rows: Dict[str, Dict[str, str]] = {}
        with self.ledger_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                nid = row.get("number_id", "")
                if nid:
                    rows[nid] = dict(row)
        return rows

    def upsert(self, entries: Iterable[LedgerEntry]) -> int:
        existing = self.read_all()
        count = 0
        for entry in entries:
            d = asdict(entry)
            existing[entry.number_id] = {k: str(d[k]) for k in LEDGER_COLUMNS}
            count += 1

        sorted_rows: List[Dict[str, str]] = [
            existing[k] for k in sorted(existing.keys())
        ]
        with self.ledger_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(LEDGER_COLUMNS))
            writer.writeheader()
            writer.writerows(sorted_rows)
        return count

    def record(
        self,
        number_id: str,
        table_ref: str,
        metric_name: str,
        value: float,
        fmt: str = "{:.4f}",
        unit: str = "",
        script_path: str = "",
        config_sha256: str = "",
        run_log_path: str = "",
        git_commit: Optional[str] = None,
    ) -> LedgerEntry:
        entry = LedgerEntry(
            number_id=number_id,
            table_ref=table_ref,
            metric_name=metric_name,
            value=float(value),
            formatted_value=fmt.format(float(value)),
            unit=unit,
            script_path=script_path,
            config_sha256=config_sha256,
            git_commit=git_commit or get_git_commit_short(),
            run_log_path=run_log_path,
        )
        self.upsert([entry])
        return entry
