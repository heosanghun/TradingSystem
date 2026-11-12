"""
Provenance ledger for thesis figures.

Every figure cited in the thesis maps to one row holding the value, the script and config
that produced it, the git commit, the run log, and the run mode.

Two files are kept side by side:
- `ledger/numbers.csv`: final measurements. Only this file feeds the thesis tables.
- `ledger/provisional.csv`: anything recorded without a `RunContext`, from a smoke or
  simulated run, or from an uncommitted working tree. Each row says why it is not final.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
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
    "run_stage",
    "run_flags",
    "provisional_reasons",
)

# A run is final only if none of these flag values is present.
_FORBIDDEN_FLAG_VALUES: Dict[str, tuple] = {
    "llm_simulated": (True,),
    "finbert_offline_hash": (True,),
    "quick_verify": (True,),
    "reduced_training_budget": (True,),
    "teacher_signals_present": (False,),
    "lob_input_mode": ("causal_hourly_microstructure_fallback", "synthetic"),
}

# Flags each stage must declare. An undeclared flag counts as a violation, so a script that
# has not been updated to describe its run cannot write final numbers by accident.
_REQUIRED_FLAGS: Dict[str, tuple] = {
    "wp0": (),
    "study1": ("finbert_offline_hash", "quick_verify", "reduced_training_budget"),
    "study2": ("llm_simulated", "quick_verify"),
    "study3": (
        "llm_simulated",
        "teacher_signals_present",
        "lob_input_mode",
        "finbert_offline_hash",
        "quick_verify",
        "reduced_training_budget",
    ),
    "baselines": ("quick_verify", "reduced_training_budget"),
}

SMOKE_ENV_VAR = "TS_SMOKE_RUN"


@dataclass
class RunContext:
    """
    Describes the run that produced a number.

    `stage` is one of `wp0`, `study1`, `study2`, `study3`, `baselines`.
    `flags` holds the run-mode flags listed in `_REQUIRED_FLAGS` for that stage.
    `enforce_git=False` skips the repository checks and is meant for unit tests.
    """

    stage: str
    flags: Dict[str, object] = field(default_factory=dict)
    enforce_git: bool = True

    def violations(self) -> List[str]:
        out: List[str] = []
        if self.stage not in _REQUIRED_FLAGS:
            out.append(f"unknown stage '{self.stage}'")
        for name in _REQUIRED_FLAGS.get(self.stage, ()):
            if name not in self.flags:
                out.append(f"flag '{name}' not declared")
        for name, bad_values in _FORBIDDEN_FLAG_VALUES.items():
            if name in self.flags and self.flags[name] in bad_values:
                out.append(f"{name}={self.flags[name]}")
        if os.environ.get(SMOKE_ENV_VAR, "").strip() not in ("", "0"):
            out.append(f"{SMOKE_ENV_VAR} is set")
        return out


@dataclass(frozen=True)
class GitState:
    commit: str
    in_project_repo: bool
    dirty: bool

    @property
    def label(self) -> str:
        if not self.in_project_repo:
            return "no-project-repo"
        return f"{self.commit}-dirty" if self.dirty else self.commit


def _files_this_run_depends_on(cwd: Path) -> set:
    """
    Repository-relative paths of the code this process has actually loaded: every imported
    module that lives inside the project, plus the entry script. A measured number can only
    depend on these files and on the configuration.
    """
    root = Path(cwd).resolve()
    paths = set()
    candidates = [getattr(mod, "__file__", None) for mod in list(sys.modules.values())]
    candidates.append(sys.argv[0] if sys.argv else None)
    for raw in candidates:
        if not raw:
            continue
        try:
            rel = Path(raw).resolve().relative_to(root)
        except (ValueError, OSError):
            continue
        rel_posix = rel.as_posix()
        if rel_posix.startswith((".venv/", "tests/")):
            continue
        paths.add(rel_posix)
    return paths


def read_git_state(cwd: Path = PROJECT_ROOT) -> GitState:
    """
    State of the repository rooted exactly at `cwd`.

    A parent repository that merely contains the project folder does not count: its commit
    would not identify this code. The run counts as dirty when a file it depends on differs
    from the commit: a loaded project module or the entry script (modified or untracked),
    anything under `configs/`, or `pyproject.toml`. Edits to files this process never loaded
    (another stage's module, documents, tests, the ledger) do not make it dirty.
    """

    def _git(*args: str) -> Optional[str]:
        try:
            res = subprocess.run(
                ["git", "-c", "core.quotepath=off", *args],
                cwd=str(cwd),
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=10,
                check=False,
            )
        except Exception:
            return None
        return res.stdout if res.returncode == 0 else None

    top = _git("rev-parse", "--show-toplevel")
    if top is None or Path(top.strip()).resolve() != Path(cwd).resolve():
        return GitState(commit="", in_project_repo=False, dirty=True)

    commit = (_git("rev-parse", "--short", "HEAD") or "").strip()
    status = _git("status", "--porcelain", "--untracked-files=all") or ""
    changed = {line[3:].strip().strip('"').replace("\\", "/") for line in status.splitlines() if line.strip()}
    loaded = _files_this_run_depends_on(cwd)
    dirty = any(
        path in loaded or path.startswith("configs/") or path == "pyproject.toml"
        for path in changed
    )
    return GitState(commit=commit, in_project_repo=bool(commit), dirty=dirty)


def get_git_commit_short(cwd: Path = PROJECT_ROOT) -> str:
    return read_git_state(cwd).label


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
    run_stage: str = ""
    run_flags: str = ""
    provisional_reasons: str = ""

    def __post_init__(self) -> None:
        if not self.recorded_at_utc:
            self.recorded_at_utc = datetime.now(timezone.utc).isoformat()


class NumberLedger:
    """Upsert-capable ledger that routes non-final records to `provisional.csv`."""

    def __init__(
        self,
        ledger_path: Union[str, Path] = DEFAULT_LEDGER_PATH,
        run_context: Optional[RunContext] = None,
    ) -> None:
        self.ledger_path = Path(ledger_path)
        self.provisional_path = self.ledger_path.with_name("provisional.csv")
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_context = run_context
        self._git_state: Optional[GitState] = None

    def _git(self) -> GitState:
        if self._git_state is None:
            self._git_state = read_git_state()
        return self._git_state

    def reasons_not_final(self) -> List[str]:
        """Empty list means records from this ledger instance are final."""
        ctx = self.run_context
        if ctx is None:
            return ["no RunContext given"]
        reasons = ctx.violations()
        if ctx.enforce_git:
            state = self._git()
            if not state.in_project_repo:
                reasons.append("project folder is not its own git repository")
            elif state.dirty:
                reasons.append("uncommitted changes in working tree")
        return reasons

    def read_all(self) -> Dict[str, Dict[str, str]]:
        """Final rows only."""
        return self._read(self.ledger_path)

    def read_provisional(self) -> Dict[str, Dict[str, str]]:
        return self._read(self.provisional_path)

    @staticmethod
    def _read(path: Path) -> Dict[str, Dict[str, str]]:
        if not path.exists():
            return {}
        rows: Dict[str, Dict[str, str]] = {}
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                nid = row.get("number_id", "")
                if nid:
                    rows[nid] = dict(row)
        return rows

    def upsert(self, entries: Iterable[LedgerEntry], provisional: bool = False) -> int:
        path = self.provisional_path if provisional else self.ledger_path
        existing = self._read(path)
        count = 0
        for entry in entries:
            d = asdict(entry)
            existing[entry.number_id] = {k: str(d[k]) for k in LEDGER_COLUMNS}
            count += 1

        sorted_rows: List[Dict[str, str]] = []
        for key in sorted(existing.keys()):
            row = existing[key]
            sorted_rows.append({col: row.get(col, "") or "" for col in LEDGER_COLUMNS})

        with path.open("w", encoding="utf-8", newline="") as f:
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
        reasons = self.reasons_not_final()
        ctx = self.run_context
        if git_commit is None:
            git_commit = self._git().label

        entry = LedgerEntry(
            number_id=number_id,
            table_ref=table_ref,
            metric_name=metric_name,
            value=float(value),
            formatted_value=fmt.format(float(value)),
            unit=unit,
            script_path=script_path,
            config_sha256=config_sha256,
            git_commit=git_commit,
            run_log_path=run_log_path,
            run_stage=ctx.stage if ctx is not None else "",
            run_flags=json.dumps(ctx.flags, sort_keys=True, ensure_ascii=False) if ctx is not None else "",
            provisional_reasons="; ".join(reasons),
        )
        self.upsert([entry], provisional=bool(reasons))
        return entry
