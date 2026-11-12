"""Transactional state for resumable teacher runs and weekly edge decisions."""
from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import MutableMapping
from pathlib import Path


class SQLiteJSONMap(MutableMapping):
    """A small JSON mapping with atomic writes and a run-specific namespace."""

    def __init__(self, path: Path, namespace: str):
        self.path = Path(path)
        self.namespace = namespace
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS state (namespace TEXT, key TEXT, value TEXT, "
                         "PRIMARY KEY(namespace,key))")

    def _connect(self):
        return sqlite3.connect(str(self.path), timeout=30)

    @staticmethod
    def _key(key):
        return json.dumps(key, separators=(",", ":"), ensure_ascii=False)

    def __getitem__(self, key):
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM state WHERE namespace=? AND key=?",
                               (self.namespace, self._key(key))).fetchone()
        if row is None:
            raise KeyError(key)
        return json.loads(row[0])

    def __setitem__(self, key, value):
        with self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO state VALUES (?,?,?)",
                         (self.namespace, self._key(key), json.dumps(value, ensure_ascii=False, allow_nan=False)))

    def __delitem__(self, key):
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM state WHERE namespace=? AND key=?",
                                  (self.namespace, self._key(key)))
            if not cursor.rowcount:
                raise KeyError(key)

    def __iter__(self):
        with self._connect() as conn:
            rows = conn.execute("SELECT key FROM state WHERE namespace=? ORDER BY key", (self.namespace,)).fetchall()
        for row in rows:
            value = json.loads(row[0])
            yield tuple(value) if isinstance(value, list) else value

    def __len__(self):
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM state WHERE namespace=?", (self.namespace,)).fetchone()[0]


def run_checkpointed_bar(store, key, engine, debate_kwargs, counterfactual=False):
    """Commit a whole bar or an explicit failure; replay a saved bar without calls.

    Interrupted bars replay their already cached responses on restart. Failed bars
    remain failed until explicitly removed, avoiding silent changes in coverage.
    """
    from dataclasses import asdict
    from ts.study2.llm_client import LLMCallError

    if key in store:
        return store[key]
    before = asdict(engine.verifier.stats)
    try:
        outcome = engine.run_bar_debate(**debate_kwargs, strip_evidence=False)
        cf = engine.run_bar_debate(**debate_kwargs, strip_evidence=True) if counterfactual else None
        result = {"status": "ok", "outcome": asdict(outcome), "counterfactual": asdict(cf) if cf else None}
    except LLMCallError as exc:
        logging.getLogger(__name__).error("Teacher bar %s failed: %s", key, exc)
        result = {"status": "failed", "error": str(exc)}
    after = asdict(engine.verifier.stats)
    result["stats_delta"] = {k: after[k] - before[k] for k in before if isinstance(before[k], int)}
    result["rejection_reasons"] = after["rejection_reasons"][len(before["rejection_reasons"]):]
    store[key] = result
    return result
