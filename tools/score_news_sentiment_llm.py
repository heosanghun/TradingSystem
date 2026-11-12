"""
Score every news headline with the thesis's sentiment classifier (Chapter III Sec. 1.4,
Table A.2: DeepSeek-R1-Distill-Qwen-32B) and store the three class logits.

Method. The model is asked to answer with one word (Negative, Neutral or Positive). The
server returns the log-probabilities of the most likely first answer tokens, and the
log-probabilities of the three class words are taken as the logits `z` of Eq. (III.7).
A softmax over three log-probabilities gives the same class probabilities as a softmax
over the underlying logits, so no rescaling is involved. The model's reasoning block is
closed in advance (see `ts.study2.llm_client.EMPTY_THINK_PREFILL`), temperature is 0.

If a class word is not among the returned top tokens, its log-probability is not known.
It is then set to the smallest log-probability that was returned, which is an upper bound
for the missing one. The number of such cases is written to the summary.

Input : data/news_headlines.parquet (published_at, headline, source, ...)
Output: the same file with `logit_neg`, `logit_neu`, `logit_pos` added. The first run keeps
        the unscored original as data/news_headlines_unscored.parquet.
        Progress is kept in data/cache/news_sentiment_llm.sqlite, so the run can be resumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ts.config import PROJECT_ROOT
from ts.study2.llm_client import EMPTY_THINK_PREFILL

SYSTEM_PROMPT = (
    "You are a financial sentiment classifier. Given a financial news headline, answer with "
    "exactly one word: Negative, Neutral, or Positive."
)
CLASS_WORDS = ("Negative", "Neutral", "Positive")
TOP_LOGPROBS = 20


def score_headline(session: requests.Session, endpoint: str, model: str, headline: str, num_ctx: int):
    body = session.post(endpoint, json={
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Headline: " + headline},
            {"role": "assistant", "content": EMPTY_THINK_PREFILL},
        ],
        "stream": False, "think": False, "logprobs": True, "top_logprobs": TOP_LOGPROBS,
        "options": {"temperature": 0, "num_predict": 1, "num_ctx": num_ctx},
    }, timeout=120).json()
    top = {t["token"]: float(t["logprob"]) for t in body["logprobs"][0]["top_logprobs"]}
    floor = min(top.values())
    logits = [top.get(word, floor) for word in CLASS_WORDS]
    n_missing = sum(word not in top for word in CLASS_WORDS)
    return logits, n_missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Score news headlines with the local LLM sentiment classifier.")
    parser.add_argument("--news-file", type=str, default=str(PROJECT_ROOT / "data" / "news_headlines.parquet"))
    parser.add_argument("--endpoint", type=str, default="http://127.0.0.1:11434/api/chat")
    parser.add_argument("--model", type=str, default="deepseek-r1:32b")
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=0, help="Score only the first N headlines (speed test).")
    args = parser.parse_args()

    news_path = Path(args.news_file)
    df = pd.read_parquet(news_path)
    backup = news_path.with_name("news_headlines_unscored.parquet")
    if not backup.exists():
        df.drop(columns=[c for c in ("logit_neg", "logit_neu", "logit_pos") if c in df.columns]).to_parquet(backup)

    db_path = PROJECT_ROOT / "data" / "cache" / "news_sentiment_llm.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(db_path))
    db.execute("CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, neg REAL, neu REAL, pos REAL, missing INTEGER)")
    run_tag = hashlib.sha256(json.dumps([args.model, SYSTEM_PROMPT, TOP_LOGPROBS]).encode()).hexdigest()[:12]

    headlines = df["headline"].astype(str).tolist()
    if args.limit:
        headlines = headlines[: args.limit]
    keys = [run_tag + ":" + hashlib.sha256(h.encode("utf-8")).hexdigest() for h in headlines]
    done = {row[0] for row in db.execute("SELECT key FROM scores WHERE key LIKE ?", (run_tag + ":%",))}

    session = requests.Session()
    t0 = time.time()
    n_new = 0
    for i, (key, headline) in enumerate(zip(keys, headlines)):
        if key in done:
            continue
        logits, n_missing = score_headline(session, args.endpoint, args.model, headline, args.num_ctx)
        db.execute("INSERT OR REPLACE INTO scores VALUES (?,?,?,?,?)", (key, *logits, n_missing))
        done.add(key)
        n_new += 1
        if n_new % 200 == 0:
            db.commit()
        if n_new % 2000 == 0:
            rate = (time.time() - t0) / n_new
            left = sum(1 for k in keys[i + 1:] if k not in done)
            print(f"[sentiment] {i + 1}/{len(keys)} scored, {rate:.2f}s each, about {left * rate / 60:.0f} min left", flush=True)
    db.commit()

    table = {row[0]: row[1:] for row in db.execute("SELECT key, neg, neu, pos, missing FROM scores WHERE key LIKE ?", (run_tag + ":%",))}
    arr = np.array([table[k] for k in keys], dtype=np.float64)
    probs = np.exp(arr[:, :3] - arr[:, :3].max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    summary = {
        "model": args.model, "system_prompt": SYSTEM_PROMPT, "n_headlines": len(keys),
        "n_scored_this_run": n_new, "seconds_this_run": round(time.time() - t0, 1),
        "argmax_share": {w: float(np.mean(arr[:, :3].argmax(axis=1) == j)) for j, w in enumerate(CLASS_WORDS)},
        "mean_score": float(np.mean(probs[:, 2] - probs[:, 0])),
        "headlines_with_a_class_outside_top_tokens": int(np.sum(arr[:, 3] > 0)),
    }
    print(json.dumps(summary, indent=1))
    if args.limit:
        return
    df["logit_neg"], df["logit_neu"], df["logit_pos"] = arr[:, 0], arr[:, 1], arr[:, 2]
    df.to_parquet(news_path)
    (PROJECT_ROOT / "data" / "news_sentiment_llm_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
