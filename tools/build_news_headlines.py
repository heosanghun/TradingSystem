"""
Build data/news_headlines.parquet from the public Hugging Face dataset `edaschau/bitcoin_news`
(Yahoo Finance articles matched on BTC/crypto keywords, UTC timestamps, license: openrail).

Download first (not redistributed here):
  https://huggingface.co/datasets/edaschau/bitcoin_news/resolve/main/BTC_match_title.csv
  https://huggingface.co/datasets/edaschau/bitcoin_news/resolve/main/BTC_match_text.csv
into data/raw/news/, then run this script. Only headline, timestamp and source are kept.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "news"
OUT = ROOT / "data" / "news_headlines.parquet"
START, END = "2020-08-01", "2023-12-31 23:59:59"


def load(name: str, match: str) -> pd.DataFrame:
    df = pd.read_csv(RAW / name, usecols=["time_unix", "title", "source"])
    df["match"] = match
    return df


def main() -> None:
    df = pd.concat([load("BTC_match_title.csv", "title"), load("BTC_match_text.csv", "text")], ignore_index=True)
    n_raw = len(df)
    df["published_at"] = pd.to_datetime(df["time_unix"], unit="s", utc=True)
    df["headline"] = df["title"].astype(str).str.strip()
    df["source"] = df["source"].fillna("Yahoo Finance").astype(str)
    df = df[(df["published_at"] >= START) & (df["published_at"] <= END) & (df["headline"].str.len() > 3)]
    n_range = len(df)
    df["_h"] = df["headline"].str.lower().map(lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest())
    # title matches sort first so a duplicated headline keeps the stricter match label
    df = df.sort_values(["published_at", "match"], ascending=[True, False]).drop_duplicates("_h", keep="first")
    out = df[["published_at", "headline", "source", "match"]].reset_index(drop=True)
    out.to_parquet(OUT, index=False)

    monthly = out["published_at"].dt.strftime("%Y-%m").value_counts().sort_index()
    summary = {
        "rows_raw_both_files": int(n_raw),
        "rows_in_range_before_dedup": int(n_range),
        "rows_final": int(len(out)),
        "title_matched": int((out["match"] == "title").sum()),
        "text_matched_only": int((out["match"] == "text").sum()),
        "first": str(out["published_at"].min()),
        "last": str(out["published_at"].max()),
        "months_with_zero": [m for m in pd.period_range(START, END, freq="M").strftime("%Y-%m") if m not in monthly.index],
        "monthly_min": int(monthly.min()),
        "monthly_max": int(monthly.max()),
        "top_sources": out["source"].value_counts().head(12).to_dict(),
        "eval_period_rows_2021-10_2023-12": int((out["published_at"] >= "2021-10-01").sum()),
    }
    (ROOT / "data" / "news_headlines_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
