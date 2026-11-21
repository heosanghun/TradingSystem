"""Download public klines or validate locally supplied research archives.

News preparation is not news acquisition or a license verification service.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def prepare_news(input_path, output_path, source_url, license_info,
                 start="2020-08-01", end="2023-12-31T23:59:59Z"):
    """Deduplicate an authorized archive and report coverage before export."""
    from ts.data.news_fetcher import PointInTimeNewsStore
    source = Path(input_path)
    frame = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)
    if not source_url.strip() or not license_info.strip():
        raise ValueError("Record the source URL and reviewed license/use terms")
    if "source" not in frame or frame["source"].isna().any() or frame["source"].astype(str).str.strip().eq("").any():
        raise ValueError("Every article must identify its source")
    parsed = [pd.Timestamp(value) for value in frame["published_at"]]
    if any(pd.isna(value) or value.tzinfo is None for value in parsed):
        raise ValueError("Publication times must include an explicit timezone; naive times are ambiguous")
    frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True)
    raw_count = len(frame)
    start_utc, end_utc = pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True)
    frame = frame.loc[frame.published_at.between(start_utc, end_utc)].copy()
    store = PointInTimeNewsStore(frame)
    clean = store.df[["published_at", "headline", "source"]]
    months = pd.period_range(start_utc.tz_localize(None).to_period("M"),
                             end_utc.tz_localize(None).to_period("M"), freq="M")
    counts = clean.published_at.dt.tz_localize(None).dt.to_period("M").value_counts().reindex(months, fill_value=0)
    report = dict(input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  source_url=source_url, license_info=license_info, raw_rows=raw_count,
                  in_range_rows=len(frame), deduplicated_rows=len(clean),
                  per_source=clean.source.value_counts().to_dict(),
                  monthly_counts={str(k): int(v) for k, v in counts.items()},
                  empty_months=[str(k) for k, v in counts.items() if v == 0])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".coverage.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if report["empty_months"]:
        raise ValueError("Archive has empty months; see coverage report. No completed news archive exported")
    clean.attrs.update(source_url=source_url, license_info=license_info,
                       publication_time_semantics="actual publication time declared by provider")
    temporary = output.with_suffix(".parquet.tmp")
    clean.to_parquet(temporary, index=False)
    temporary.replace(output)
    PointInTimeNewsStore.from_file(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    klines = commands.add_parser("klines")
    klines.add_argument("--symbol", default="BTCUSDT")
    klines.add_argument("--interval", choices=["1h", "1m"], default="1h")
    klines.add_argument("--start", required=True)
    klines.add_argument("--end", required=True)
    hourly = commands.add_parser("inspect-hourly")
    hourly.add_argument("--input", required=True)
    news = commands.add_parser("prepare-news")
    news.add_argument("--input", required=True)
    news.add_argument("--output", default=str(Path(__file__).parent / "news_headlines.parquet"))
    news.add_argument("--source-url", required=True)
    news.add_argument("--license-info", required=True)
    args = parser.parse_args()
    if args.command == "klines":
        from ts.data.binance_fetcher import BinanceDataFetcher
        frame, gaps = BinanceDataFetcher(Path(__file__).parent / "cache").fetch_ohlcv(
            args.symbol, args.interval, args.start, args.end, realign_grid=False)
        print(json.dumps(dict(rows=len(frame), start=str(frame.index.min()), end=str(frame.index.max()), gaps=gaps), indent=2))
    elif args.command == "inspect-hourly":
        from ts.data.macro_fetcher import MacroAssetFetcher
        frame = MacroAssetFetcher._hourly_archive(Path(args.input))
        print(json.dumps(dict(rows=len(frame), start=str(frame.index.min()), end=str(frame.index.max()), attrs=frame.attrs), indent=2))
    else:
        print(json.dumps(prepare_news(args.input, args.output, args.source_url, args.license_info), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
