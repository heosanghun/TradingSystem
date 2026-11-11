"""
Point-in-Time Financial News Store & Deduplicator.

Enforces strict publication-timestamp causal filtering (`published_at < bar_close_utc`)
for both:
- Study 1: 3-class sentiment logits + 24-hour EWMA (`alpha = 0.08`, Eq. III.7–III.8)
- Study 2: Edge Verifier & Multi-Agent Debate headline context (Ch. IV Sec 2.3 & 3)
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ts.data.temporal_guard import assert_no_future_leakage


class PointInTimeNewsStore:
    """
    Causal news repository indexed by UTC publication timestamp.

    Expected columns in input DataFrame / CSV / Parquet:
    - `published_at` (UTC parseable timestamp)
    - `headline` (non-empty real headline string; synthetic "Crypto News Article N" placeholders are rejected!)
    - Optional: `source`, `logit_neg`, `logit_neu`, `logit_pos`
    """

    def __init__(self, articles_df: Optional[pd.DataFrame] = None) -> None:
        if articles_df is None or articles_df.empty:
            self.df = pd.DataFrame(
                columns=["published_at", "headline", "source", "logit_neg", "logit_neu", "logit_pos"]
            )
            self.df["published_at"] = pd.to_datetime(self.df["published_at"], utc=True)
            return

        df = articles_df.copy()
        if "published_at" not in df.columns or "headline" not in df.columns:
            raise ValueError("News DataFrame must contain `published_at` and `headline` columns.")

        df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
        df["headline"] = df["headline"].astype(str).str.strip()
        df = df[df["headline"].str.len() > 3].copy()

        # Guard against synthetic placeholder articles flagged in `구현계획_261111.md` Sec 3.1
        is_fake_placeholder = df["headline"].str.match(r"^Crypto News Article\s+\d+$", case=False, na=False)
        if is_fake_placeholder.any():
            raise ValueError(
                f"Detected {int(is_fake_placeholder.sum())} synthetic 'Crypto News Article N' rows; "
                "synthetic news placeholders are strictly prohibited."
            )

        # Deduplicate by normalized headline SHA-1 hash (Thesis Ch. VI Sec 2)
        df["_title_hash"] = df["headline"].str.lower().map(
            lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest()
        )
        df = (
            df.sort_values("published_at")
            .drop_duplicates(subset=["_title_hash"], keep="first")
            .drop(columns=["_title_hash"])
            .reset_index(drop=True)
        )
        if "source" not in df.columns:
            df["source"] = "unknown"
        for col in ("logit_neg", "logit_neu", "logit_pos"):
            if col not in df.columns:
                df[col] = np.nan

        self.df = df

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "PointInTimeNewsStore":
        p = Path(path)
        if p.suffix.lower() == ".parquet":
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)
        return cls(df)

    def query_before(
        self,
        decision_time_utc: pd.Timestamp,
        lookback_hours: float = 24.0,
        max_headlines: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Return articles strictly satisfying `decision_time - lookback <= published_at < decision_time`.
        """
        if self.df.empty:
            return self.df.copy()

        dec_ts = (
            decision_time_utc.tz_localize("UTC")
            if decision_time_utc.tzinfo is None
            else decision_time_utc.tz_convert("UTC")
        )
        start_ts = dec_ts - pd.Timedelta(hours=lookback_hours)

        # Binary search on sorted `published_at`
        pub_series = self.df["published_at"]
        left = int(pub_series.searchsorted(start_ts, side="left"))
        right = int(pub_series.searchsorted(dec_ts, side="left"))  # strict `< dec_ts`
        sub = self.df.iloc[left:right].copy()

        assert_no_future_leakage(sub["published_at"], dec_ts, strict_less=True, context="PointInTimeNewsStore")
        if max_headlines is not None and len(sub) > max_headlines:
            sub = sub.iloc[-max_headlines:].copy()
        return sub

    def format_headlines_for_llm(
        self, decision_time_utc: pd.Timestamp, lookback_hours: float = 24.0, max_items: int = 15
    ) -> str:
        sub = self.query_before(decision_time_utc, lookback_hours=lookback_hours, max_headlines=max_items)
        if sub.empty:
            return "No verified headlines published in the preceding window."
        lines: List[str] = []
        for _, row in sub.iterrows():
            ts_str = pd.Timestamp(row["published_at"]).strftime("%Y-%m-%d %H:%M UTC")
            lines.append(f"[{ts_str}] ({row['source']}) {row['headline']}")
        return "\n".join(lines)

    @staticmethod
    def score_from_logits(logits_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Implements Eq. (III.7):
          p_c = exp(z_c) / sum_j exp(z_j) for c in {-1, 0, +1}
          Score = -1 * p_{-1} + 0 * p_0 + 1 * p_{+1} = p_{+1} - p_{-1}
        Returns `(scores, probs)` of shapes `(N,)` and `(N, 3)`.
        """
        z = np.asarray(logits_3d, dtype=np.float64)
        if z.ndim == 1:
            z = z[None, :]
        z_shift = z - np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(z_shift)
        probs = exp_z / np.sum(exp_z, axis=-1, keepdims=True)
        scores = probs[:, 2] - probs[:, 0]
        return scores, probs

    def build_hourly_sentiment_summary(
        self,
        hourly_index: pd.DatetimeIndex,
        ewma_span_bars: int = 24,
    ) -> pd.DataFrame:
        """
        Implements Eq. (III.7)–(III.8) & Section 1.4 summary vector (8 base statistics
        prior to the learnable Linear+LayerNorm -> 32d projection):
          `[ewma_score, p_neg_24h, p_neu_24h, p_pos_24h, log1p_count_24h, std_24h, max_24h, min_24h]`
        Causal convention: bar opening at `t` makes its decision at bar close `t + 1h`,
        using only articles with `published_at < bar_decision_time`.
        """
        alpha = 2.0 / (ewma_span_bars + 1.0)  # N=24 -> 0.08
        n_bars = len(hourly_index)
        out = np.zeros((n_bars, 8), dtype=np.float32)
        out[:, 2] = 1.0  # neutral probability default when count == 0

        if self.df.empty or self.df[["logit_neg", "logit_neu", "logit_pos"]].isna().all().all():
            return pd.DataFrame(
                out,
                index=hourly_index,
                columns=[
                    "ewma_score", "p_neg_24h", "p_neu_24h", "p_pos_24h",
                    "log1p_count_24h", "std_24h", "max_24h", "min_24h",
                ],
            )

        valid = self.df.dropna(subset=["logit_neg", "logit_neu", "logit_pos"]).copy()
        logits = valid[["logit_neg", "logit_neu", "logit_pos"]].to_numpy(dtype=np.float64)
        scores, probs = self.score_from_logits(logits)
        pub_times = valid["published_at"]

        ewma_val = 0.0
        for i, bar_ts in enumerate(hourly_index):
            dec_ts = bar_ts + pd.Timedelta(hours=1)
            bar_start = bar_ts
            win_start = dec_ts - pd.Timedelta(hours=24)

            l_bar = int(pub_times.searchsorted(bar_start, side="left"))
            r_dec = int(pub_times.searchsorted(dec_ts, side="left"))
            l_24 = int(pub_times.searchsorted(win_start, side="left"))

            bar_score = float(np.mean(scores[l_bar:r_dec])) if r_dec > l_bar else 0.0
            ewma_val = alpha * bar_score + (1.0 - alpha) * ewma_val

            out[i, 0] = ewma_val
            if r_dec > l_24:
                s_win = scores[l_24:r_dec]
                p_win = probs[l_24:r_dec]
                out[i, 1:4] = np.mean(p_win, axis=0)
                out[i, 4] = np.log1p(len(s_win))
                out[i, 5] = float(np.std(s_win)) if len(s_win) > 1 else 0.0
                out[i, 6] = float(np.max(s_win))
                out[i, 7] = float(np.min(s_win))
            else:
                out[i, 1:4] = [0.0, 1.0, 0.0]

        return pd.DataFrame(
            out,
            index=hourly_index,
            columns=[
                "ewma_score", "p_neg_24h", "p_neu_24h", "p_pos_24h",
                "log1p_count_24h", "std_24h", "max_24h", "min_24h",
            ],
        )
