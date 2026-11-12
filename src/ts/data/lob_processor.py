"""
Limit Order Book (LOB) & Sub-Bar Microstructure Window Processor (`100 × 40`).

Implements Chapter V Sec 3.1 (Table 5.1) & Appendix A (Table A.4):
- Window shape: `100 ticks × 40 features` (transposed to `(40, 100)` for PyTorch `Conv1d`)
- 40-feature composition (`20 + 10 + 10`):
  1. [0..19]  Top-10 Bid & Ask Depth Volume Ratios (20 features)
  2. [20..29] Level Spreads & Multi-Lag Mid-Price Change Rates (10 features)
  3. [30..39] Recent Tick Trade Volume, Direction, & Flow Imbalance (10 features)
- Standardized via within-window Z-score (`mean=0, std=1` per feature channel across the 100 ticks).
- Temporal alignment (Ch. V Sec 4.4): Uniformly extracts 10 windows per 1-hour bar for offline distillation,
  and extracts the latest pre-close 100-tick window for online 1-hour Student execution.

Also provides an explicit Sub-Bar Trade/1m Microstructure Adapter (`extract_from_subbar_klines`)
matching `구현계획_261111.md` Sec 6.2 fallback when commercial Tardis L2 snapshots are unavailable.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


class LOBWindowProcessor:
    """Builds Z-scored `(40, 100)` microstructure tensors from L2 LOB snapshots or sub-bar trades."""

    def __init__(self, seq_len: int = 100, num_features: int = 40, windows_per_bar: int = 10) -> None:
        if num_features != 40:
            raise ValueError(f"Thesis Table 5.1 specifies exactly 40 LOB features, got {num_features}")
        self.seq_len = seq_len
        self.num_features = num_features
        self.windows_per_bar = windows_per_bar

    @staticmethod
    def zscore_window(window_tf: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """
        Standardize a `(L, 40)` window along the temporal axis `L` and return `(40, L)` `float32`.
        """
        mean = np.mean(window_tf, axis=0, keepdims=True)
        std = np.std(window_tf, axis=0, keepdims=True)
        normed = (window_tf - mean) / np.maximum(std, eps)
        normed = np.nan_to_num(normed, nan=0.0, posinf=0.0, neginf=0.0)
        return normed.T.astype(np.float32)  # (40, L)

    def compute_40_lob_features(
        self,
        bid_prices: np.ndarray,
        bid_vols: np.ndarray,
        ask_prices: np.ndarray,
        ask_vols: np.ndarray,
        trade_vols: np.ndarray,
        trade_signs: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the raw `(N_ticks, 40)` microstructure matrix from 10-level L2 arrays:
        - `bid_prices`, `bid_vols`, `ask_prices`, `ask_vols`: shape `(N, 10)`
        - `trade_vols`, `trade_signs` (`+1` buy aggressor, `-1` sell aggressor, `0` none): shape `(N,)`
        """
        n = bid_prices.shape[0]
        if bid_prices.shape != (n, 10) or ask_prices.shape != (n, 10):
            raise ValueError("L2 arrays must have shape `(N_ticks, 10)`.")

        eps = 1e-8
        feats = np.zeros((n, 40), dtype=np.float64)

        # 1. [0..19] Top-10 Bid & Ask Depth Volume Ratios (20 features)
        total_depth = np.sum(bid_vols + ask_vols, axis=1, keepdims=True) + eps
        feats[:, 0:10] = bid_vols / total_depth
        feats[:, 10:20] = ask_vols / total_depth

        # 2. [20..29] Spreads & Mid-Price Change Rates (10 features)
        mid = 0.5 * (bid_prices[:, 0] + ask_prices[:, 0])
        mid_safe = np.maximum(mid, eps)
        # 2a. [20..24] Relative bid-ask spread at levels 1..5
        for lvl in range(5):
            feats[:, 20 + lvl] = (ask_prices[:, lvl] - bid_prices[:, lvl]) / mid_safe
        # 2b. [25] Level-1 micro-price relative deviation from mid
        lvl1_vol = bid_vols[:, 0] + ask_vols[:, 0] + eps
        micro_price = (bid_prices[:, 0] * ask_vols[:, 0] + ask_prices[:, 0] * bid_vols[:, 0]) / lvl1_vol
        feats[:, 25] = (micro_price - mid) / mid_safe
        # 2c. [26..29] Multi-lag log mid-price returns (lags 1, 3, 5, 10)
        log_mid = np.log(mid_safe)
        for idx, lag in enumerate((1, 3, 5, 10)):
            ret = np.zeros(n, dtype=np.float64)
            if n > lag:
                ret[lag:] = log_mid[lag:] - log_mid[:-lag]
            feats[:, 26 + idx] = ret

        # 3. [30..39] Recent Tick Trade Volume, Direction, & Flow Imbalance (10 features)
        signed_vol = trade_vols * trade_signs
        feats[:, 30] = np.log1p(np.maximum(trade_vols, 0.0))
        feats[:, 31] = trade_signs
        feats[:, 32] = signed_vol / (lvl1_vol + eps)

        # Level-1 Order Flow Imbalance (Cont et al. OFI)
        d_bid_p = np.diff(bid_prices[:, 0], prepend=bid_prices[0, 0])
        d_ask_p = np.diff(ask_prices[:, 0], prepend=ask_prices[0, 0])
        prev_bv = np.roll(bid_vols[:, 0], 1)
        prev_bv[0] = bid_vols[0, 0]
        prev_av = np.roll(ask_vols[:, 0], 1)
        prev_av[0] = ask_vols[0, 0]
        ofi_1 = np.where(d_bid_p >= 0, bid_vols[:, 0], 0.0) - np.where(d_bid_p <= 0, prev_bv, 0.0)
        ofi_1 -= np.where(d_ask_p <= 0, ask_vols[:, 0], 0.0) - np.where(d_ask_p >= 0, prev_av, 0.0)
        feats[:, 33] = ofi_1 / (lvl1_vol + eps)

        # Rolling trade flow imbalance & buy-ratios over 5, 10, 20 ticks
        sv_series = pd.Series(signed_vol)
        tv_series = pd.Series(np.maximum(trade_vols, 0.0))
        buy_v_series = pd.Series(np.where(trade_signs > 0, trade_vols, 0.0))
        for j, win in enumerate((5, 10, 20)):
            denom = tv_series.rolling(win, min_periods=1).sum().to_numpy() + eps
            feats[:, 34 + j] = sv_series.rolling(win, min_periods=1).sum().to_numpy() / denom
            feats[:, 37 + j] = buy_v_series.rolling(win, min_periods=1).sum().to_numpy() / denom

        return feats

    def sample_bar_windows_for_distillation(
        self, bar_lob_features_tf: np.ndarray
    ) -> List[np.ndarray]:
        """
        Given `(N_ticks_in_bar, 40)` features inside one 1-hour bar (`N >= 100`),
        uniformly extracts `self.windows_per_bar` (10) Z-scored `(40, 100)` windows.
        """
        n = bar_lob_features_tf.shape[0]
        L = self.seq_len
        if n < L:
            pad = np.repeat(bar_lob_features_tf[:1], L - n, axis=0)
            bar_lob_features_tf = np.concatenate([pad, bar_lob_features_tf], axis=0)
            n = L

        if n == L or self.windows_per_bar == 1:
            return [self.zscore_window(bar_lob_features_tf[-L:]) for _ in range(self.windows_per_bar)]

        starts = np.linspace(0, n - L, num=self.windows_per_bar, dtype=int)
        return [self.zscore_window(bar_lob_features_tf[s : s + L]) for s in starts]

    def latest_decision_window(self, pre_close_lob_features_tf: np.ndarray) -> np.ndarray:
        """Extract the single latest `(40, 100)` Z-scored window immediately before bar close."""
        n = pre_close_lob_features_tf.shape[0]
        L = self.seq_len
        if n < L:
            pad = np.repeat(pre_close_lob_features_tf[:1], L - n, axis=0)
            win = np.concatenate([pad, pre_close_lob_features_tf], axis=0)
        else:
            win = pre_close_lob_features_tf[-L:]
        return self.zscore_window(win)

    def extract_from_subbar_klines(
        self, subbar_df: pd.DataFrame, decision_idx_inclusive: int
    ) -> Tuple[np.ndarray, str]:
        """Build a 40-channel proxy from trailing OHLCV sub-bars.

        This adapter has no observed bid/ask depth and does not reconstruct an
        L2 order book. Its channels and sampling clock differ from tick-level
        LOB inputs despite the shared (40,100) shape. Missing taker volume and
        trade count use explicit estimates in the code, not observed trades.
        Short histories repeat the earliest available sub-bar. The caller must
        record this proxy mode and its actual bar duration.
        """
        L = self.seq_len
        start = max(0, decision_idx_inclusive - L + 1)
        sub = subbar_df.iloc[start : decision_idx_inclusive + 1]
        if len(sub) < L:
            pad_df = pd.concat([sub.iloc[[0]]] * (L - len(sub)), ignore_index=False)
            sub = pd.concat([pad_df, sub], axis=0)

        eps = 1e-8
        o = sub["open"].to_numpy(dtype=np.float64)
        h = sub["high"].to_numpy(dtype=np.float64)
        l = sub["low"].to_numpy(dtype=np.float64)
        c = sub["close"].to_numpy(dtype=np.float64)
        v = np.maximum(sub["volume"].to_numpy(dtype=np.float64), 0.0)
        tb = (
            np.maximum(sub["taker_buy_base_vol"].to_numpy(dtype=np.float64), 0.0)
            if "taker_buy_base_vol" in sub.columns
            else 0.5 * v
        )
        ts = np.maximum(v - tb, 0.0)
        nt = (
            sub["num_trades"].to_numpy(dtype=np.float64)
            if "num_trades" in sub.columns
            else np.ones_like(v)
        )

        feats = np.zeros((L, 40), dtype=np.float64)
        c_safe = np.maximum(c, eps)

        # [0..19] Multi-horizon buy/sell volume depth shares across 10 rolling horizons
        s_tb = pd.Series(tb)
        s_ts = pd.Series(ts)
        for idx, h_win in enumerate((1, 2, 3, 5, 8, 13, 21, 34, 55, 89)):
            b_sum = s_tb.rolling(h_win, min_periods=1).sum().to_numpy()
            a_sum = s_ts.rolling(h_win, min_periods=1).sum().to_numpy()
            tot = b_sum + a_sum + eps
            feats[:, idx] = b_sum / tot
            feats[:, 10 + idx] = a_sum / tot

        # [20..29] High-low range, upper/lower wicks, body, and multi-lag log returns
        feats[:, 20] = (h - l) / c_safe
        feats[:, 21] = (h - np.maximum(o, c)) / c_safe
        feats[:, 22] = (np.minimum(o, c) - l) / c_safe
        feats[:, 23] = (c - o) / c_safe
        feats[:, 24] = np.abs(c - o) / (h - l + eps)
        log_c = np.log(c_safe)
        for j, lag in enumerate((1, 2, 5, 10, 20)):
            r = np.zeros(L, dtype=np.float64)
            if L > lag:
                r[lag:] = log_c[lag:] - log_c[:-lag]
            feats[:, 25 + j] = r

        # [30..39] Trade volume, direction, imbalance, intensity, Amihud illiquidity
        imb = (tb - ts) / (v + eps)
        feats[:, 30] = np.log1p(v)
        feats[:, 31] = np.sign(c - o)
        feats[:, 32] = imb
        feats[:, 33] = np.log1p(np.maximum(nt, 0.0))
        feats[:, 34] = v / (nt + 1.0)
        feats[:, 35] = np.abs(feats[:, 25]) / (v * c_safe + eps)
        s_imb = pd.Series(tb - ts)
        s_v = pd.Series(v)
        for j, win in enumerate((5, 10, 20, 50)):
            feats[:, 36 + j] = (
                s_imb.rolling(win, min_periods=1).sum().to_numpy()
                / (s_v.rolling(win, min_periods=1).sum().to_numpy() + eps)
            )

        return self.zscore_window(feats), "subbar_trade_microstructure"
