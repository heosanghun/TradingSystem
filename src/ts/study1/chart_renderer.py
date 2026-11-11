"""
Fast In-Memory PIL Candlestick + Volume Chart Renderer (`224 × 224 × 3`).

Implements Chapter III Sec 1.1 & Table A.2:
- Window `W = 60` hourly OHLCV bars
- White background `(255, 255, 255)`
- Red bullish candles (`close >= open`, Korean convention 양봉 빨간색)
- Blue bearish candles (`close < open`, Korean convention 음봉 파란색)
- Bottom 22% panel renders synchronized volume bars
- Outputs ImageNet-normalized `(3, 224, 224)` float32 tensor (or PIL Image)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

COLOR_BG = (255, 255, 255)
COLOR_BULL = (214, 39, 40)    # Red (양봉)
COLOR_BEAR = (31, 119, 180)   # Blue (음봉)


class CandlestickChartRenderer:
    """Renders 60-bar OHLCV windows into `(3, 224, 224)` tensors without Matplotlib overhead."""

    def __init__(self, window_bars: int = 60, image_size: Tuple[int, int] = (224, 224)) -> None:
        self.window_bars = window_bars
        self.width, self.height = int(image_size[0]), int(image_size[1])

    def render_pil(
        self, opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray
    ) -> Image.Image:
        img = Image.new("RGB", (self.width, self.height), COLOR_BG)
        draw = ImageDraw.Draw(img)

        n = len(closes)
        if n == 0:
            return img

        pad_x = 4
        pad_top = 6
        price_bottom = int(self.height * 0.74)
        vol_top = int(self.height * 0.78)
        vol_bottom = self.height - 4

        p_min = float(np.min(lows))
        p_max = float(np.max(highs))
        p_span = max(p_max - p_min, 1e-8)

        v_max = max(float(np.max(volumes)), 1e-8)

        usable_w = self.width - 2 * pad_x
        slot_w = usable_w / max(n, 1)
        half_body = max(1, int(slot_w * 0.35))

        def price_to_y(p: float) -> int:
            ratio = (p - p_min) / p_span
            return int(round(price_bottom - ratio * (price_bottom - pad_top)))

        for i in range(n):
            cx = int(round(pad_x + (i + 0.5) * slot_w))
            is_bull = closes[i] >= opens[i]
            color = COLOR_BULL if is_bull else COLOR_BEAR

            # High-Low wick
            y_high = price_to_y(float(highs[i]))
            y_low = price_to_y(float(lows[i]))
            draw.line([(cx, y_high), (cx, y_low)], fill=color, width=1)

            # Open-Close body
            y_open = price_to_y(float(opens[i]))
            y_close = price_to_y(float(closes[i]))
            top_y = min(y_open, y_close)
            bot_y = max(y_open, y_close)
            if bot_y == top_y:
                bot_y = min(price_bottom, top_y + 1)
            draw.rectangle([(cx - half_body, top_y), (cx + half_body, bot_y)], fill=color, outline=color)

            # Volume bar
            v_ratio = max(0.0, float(volumes[i])) / v_max
            y_vol = int(round(vol_bottom - v_ratio * (vol_bottom - vol_top)))
            draw.rectangle([(cx - half_body, y_vol), (cx + half_body, vol_bottom)], fill=color)

        return img

    def render_tensor_chw(
        self, opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray
    ) -> np.ndarray:
        """Render and ImageNet-normalize a single window to `(3, H, W)` float32."""
        pil_img = self.render_pil(opens, highs, lows, closes, volumes)
        arr_hwc = np.asarray(pil_img, dtype=np.float32) / 255.0
        arr_chw = np.transpose(arr_hwc, (2, 0, 1))
        return (arr_chw - IMAGENET_MEAN) / IMAGENET_STD

    def render_series_batch(self, ohlcv_df: pd.DataFrame) -> np.ndarray:
        """
        Render causal trailing `W`-bar chart tensors `(N, 3, 224, 224)` for every bar in `ohlcv_df`.
        Early bars `i < W - 1` left-pad with bar `0` so no future bar is ever referenced.
        """
        o = ohlcv_df["open"].to_numpy(dtype=np.float64)
        h = ohlcv_df["high"].to_numpy(dtype=np.float64)
        l = ohlcv_df["low"].to_numpy(dtype=np.float64)
        c = ohlcv_df["close"].to_numpy(dtype=np.float64)
        v = ohlcv_df["volume"].to_numpy(dtype=np.float64)

        n = len(ohlcv_df)
        W = self.window_bars
        out = np.empty((n, 3, self.height, self.width), dtype=np.float32)

        for i in range(n):
            start = max(0, i - W + 1)
            win_o = o[start : i + 1]
            win_h = h[start : i + 1]
            win_l = l[start : i + 1]
            win_c = c[start : i + 1]
            win_v = v[start : i + 1]
            if len(win_c) < W:
                pad = W - len(win_c)
                win_o = np.concatenate([np.repeat(win_o[:1], pad), win_o])
                win_h = np.concatenate([np.repeat(win_h[:1], pad), win_h])
                win_l = np.concatenate([np.repeat(win_l[:1], pad), win_l])
                win_c = np.concatenate([np.repeat(win_c[:1], pad), win_c])
                win_v = np.concatenate([np.repeat(win_v[:1], pad), win_v])
            out[i] = self.render_tensor_chw(win_o, win_h, win_l, win_c, win_v)
        return out
