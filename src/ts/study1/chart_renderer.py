"""
In-memory PIL candlestick and volume chart renderer (`224 x 224 x 3`).

Implements Chapter III Sec 1.1 and Table A.2:
- Window `W = 60` hourly OHLCV bars
- White background `(255, 255, 255)`
- Red bullish candles (`close >= open`, Korean convention)
- Blue bearish candles (`close < open`, Korean convention)
- Bottom panel renders synchronized volume bars
- Stores full series compactly as `uint8` `(N, 224, 224, 3)` (or `np.memmap`) and applies
  CHW transposition and ImageNet normalization lazily per minibatch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
_IMAGENET_MEAN_TUPLE = (0.485, 0.456, 0.406)
_IMAGENET_STD_TUPLE = (0.229, 0.224, 0.225)

COLOR_BG = (255, 255, 255)
COLOR_BULL = (214, 39, 40)
COLOR_BEAR = (31, 119, 180)


def normalize_chart_batch_to_tensor(batch_imgs: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert a minibatch of chart images into an ImageNet-normalized `(B, 3, H, W)` float32 tensor.

    Accepts either:
    - `uint8` HWC minibatches `(B, H, W, 3)` produced by `render_series_uint8`, or
    - pre-normalized `float32` CHW minibatches `(B, 3, H, W)` produced by `render_series_batch`.
    """
    if batch_imgs.ndim != 4:
        raise ValueError(f"Expected 4D chart batch, got shape {batch_imgs.shape}")

    if batch_imgs.dtype == np.uint8 and batch_imgs.shape[-1] == 3:
        arr = np.ascontiguousarray(batch_imgs)
        t = torch.from_numpy(arr).to(device=device, non_blocking=False)
        t = t.permute(0, 3, 1, 2).to(dtype=torch.float32).div(255.0)
        mean = torch.tensor(_IMAGENET_MEAN_TUPLE, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD_TUPLE, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        return (t - mean) / std

    if batch_imgs.shape[1] == 3:
        arr_f32 = np.ascontiguousarray(batch_imgs, dtype=np.float32)
        return torch.from_numpy(arr_f32).to(device=device, dtype=torch.float32)

    raise ValueError(f"Unsupported chart batch layout/dtype: shape={batch_imgs.shape}, dtype={batch_imgs.dtype}")


class CandlestickChartRenderer:
    """Renders 60-bar OHLCV windows into `224 x 224` RGB images or normalized tensors."""

    def __init__(self, window_bars: int = 60, image_size: Union[int, Tuple[int, int]] = (224, 224)) -> None:
        self.window_bars = window_bars
        if isinstance(image_size, int):
            self.width, self.height = int(image_size), int(image_size)
        else:
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

            y_high = price_to_y(float(highs[i]))
            y_low = price_to_y(float(lows[i]))
            draw.line([(cx, y_high), (cx, y_low)], fill=color, width=1)

            y_open = price_to_y(float(opens[i]))
            y_close = price_to_y(float(closes[i]))
            top_y = min(y_open, y_close)
            bot_y = max(y_open, y_close)
            if bot_y == top_y:
                bot_y = min(price_bottom, top_y + 1)
            draw.rectangle([(cx - half_body, top_y), (cx + half_body, bot_y)], fill=color, outline=color)

            v_ratio = max(0.0, float(volumes[i])) / v_max
            y_vol = int(round(vol_bottom - v_ratio * (vol_bottom - vol_top)))
            draw.rectangle([(cx - half_body, y_vol), (cx + half_body, vol_bottom)], fill=color)

        return img

    def render_rgb_uint8(
        self, opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray
    ) -> np.ndarray:
        """Render a single window to `(H, W, 3)` uint8 RGB array."""
        pil_img = self.render_pil(opens, highs, lows, closes, volumes)
        return np.asarray(pil_img, dtype=np.uint8)

    def render_tensor_chw(
        self, opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray
    ) -> np.ndarray:
        """Render and ImageNet-normalize a single window to `(3, H, W)` float32."""
        pil_img = self.render_pil(opens, highs, lows, closes, volumes)
        arr_hwc = np.asarray(pil_img, dtype=np.float32) / 255.0
        arr_chw = np.transpose(arr_hwc, (2, 0, 1))
        return (arr_chw - IMAGENET_MEAN) / IMAGENET_STD

    def _extract_causal_window(
        self,
        o: np.ndarray,
        h: np.ndarray,
        l: np.ndarray,
        c: np.ndarray,
        v: np.ndarray,
        bar_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        w = self.window_bars
        start = max(0, bar_idx - w + 1)
        win_o = o[start : bar_idx + 1]
        win_h = h[start : bar_idx + 1]
        win_l = l[start : bar_idx + 1]
        win_c = c[start : bar_idx + 1]
        win_v = v[start : bar_idx + 1]
        if len(win_c) < w:
            pad = w - len(win_c)
            win_o = np.concatenate([np.repeat(win_o[:1], pad), win_o])
            win_h = np.concatenate([np.repeat(win_h[:1], pad), win_h])
            win_l = np.concatenate([np.repeat(win_l[:1], pad), win_l])
            win_c = np.concatenate([np.repeat(win_c[:1], pad), win_c])
            win_v = np.concatenate([np.repeat(win_v[:1], pad), win_v])
        return win_o, win_h, win_l, win_c, win_v

    def render_series_uint8(
        self,
        ohlcv_df: pd.DataFrame,
        mmap_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """
        Render causal trailing `W`-bar charts into a compact `(N, H, W, 3)` uint8 array (or memmap).
        Early bars `i < W - 1` left-pad with bar `0` so no future bar is referenced.
        """
        o = ohlcv_df["open"].to_numpy(dtype=np.float64)
        h = ohlcv_df["high"].to_numpy(dtype=np.float64)
        l = ohlcv_df["low"].to_numpy(dtype=np.float64)
        c = ohlcv_df["close"].to_numpy(dtype=np.float64)
        v = ohlcv_df["volume"].to_numpy(dtype=np.float64)

        n = len(ohlcv_df)
        shape = (n, self.height, self.width, 3)
        if mmap_path is not None:
            mmap_file = Path(mmap_path)
            mmap_file.parent.mkdir(parents=True, exist_ok=True)
            out: np.ndarray = np.memmap(mmap_file, dtype=np.uint8, mode="w+", shape=shape)
        else:
            out = np.empty(shape, dtype=np.uint8)

        for i in range(n):
            win_o, win_h, win_l, win_c, win_v = self._extract_causal_window(o, h, l, c, v, i)
            out[i] = self.render_rgb_uint8(win_o, win_h, win_l, win_c, win_v)

        if isinstance(out, np.memmap):
            out.flush()
        return out

    def render_series_batch(self, ohlcv_df: pd.DataFrame) -> np.ndarray:
        """
        Render causal trailing `W`-bar chart tensors `(N, 3, H, W)` float32 for small batches or tests.
        For full multi-year series, prefer `render_series_uint8` to avoid float32 memory expansion.
        """
        o = ohlcv_df["open"].to_numpy(dtype=np.float64)
        h = ohlcv_df["high"].to_numpy(dtype=np.float64)
        l = ohlcv_df["low"].to_numpy(dtype=np.float64)
        c = ohlcv_df["close"].to_numpy(dtype=np.float64)
        v = ohlcv_df["volume"].to_numpy(dtype=np.float64)

        n = len(ohlcv_df)
        out = np.empty((n, 3, self.height, self.width), dtype=np.float32)
        for i in range(n):
            win_o, win_h, win_l, win_c, win_v = self._extract_causal_window(o, h, l, c, v, i)
            out[i] = self.render_tensor_chw(win_o, win_h, win_l, win_c, win_v)
        return out
