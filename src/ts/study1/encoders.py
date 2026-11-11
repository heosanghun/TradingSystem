"""
Study 1 Multimodal Feature Encoders & 2-Token Cross-Modal Attention (Ch. III Sec 1).

Implements exact equations from `[개정본] 박사논문_개정본_본보기_261110.pdf`:
- Visual (`128-d`): ResNet-18 + CBAM (`r=16`, `7x7` spatial conv, Eq. III.2–III.3, Table 3.1)
- Technical (`64-d`): MACD(12,26,9), RSI(14), Wilder ATR(14) + causal Z-score indicators (Eq. III.4–III.6)
- Sentiment (`32-d`): 24h EWMA (`alpha=0.08`, Eq. III.7–III.8) + 24h window statistics -> Linear + LayerNorm
- Cross-Modal Fusion (`128-d`): Sentiment = Query (`1×128`), `[Visual, Tech]` = Key/Value (`2×128`),
  `s_t = LayerNorm(Q + softmax(Q K^T / sqrt(128)) V)` (Eq. III.9–III.12)
- Fold-local Auxiliary Regime Pretrainer: Trains encoder weights strictly on each training fold's
  regime labels before freezing for XGBoost & PPO (resolves Issue #9 in `구현계획_261111.md`).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module (Woo et al.; Thesis Eq. III.2–III.3).
    """

    def __init__(self, channels: int = 512, reduction: int = 16, spatial_kernel: int = 7) -> None:
        super().__init__()
        mid = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        padding = spatial_kernel // 2
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        b, c, _, _ = x.shape
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)
        mc = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool)).view(b, c, 1, 1)
        x_ch = x * mc

        avg_sp = torch.mean(x_ch, dim=1, keepdim=True)
        max_sp, _ = torch.max(x_ch, dim=1, keepdim=True)
        ms = torch.sigmoid(self.spatial_conv(torch.cat([avg_sp, max_sp], dim=1)))
        return x_ch * ms


class VisualResNetCBAMEncoder(nn.Module):
    """Table 3.1: ResNet-18 + CBAM -> GAP(512) -> Linear(128) + ReLU + LayerNorm."""

    def __init__(
        self,
        out_dim: int = 128,
        cbam_reduction: int = 16,
        spatial_kernel: int = 7,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.cbam = CBAMBlock(channels=512, reduction=cbam_reduction, spatial_kernel=spatial_kernel)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(512, out_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(out_dim),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # img: (B, 3, 224, 224)
        f = self.stem(img)
        f = self.layer4(self.layer3(self.layer2(self.layer1(f))))  # (B, 512, 7, 7)
        f = self.cbam(f)
        f = self.gap(f).flatten(1)  # (B, 512)
        return self.proj(f)         # (B, 128)


class TechnicalIndicatorCalculator:
    """
    Computes strictly causal hourly technical indicators (Eq. III.4–III.6 + momentum/vol context)
    and applies trailing window Z-score standardization (`window = 120` bars).
    Produces 16 standardized numerical features per bar.
    """

    FEATURE_NAMES = (
        "macd", "macd_signal", "macd_hist", "rsi_14", "atr_14_pct",
        "ret_1h", "ret_4h", "ret_24h", "sma20_dist", "sma50_dist",
        "bb_width_20", "bb_pct_b_20", "hl_range_pct", "vol_z_ratio",
        "taker_imb", "realized_vol_24h",
    )

    def __init__(self, zscore_window: int = 120) -> None:
        self.zscore_window = zscore_window

    def compute(self, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        eps = 1e-8
        c = ohlcv_df["close"].astype(np.float64)
        h = ohlcv_df["high"].astype(np.float64)
        l = ohlcv_df["low"].astype(np.float64)
        v = ohlcv_df["volume"].astype(np.float64)

        # Eq. III.4: MACD(12, 26, 9)
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        macd = (ema12 - ema26) / (c + eps)
        macd_sig = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - macd_sig

        # Eq. III.5: Wilder RSI(14)
        delta = c.diff().fillna(0.0)
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1.0 / 14.0, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / 14.0, adjust=False).mean()
        rs = avg_gain / (avg_loss + eps)
        rsi_14 = (100.0 - (100.0 / (1.0 + rs))) / 100.0

        # Eq. III.6: Wilder ATR(14)
        prev_c = c.shift(1).fillna(c)
        tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        atr_14 = tr.ewm(alpha=1.0 / 14.0, adjust=False).mean()
        atr_14_pct = atr_14 / (c + eps)

        # Additional causal price/volume context
        log_c = np.log(c.clip(lower=eps))
        ret_1h = log_c.diff(1).fillna(0.0)
        ret_4h = log_c.diff(4).fillna(0.0)
        ret_24h = log_c.diff(24).fillna(0.0)

        sma20 = c.rolling(20, min_periods=1).mean()
        sma50 = c.rolling(50, min_periods=1).mean()
        std20 = c.rolling(20, min_periods=1).std().fillna(0.0)

        sma20_dist = (c - sma20) / (sma20 + eps)
        sma50_dist = (c - sma50) / (sma50 + eps)
        bb_width_20 = (4.0 * std20) / (sma20 + eps)
        bb_pct_b_20 = (c - (sma20 - 2.0 * std20)) / (4.0 * std20 + eps)
        hl_range_pct = (h - l) / (c + eps)
        vol_z_ratio = np.log1p(v.clip(lower=0.0)) - np.log1p(v.clip(lower=0.0)).rolling(24, min_periods=1).mean()

        if "taker_buy_base_vol" in ohlcv_df.columns:
            tb = ohlcv_df["taker_buy_base_vol"].astype(np.float64)
            taker_imb = (2.0 * tb - v) / (v + eps)
        else:
            taker_imb = pd.Series(0.0, index=ohlcv_df.index)

        realized_vol_24h = ret_1h.rolling(24, min_periods=1).std().fillna(0.0)

        raw = pd.DataFrame(
            {
                "macd": macd,
                "macd_signal": macd_sig,
                "macd_hist": macd_hist,
                "rsi_14": rsi_14,
                "atr_14_pct": atr_14_pct,
                "ret_1h": ret_1h,
                "ret_4h": ret_4h,
                "ret_24h": ret_24h,
                "sma20_dist": sma20_dist,
                "sma50_dist": sma50_dist,
                "bb_width_20": bb_width_20,
                "bb_pct_b_20": bb_pct_b_20,
                "hl_range_pct": hl_range_pct,
                "vol_z_ratio": vol_z_ratio,
                "taker_imb": taker_imb,
                "realized_vol_24h": realized_vol_24h,
            },
            index=ohlcv_df.index,
        )

        # Causal rolling Z-score standardization
        roll_mean = raw.rolling(self.zscore_window, min_periods=2).mean().fillna(raw)
        roll_std = raw.rolling(self.zscore_window, min_periods=2).std().fillna(1.0).replace(0.0, 1.0)
        z = ((raw - roll_mean) / (roll_std + 1e-6)).clip(-5.0, 5.0).fillna(0.0)
        return z.astype(np.float32)


class CrossModalStateFusion(nn.Module):
    """
    2-Token Cross-Modal Attention (Eq. III.9–III.12):
    - Query `Q`: projected from Sentiment (`32 -> 128`), shape `(B, 1, 128)`
    - Key `K` & Value `V`: token-specific projections from Visual (`128 -> 128`)
      and Technical (`64 -> 128`), stacked to `(B, 2, 128)`
    - Residual + LayerNorm: `s_t = LayerNorm(Q + A @ V)` -> `(B, 128)`
    """

    def __init__(
        self,
        d_visual: int = 128,
        d_tech: int = 64,
        d_senti: int = 32,
        d_model: int = 128,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.w_q = nn.Linear(d_senti, d_model, bias=False)
        self.w_k_vis = nn.Linear(d_visual, d_model, bias=False)
        self.w_k_tech = nn.Linear(d_tech, d_model, bias=False)
        self.w_v_vis = nn.Linear(d_visual, d_model, bias=False)
        self.w_v_tech = nn.Linear(d_tech, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward_with_weights(
        self, f_visual: torch.Tensor, f_tech: torch.Tensor, f_senti: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.w_q(f_senti).unsqueeze(1)  # (B, 1, 128)
        k = torch.stack([self.w_k_vis(f_visual), self.w_k_tech(f_tech)], dim=1)  # (B, 2, 128)
        v = torch.stack([self.w_v_vis(f_visual), self.w_v_tech(f_tech)], dim=1)  # (B, 2, 128)

        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.d_model)  # (B, 1, 2)
        attn_weights = F.softmax(scores, dim=-1)                            # (B, 1, 2), sum == 1
        h_cross = torch.bmm(attn_weights, v)                                # (B, 1, 128)
        s_t = self.norm(q + h_cross).squeeze(1)                             # (B, 128)
        return s_t, attn_weights.squeeze(1)

    def forward(
        self, f_visual: torch.Tensor, f_tech: torch.Tensor, f_senti: torch.Tensor
    ) -> torch.Tensor:
        s_t, _ = self.forward_with_weights(f_visual, f_tech, f_senti)
        return s_t


class MultimodalStateEncoder(nn.Module):
    """
    Unified Study 1 Encoder producing `s_t ∈ R^128` (plus individual modality embeddings
    `f_visual ∈ R^128`, `f_tech ∈ R^64`, `f_senti ∈ R^32` for KernelSHAP Eq. VI.12).
    """

    def __init__(
        self,
        tech_in_dim: int = 16,
        senti_in_dim: int = 8,
        d_visual: int = 128,
        d_tech: int = 64,
        d_senti: int = 32,
        d_state: int = 128,
        pretrained_resnet: bool = False,
    ) -> None:
        super().__init__()
        self.visual_encoder = VisualResNetCBAMEncoder(out_dim=d_visual, pretrained=pretrained_resnet)
        self.tech_encoder = nn.Sequential(
            nn.Linear(tech_in_dim, d_tech),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_tech),
        )
        self.senti_encoder = nn.Sequential(
            nn.Linear(senti_in_dim, d_senti),
            nn.GELU(),
            nn.LayerNorm(d_senti),
        )
        self.fusion = CrossModalStateFusion(
            d_visual=d_visual, d_tech=d_tech, d_senti=d_senti, d_model=d_state
        )
        self.aux_regime_head = nn.Linear(d_state, 3)

    def extract_modalities(
        self, chart_imgs: torch.Tensor, tech_x: torch.Tensor, senti_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f_vis = self.visual_encoder(chart_imgs)
        f_tech = self.tech_encoder(tech_x)
        f_senti = self.senti_encoder(senti_x)
        return f_vis, f_tech, f_senti

    def forward(
        self, chart_imgs: torch.Tensor, tech_x: torch.Tensor, senti_x: torch.Tensor
    ) -> torch.Tensor:
        f_vis, f_tech, f_senti = self.extract_modalities(chart_imgs, tech_x, senti_x)
        return self.fusion(f_vis, f_tech, f_senti)

    def pretrain_on_fold_regimes(
        self,
        chart_imgs: np.ndarray,
        tech_x: np.ndarray,
        senti_x: np.ndarray,
        regime_labels: np.ndarray,
        epochs: int = 5,
        batch_size: int = 64,
        lr: float = 3e-4,
        device: Optional[torch.device] = None,
    ) -> list[float]:
        """
        Fold-local auxiliary pretraining (`구현계획_261111.md` Sec 4 #9 & Sec 6.3 WP2):
        Trains encoder + fusion parameters strictly on the training fold's regime labels,
        then freezes the encoder prior to XGBoost and PPO training.
        """
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(dev)
        self.train()
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)

        valid_mask = regime_labels >= 0
        idx_all = np.where(valid_mask)[0]
        losses: list[float] = []
        if len(idx_all) == 0:
            return losses

        rng = np.random.default_rng(42)
        for _ in range(epochs):
            perm = rng.permutation(idx_all)
            ep_loss = 0.0
            n_batches = 0
            for s in range(0, len(perm), batch_size):
                b_idx = perm[s : s + batch_size]
                imgs_t = torch.tensor(chart_imgs[b_idx], dtype=torch.float32, device=dev)
                tech_t = torch.tensor(tech_x[b_idx], dtype=torch.float32, device=dev)
                senti_t = torch.tensor(senti_x[b_idx], dtype=torch.float32, device=dev)
                y_t = torch.tensor(regime_labels[b_idx], dtype=torch.long, device=dev)

                opt.zero_grad()
                state = self.forward(imgs_t, tech_t, senti_t)
                logits = self.aux_regime_head(state)
                loss = F.cross_entropy(logits, y_t)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                opt.step()

                ep_loss += float(loss.item())
                n_batches += 1
            losses.append(ep_loss / max(n_batches, 1))

        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)
        return losses

    @torch.no_grad()
    def encode_dataset(
        self,
        chart_imgs: np.ndarray,
        tech_x: np.ndarray,
        senti_x: np.ndarray,
        batch_size: int = 128,
        device: Optional[torch.device] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch-encode `(N, 128)` fused states `s_t` and `(N, 224)` concatenated modality vectors
        `[f_vis(128), f_tech(64), f_senti(32)]` (used for SHAP attribution Eq. VI.12).
        """
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(dev)
        self.eval()

        n = len(tech_x)
        states = np.empty((n, self.fusion.d_model), dtype=np.float32)
        concat_mods = np.empty((n, 128 + 64 + 32), dtype=np.float32)

        for s in range(0, n, batch_size):
            e = min(n, s + batch_size)
            imgs_t = torch.tensor(chart_imgs[s:e], dtype=torch.float32, device=dev)
            tech_t = torch.tensor(tech_x[s:e], dtype=torch.float32, device=dev)
            senti_t = torch.tensor(senti_x[s:e], dtype=torch.float32, device=dev)

            f_vis, f_tech, f_senti = self.extract_modalities(imgs_t, tech_t, senti_t)
            st = self.fusion(f_vis, f_tech, f_senti)
            states[s:e] = st.cpu().numpy()
            concat_mods[s:e] = torch.cat([f_vis, f_tech, f_senti], dim=-1).cpu().numpy()

        return states, concat_mods
