"""
Study 1 multimodal feature encoders and 2-token cross-modal attention (Chapter III Sec 1).

Implements the equations in the revised thesis:
- Visual (`128-d`): ImageNet-pretrained ResNet-18 + CBAM (`r=16`, `7x7` spatial conv, Eq. III.2-III.3, Table 3.1)
- Technical (`64-d`): MACD(12,26,9), RSI(14), Wilder ATR(14) and causal rolling Z-score indicators (Eq. III.4-III.6)
- Sentiment (`32-d`): 24h EWMA (`alpha=0.08`, Eq. III.7-III.8) and 24h window statistics -> Linear + LayerNorm
- Cross-Modal Fusion (`128-d`): Sentiment = Query (`1x128`), `[Visual, Tech]` = Key/Value (`2x128`),
  `s_t = LayerNorm(Q + softmax(Q K^T / sqrt(128)) V)` (Eq. III.9-III.12)
- Supports controlled modality ablation (`modality_mode` in `{"multimodal", "tech_only", "visual_tech_no_senti"}`)
  so Variant A keeps the downstream regime classifier and ensemble architecture unchanged.
"""

from __future__ import annotations

import logging
import math
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from ts.study1.chart_renderer import normalize_chart_batch_to_tensor

logger = logging.getLogger(__name__)

ModalityMode = Literal["multimodal", "tech_only", "visual_tech_no_senti"]


class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module (Woo et al.; Thesis Eq. III.2-III.3).
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
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.pretrained = bool(pretrained)
        weights = ResNet18_Weights.IMAGENET1K_V1 if self.pretrained else None
        self.pretrained_weights_name = "IMAGENET1K_V1" if self.pretrained else "none"
        backbone = resnet18(weights=weights)
        if self.pretrained:
            msg = f"[Study 1] Loaded ResNet-18 pretrained weights ({self.pretrained_weights_name})."
            logger.info(msg)
            print(msg)
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
        f = self.stem(img)
        f = self.layer4(self.layer3(self.layer2(self.layer1(f))))
        f = self.cbam(f)
        f = self.gap(f).flatten(1)
        return self.proj(f)


class TechnicalIndicatorCalculator:
    """
    Computes causal hourly technical indicators (Eq. III.4-III.6 and momentum/volatility context)
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

        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        macd = (ema12 - ema26) / (c + eps)
        macd_sig = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - macd_sig

        delta = c.diff().fillna(0.0)
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1.0 / 14.0, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / 14.0, adjust=False).mean()
        rs = avg_gain / (avg_loss + eps)
        rsi_14 = (100.0 - (100.0 / (1.0 + rs))) / 100.0

        prev_c = c.shift(1).fillna(c)
        tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        atr_14 = tr.ewm(alpha=1.0 / 14.0, adjust=False).mean()
        atr_14_pct = atr_14 / (c + eps)

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

        roll_mean = raw.rolling(self.zscore_window, min_periods=2).mean().fillna(raw)
        roll_std = raw.rolling(self.zscore_window, min_periods=2).std().fillna(1.0).replace(0.0, 1.0)
        z = ((raw - roll_mean) / (roll_std + 1e-6)).clip(-5.0, 5.0).fillna(0.0)
        return z.astype(np.float32)


class CrossModalStateFusion(nn.Module):
    """
    2-Token Cross-Modal Attention (Eq. III.9-III.12):
    - Query `Q`: projected from Sentiment (`32 -> 128`), shape `(B, 1, 128)`
    - Key `K` and Value `V`: token-specific projections from Visual (`128 -> 128`)
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
        q = self.w_q(f_senti).unsqueeze(1)
        k = torch.stack([self.w_k_vis(f_visual), self.w_k_tech(f_tech)], dim=1)
        v = torch.stack([self.w_v_vis(f_visual), self.w_v_tech(f_tech)], dim=1)

        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.d_model)
        attn_weights = F.softmax(scores, dim=-1)
        h_cross = torch.bmm(attn_weights, v)
        s_t = self.norm(q + h_cross).squeeze(1)
        return s_t, attn_weights.squeeze(1)

    def forward(
        self, f_visual: torch.Tensor, f_tech: torch.Tensor, f_senti: torch.Tensor
    ) -> torch.Tensor:
        s_t, _ = self.forward_with_weights(f_visual, f_tech, f_senti)
        return s_t


class MultimodalStateEncoder(nn.Module):
    """
    Study 1 encoder producing `s_t` in `R^128` alongside individual modality vectors
    `f_visual` in `R^128`, `f_tech` in `R^64`, and `f_senti` in `R^32`.

    `modality_mode`:
      - `"multimodal"`: Chart image + technical indicators + news sentiment with 2-token cross-modal fusion (Variant D).
      - `"tech_only"`: Single-modality technical indicator projection `16 -> 64 -> 128` (Variant A default).
      - `"visual_tech_no_senti"`: Chart image + technical indicators without news sentiment (`192 -> 128`).
    """

    def __init__(
        self,
        tech_in_dim: int = 16,
        senti_in_dim: int = 8,
        d_visual: int = 128,
        d_tech: int = 64,
        d_senti: int = 32,
        d_state: int = 128,
        pretrained_resnet: bool = True,
        modality_mode: ModalityMode = "multimodal",
    ) -> None:
        super().__init__()
        if modality_mode not in ("multimodal", "tech_only", "visual_tech_no_senti"):
            raise ValueError(f"Unsupported modality_mode: {modality_mode}")
        self.d_visual = d_visual
        self.d_tech = d_tech
        self.d_senti = d_senti
        self.d_state = d_state
        self.pretrained_resnet = bool(pretrained_resnet)
        self.modality_mode: ModalityMode = modality_mode

        self.tech_encoder = nn.Sequential(
            nn.Linear(tech_in_dim, d_tech),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_tech),
        )

        if self.modality_mode in ("multimodal", "visual_tech_no_senti"):
            self.visual_encoder: Optional[VisualResNetCBAMEncoder] = VisualResNetCBAMEncoder(
                out_dim=d_visual, pretrained=self.pretrained_resnet
            )
        else:
            self.visual_encoder = None

        if self.modality_mode == "multimodal":
            self.senti_encoder: Optional[nn.Module] = nn.Sequential(
                nn.Linear(senti_in_dim, d_senti),
                nn.GELU(),
                nn.LayerNorm(d_senti),
            )
            self.fusion: Optional[CrossModalStateFusion] = CrossModalStateFusion(
                d_visual=d_visual, d_tech=d_tech, d_senti=d_senti, d_model=d_state
            )
            self.ablation_proj: Optional[nn.Module] = None
        elif self.modality_mode == "tech_only":
            self.senti_encoder = None
            self.fusion = None
            self.ablation_proj = nn.Sequential(
                nn.Linear(d_tech, d_state),
                nn.LayerNorm(d_state),
            )
        else:
            self.senti_encoder = None
            self.fusion = None
            self.ablation_proj = nn.Sequential(
                nn.Linear(d_visual + d_tech, d_state),
                nn.ReLU(inplace=True),
                nn.LayerNorm(d_state),
            )

        self.aux_regime_head = nn.Linear(d_state, 3)

    def extract_modalities(
        self,
        chart_imgs: Optional[torch.Tensor],
        tech_x: torch.Tensor,
        senti_x: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = tech_x.shape[0]
        dev = tech_x.device
        f_tech = self.tech_encoder(tech_x)

        if self.visual_encoder is not None:
            if chart_imgs is None:
                raise ValueError("chart_imgs is required when visual modality is active.")
            f_vis = self.visual_encoder(chart_imgs)
        else:
            f_vis = torch.zeros((b, self.d_visual), dtype=tech_x.dtype, device=dev)

        if self.senti_encoder is not None:
            if senti_x is None:
                raise ValueError("senti_x is required when sentiment modality is active.")
            f_senti = self.senti_encoder(senti_x)
        else:
            f_senti = torch.zeros((b, self.d_senti), dtype=tech_x.dtype, device=dev)

        return f_vis, f_tech, f_senti

    def fuse_modalities(
        self, f_vis: torch.Tensor, f_tech: torch.Tensor, f_senti: torch.Tensor
    ) -> torch.Tensor:
        if self.modality_mode == "multimodal":
            assert self.fusion is not None
            return self.fusion(f_vis, f_tech, f_senti)
        if self.modality_mode == "tech_only":
            assert self.ablation_proj is not None
            return self.ablation_proj(f_tech)
        assert self.ablation_proj is not None
        return self.ablation_proj(torch.cat([f_vis, f_tech], dim=-1))

    def forward(
        self,
        chart_imgs: Optional[torch.Tensor],
        tech_x: torch.Tensor,
        senti_x: Optional[torch.Tensor],
    ) -> torch.Tensor:
        f_vis, f_tech, f_senti = self.extract_modalities(chart_imgs, tech_x, senti_x)
        return self.fuse_modalities(f_vis, f_tech, f_senti)

    def pretrain_on_fold_regimes(
        self,
        chart_imgs: Optional[np.ndarray],
        tech_x: np.ndarray,
        senti_x: Optional[np.ndarray],
        regime_labels: np.ndarray,
        epochs: int = 5,
        batch_size: int = 64,
        lr: float = 3e-4,
        device: Optional[torch.device] = None,
        indices: Optional[np.ndarray] = None,
    ) -> list[float]:
        """
        Fold-local auxiliary pretraining: trains encoder parameters strictly on the training
        fold's purged regime labels, then freezes the encoder prior to XGBoost and PPO.
        """
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(dev)
        self.train()
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)

        if indices is not None:
            fold_idx = np.asarray(indices, dtype=np.int64)
            fold_labels = (
                regime_labels
                if len(regime_labels) == len(fold_idx)
                else regime_labels[fold_idx]
            )
        else:
            fold_idx = None
            fold_labels = regime_labels

        valid_local = np.where(fold_labels >= 0)[0]
        losses: list[float] = []
        if len(valid_local) == 0:
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)
            return losses

        uses_visual = self.visual_encoder is not None
        uses_senti = self.senti_encoder is not None

        rng = np.random.default_rng(42)
        for _ in range(epochs):
            perm = rng.permutation(valid_local)
            ep_loss = 0.0
            n_batches = 0
            for s in range(0, len(perm), batch_size):
                b_local = perm[s : s + batch_size]
                b_src = fold_idx[b_local] if fold_idx is not None else b_local

                if uses_visual:
                    if chart_imgs is None:
                        raise ValueError("chart_imgs cannot be None when visual modality is active.")
                    imgs_t: Optional[torch.Tensor] = normalize_chart_batch_to_tensor(chart_imgs[b_src], dev)
                else:
                    imgs_t = None

                tech_t = torch.tensor(tech_x[b_src], dtype=torch.float32, device=dev)

                if uses_senti:
                    if senti_x is None:
                        raise ValueError("senti_x cannot be None when sentiment modality is active.")
                    senti_t: Optional[torch.Tensor] = torch.tensor(senti_x[b_src], dtype=torch.float32, device=dev)
                else:
                    senti_t = None

                y_t = torch.tensor(fold_labels[b_local], dtype=torch.long, device=dev)

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
        chart_imgs: Optional[np.ndarray],
        tech_x: np.ndarray,
        senti_x: Optional[np.ndarray],
        batch_size: int = 128,
        device: Optional[torch.device] = None,
        indices: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch-encode `(N, 128)` states `s_t` and `(N, 224)` concatenated modality vectors
        `[f_vis(128), f_tech(64), f_senti(32)]` with minibatch-level uint8 normalization.
        """
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(dev)
        self.eval()

        fold_idx = np.asarray(indices, dtype=np.int64) if indices is not None else None
        n = len(fold_idx) if fold_idx is not None else len(tech_x)
        states = np.empty((n, self.d_state), dtype=np.float32)
        concat_mods = np.empty((n, self.d_visual + self.d_tech + self.d_senti), dtype=np.float32)

        uses_visual = self.visual_encoder is not None
        uses_senti = self.senti_encoder is not None

        for s in range(0, n, batch_size):
            e = min(n, s + batch_size)
            b_src = fold_idx[s:e] if fold_idx is not None else slice(s, e)

            if uses_visual:
                if chart_imgs is None:
                    raise ValueError("chart_imgs cannot be None when visual modality is active.")
                imgs_t: Optional[torch.Tensor] = normalize_chart_batch_to_tensor(chart_imgs[b_src], dev)
            else:
                imgs_t = None

            tech_t = torch.tensor(tech_x[b_src], dtype=torch.float32, device=dev)

            if uses_senti:
                if senti_x is None:
                    raise ValueError("senti_x cannot be None when sentiment modality is active.")
                senti_t: Optional[torch.Tensor] = torch.tensor(senti_x[b_src], dtype=torch.float32, device=dev)
            else:
                senti_t = None

            f_vis, f_tech, f_senti = self.extract_modalities(imgs_t, tech_t, senti_t)
            st = self.fuse_modalities(f_vis, f_tech, f_senti)
            states[s:e] = st.cpu().numpy()
            concat_mods[s:e] = torch.cat([f_vis, f_tech, f_senti], dim=-1).cpu().numpy()

        return states, concat_mods
