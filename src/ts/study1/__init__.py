"""Study 1 (Ch. III): Market Regime-Adaptive Multimodal Ensemble RL."""

from ts.study1.chart_renderer import CandlestickChartRenderer
from ts.study1.encoders import (
    CBAMBlock,
    CrossModalStateFusion,
    MultimodalStateEncoder,
    TechnicalIndicatorCalculator,
    VisualResNetCBAMEncoder,
)
from ts.study1.ensemble import DynamicRegimeEnsemble
from ts.study1.per_buffer import SegmentTreePERBuffer
from ts.study1.ppo_expert import RegimePPOExpert, compute_regime_rewards
from ts.study1.regime_classifier import XGBoostRegimeClassifier
from ts.study1.regime_labeling import (
    REGIME_BEAR,
    REGIME_BULL,
    REGIME_NAMES,
    REGIME_SIDEWAYS,
    SMARegimeLabeler,
    TrendScanningLabeler,
)

__all__ = [
    "CandlestickChartRenderer",
    "CBAMBlock",
    "CrossModalStateFusion",
    "MultimodalStateEncoder",
    "TechnicalIndicatorCalculator",
    "VisualResNetCBAMEncoder",
    "DynamicRegimeEnsemble",
    "SegmentTreePERBuffer",
    "RegimePPOExpert",
    "compute_regime_rewards",
    "XGBoostRegimeClassifier",
    "REGIME_BULL",
    "REGIME_BEAR",
    "REGIME_SIDEWAYS",
    "REGIME_NAMES",
    "SMARegimeLabeler",
    "TrendScanningLabeler",
]
