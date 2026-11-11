"""
XGBoost Market Regime Classifier with Confidence Hysteresis Gate (`θ = 0.60`).

Implements Chapter III Sec 2.2–2.3 (Eq. III.14–III.16) & Table A.2:
- `n_estimators = 300`, `max_depth = 6`, `learning_rate = 0.05`
- Inverse-frequency sample weights to handle class imbalance across regimes
- Sequential hysteresis transition (Eq. III.16):
    `R_t = argmax_R P(R | s_t)` if `max_R P(R | s_t) >= 0.60`, else `R_{t-1}`
  Initialized to `REGIME_SIDEWAYS` (`2`).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import xgboost as xgb

from ts.study1.regime_labeling import REGIME_SIDEWAYS


class XGBoostRegimeClassifier:
    """3-class XGBoost classifier + `theta = 0.60` confidence hysteresis gate."""

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        switch_threshold: float = 0.60,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.switch_threshold = switch_threshold
        self.random_state = random_state
        self.model: Optional[xgb.XGBClassifier] = None
        self._constant_fallback_class: Optional[int] = None

    def fit(self, states_x: np.ndarray, labels_y: np.ndarray) -> "XGBoostRegimeClassifier":
        mask = labels_y >= 0
        X = np.asarray(states_x[mask], dtype=np.float32)
        y = np.asarray(labels_y[mask], dtype=np.int64)
        if len(y) == 0:
            self._constant_fallback_class = REGIME_SIDEWAYS
            return self

        present_classes = np.unique(y)
        if len(present_classes) < 3:
            # Ensure all 3 classes exist for `multi:softprob` even on tiny unit-test slices
            missing = [c for c in (0, 1, 2) if c not in present_classes]
            pad_X = np.repeat(X[:1], len(missing), axis=0)
            pad_y = np.array(missing, dtype=np.int64)
            X = np.concatenate([X, pad_X], axis=0)
            y = np.concatenate([y, pad_y], axis=0)

        counts = np.bincount(y, minlength=3).astype(np.float64)
        inv_freq = len(y) / (3.0 * np.maximum(counts, 1.0))
        sample_weights = inv_freq[y]

        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective="multi:softprob",
            num_class=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=self.random_state,
            n_jobs=4,
            eval_metric="mlogloss",
            verbosity=0,
        )
        self.model.fit(X, y, sample_weight=sample_weights)
        self._constant_fallback_class = None
        return self

    def predict_proba(self, states_x: np.ndarray) -> np.ndarray:
        X = np.asarray(states_x, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        if self.model is None:
            probs = np.full((len(X), 3), 1.0 / 3.0, dtype=np.float32)
            if self._constant_fallback_class is not None:
                probs[:] = 0.1
                probs[:, self._constant_fallback_class] = 0.8
            return probs
        return np.asarray(self.model.predict_proba(X), dtype=np.float32)

    @staticmethod
    def apply_hysteresis_gate(
        probs_n3: np.ndarray,
        threshold: float = 0.60,
        initial_regime: int = REGIME_SIDEWAYS,
    ) -> np.ndarray:
        """
        Exact Eq. (III.16):
          R_t = argmax_R P(R | s_t) if max_R P(R | s_t) >= threshold else R_{t-1}
        """
        n = len(probs_n3)
        regimes = np.empty(n, dtype=np.int64)
        cur = int(initial_regime)
        for t in range(n):
            p = probs_n3[t]
            best_r = int(np.argmax(p))
            if float(p[best_r]) >= threshold:
                cur = best_r
            regimes[t] = cur
        return regimes

    def predict_regimes_causal(
        self, states_x: np.ndarray, initial_regime: int = REGIME_SIDEWAYS
    ) -> tuple[np.ndarray, np.ndarray]:
        probs = self.predict_proba(states_x)
        regimes = self.apply_hysteresis_gate(
            probs, threshold=self.switch_threshold, initial_regime=initial_regime
        )
        return regimes, probs
