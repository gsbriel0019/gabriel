"""
Market Regime Detection Engine.
Implements unsupervised Hidden Markov Models (HMM) and Gaussian Mixture Models (GMM)
with Information Criteria (AIC/BIC) and volatility-sorted state interpretation.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from hmmlearn.hmm import GaussianHMM


class MarketRegimeDetector:
    """
    Unsupervised Market Regime Classifier.
    Sorts latent states by empirical volatility to ensure deterministic interpretations:
    - Regime 0: Low Volatility (Bullish / Steady Growth)
    - Regime 1: Moderate Volatility (Rangebound / Mean-Reverting)
    - Regime 2: High Volatility (Market Stress / Crisis / Crash)
    """

    REGIME_NAMES = {
        0: "Low Volatility (Bullish / Expansion)",
        1: "Moderate Volatility (Transitional / Rangebound)",
        2: "High Volatility (High Risk / Market Stress)",
    }

    REGIME_COLORS = {
        0: "#22c55e",  # Emerald Green
        1: "#eab308",  # Amber Yellow
        2: "#ef4444",  # Crimson Red
    }

    def __init__(
        self,
        n_regimes: int = 3,
        model_type: str = "HMM",
        covariance_type: str = "full",
        n_iter: int = 500,
        random_state: int = 42,
    ):
        self.n_regimes = n_regimes
        self.model_type = model_type.upper()
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state

        self.model: Optional[Union[GaussianHMM, GaussianMixture]] = None
        self.state_order_map: Dict[int, int] = {}
        self.fitted_: bool = False

    def fit(self, features: Union[pd.DataFrame, np.ndarray]) -> "MarketRegimeDetector":
        """
        Fit HMM or GMM onto feature matrix (typically [Log Returns, Realized Volatility]).
        """
        X = np.asarray(features)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        if self.model_type == "HMM":
            self.model = GaussianHMM(
                n_components=self.n_regimes,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=self.random_state,
            )
            self.model.fit(X)
            raw_states = self.model.predict(X)
        elif self.model_type == "GMM":
            self.model = GaussianMixture(
                n_components=self.n_regimes,
                covariance_type=self.covariance_type,
                max_iter=self.n_iter,
                random_state=self.random_state,
            )
            self.model.fit(X)
            raw_states = self.model.predict(X)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}. Choose 'HMM' or 'GMM'.")

        # Map raw unsupervised cluster IDs to volatility-ordered regimes
        vol_per_state = []
        for state in range(self.n_regimes):
            mask = raw_states == state
            state_vol = np.std(X[mask, 0]) if np.sum(mask) > 1 else 0.0
            vol_per_state.append((state, state_vol))

        # Sort ascending by volatility
        sorted_states = sorted(vol_per_state, key=lambda x: x[1])
        self.state_order_map = {old_state: new_order for new_order, (old_state, _) in enumerate(sorted_states)}
        self.fitted_ = True
        return self

    def predict(self, features: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict market regimes for given features with standardized volatility order.
        """
        if not self.fitted_ or self.model is None:
            raise RuntimeError("Model must be fitted before calling predict.")

        X = np.asarray(features)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        raw_preds = self.model.predict(X)
        ordered_preds = np.array([self.state_order_map[s] for s in raw_preds])
        return ordered_preds

    def predict_proba(self, features: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Return posterior regime probability distributions P(S_t = k | X_t).
        """
        if not self.fitted_ or self.model is None:
            raise RuntimeError("Model must be fitted before calling predict_proba.")

        X = np.asarray(features)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        if self.model_type == "HMM":
            raw_proba = self.model.predict_proba(X)
        else:
            raw_proba = self.model.predict_proba(X)

        ordered_proba = np.zeros_like(raw_proba)
        for old_state, new_state in self.state_order_map.items():
            ordered_proba[:, new_state] = raw_proba[:, old_state]

        return ordered_proba

    def get_transition_matrix(self) -> Optional[pd.DataFrame]:
        """
        Retrieve reordered state transition probability matrix P:
        P_{i,j} = Probability of transitioning from Regime i to Regime j.
        """
        if not self.fitted_ or self.model is None or self.model_type != "HMM":
            return None

        raw_transmat = self.model.transmat_
        n = self.n_regimes
        ordered_transmat = np.zeros((n, n))

        for old_i, new_i in self.state_order_map.items():
            for old_j, new_j in self.state_order_map.items():
                ordered_transmat[new_i, new_j] = raw_transmat[old_i, old_j]

        labels = [f"Regime {i}: {self.REGIME_NAMES.get(i, str(i))}" for i in range(n)]
        return pd.DataFrame(ordered_transmat, index=labels, columns=labels)

    def calculate_expected_durations(self) -> Dict[str, float]:
        """
        Calculate expected duration (persistence in trading days) for each regime:
        E[D_i] = 1 / (1 - P_{i,i})
        """
        trans_df = self.get_transition_matrix()
        if trans_df is None:
            return {}

        durations = {}
        for i in range(self.n_regimes):
            p_ii = trans_df.iloc[i, i]
            dur = 1.0 / (1.0 - p_ii) if p_ii < 1.0 else np.nan
            durations[f"Regime {i}"] = float(np.round(dur, 1))

        return durations

    @staticmethod
    def select_optimal_regimes(
        features: Union[pd.DataFrame, np.ndarray], max_regimes: int = 5
    ) -> pd.DataFrame:
        """
        Compare models across k in [2, max_regimes] using AIC and BIC for statistical model validation.
        """
        X = np.asarray(features)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        results = []
        for k in range(2, max_regimes + 1):
            gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42)
            gmm.fit(X)
            results.append({
                "n_regimes": k,
                "AIC": gmm.aic(X),
                "BIC": gmm.bic(X),
                "Log_Likelihood": gmm.score(X) * len(X),
            })

        return pd.DataFrame(results).set_index("n_regimes")
