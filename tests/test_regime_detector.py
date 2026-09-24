"""
Unit tests for Market Regime Detection module (HMM & GMM).
"""

import numpy as np
import pandas as pd
import pytest
from src.models.regime_detector import MarketRegimeDetector


@pytest.fixture
def sample_features():
    rng = np.random.default_rng(42)
    # Generate 2 regimes: Regime 0 low vol, Regime 1 high vol
    r0 = rng.normal(0.001, 0.005, (300, 2))
    r1 = rng.normal(-0.002, 0.025, (300, 2))
    data = np.vstack([r0, r1])
    return pd.DataFrame(data, columns=["log_return", "vol"])


def test_gmm_regime_fitting(sample_features):
    detector = MarketRegimeDetector(n_regimes=2, model_type="GMM", random_state=42)
    detector.fit(sample_features)
    preds = detector.predict(sample_features)

    assert detector.fitted_
    assert len(preds) == len(sample_features)
    assert set(np.unique(preds)).issubset({0, 1})

    # Regime 0 must have lower volatility than Regime 1
    vol_0 = np.std(sample_features.iloc[preds == 0, 0])
    vol_1 = np.std(sample_features.iloc[preds == 1, 0])
    assert vol_0 <= vol_1


def test_hmm_regime_fitting_and_transition_matrix(sample_features):
    detector = MarketRegimeDetector(n_regimes=2, model_type="HMM", random_state=42)
    detector.fit(sample_features)
    preds = detector.predict(sample_features)
    proba = detector.predict_proba(sample_features)

    assert len(preds) == len(sample_features)
    assert proba.shape == (len(sample_features), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)

    trans_mat = detector.get_transition_matrix()
    assert trans_mat is not None
    assert trans_mat.shape == (2, 2)
    # Each row of transition matrix must sum to 1.0
    assert np.allclose(trans_mat.sum(axis=1), 1.0)

    durations = detector.calculate_expected_durations()
    assert len(durations) == 2
    for dur in durations.values():
        assert dur >= 1.0


def test_optimal_regime_selection(sample_features):
    ic_df = MarketRegimeDetector.select_optimal_regimes(sample_features, max_regimes=4)
    assert len(ic_df) == 3  # k = 2, 3, 4
    assert "AIC" in ic_df.columns
    assert "BIC" in ic_df.columns
