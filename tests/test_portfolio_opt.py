"""
Unit tests for Portfolio Optimization (HRP & Markowitz) and Stress-Testing modules.
"""

import numpy as np
import pandas as pd
import pytest
from src.models.portfolio_opt import PortfolioOptimizer
from src.models.stress_testing import StressTestEngine


@pytest.fixture
def asset_returns():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=250, freq="B")
    data = {
        "SPY": rng.normal(0.0004, 0.012, 250),
        "QQQ": rng.normal(0.0006, 0.016, 250),
        "TLT": rng.normal(0.0001, 0.008, 250),
        "GLD": rng.normal(0.0002, 0.010, 250),
    }
    return pd.DataFrame(data, index=dates)


def test_hrp_weights_and_constraints(asset_returns):
    opt = PortfolioOptimizer(risk_free_rate=0.045)
    weights = opt.optimize_hrp(asset_returns)

    assert isinstance(weights, dict)
    assert len(weights) == 4
    # Weights must sum to 1.0 (long-only budget constraint)
    assert np.isclose(sum(weights.values()), 1.0)
    # Long-only: all weights >= 0
    for w in weights.values():
        assert w >= 0.0


def test_min_volatility_and_max_sharpe(asset_returns):
    opt = PortfolioOptimizer(risk_free_rate=0.045)
    min_vol_w = opt.optimize_min_volatility(asset_returns)
    max_shp_w = opt.optimize_max_sharpe(asset_returns)

    assert np.isclose(sum(min_vol_w.values()), 1.0)
    assert np.isclose(sum(max_shp_w.values()), 1.0)

    metrics_min = opt.compute_portfolio_metrics(min_vol_w, asset_returns)
    metrics_shp = opt.compute_portfolio_metrics(max_shp_w, asset_returns)

    assert metrics_min["Annualized_Volatility"] > 0
    assert "Sharpe_Ratio" in metrics_shp


def test_correlated_monte_carlo_stress_testing(asset_returns):
    stress = StressTestEngine(random_state=42)
    weights = {"SPY": 0.4, "QQQ": 0.2, "TLT": 0.2, "GLD": 0.2}

    res = stress.run_correlated_monte_carlo(
        asset_returns, weights, n_simulations=500, horizon_days=20, initial_portfolio_value=1_000_000.0
    )

    wealth_paths = res["wealth_paths"]
    assert wealth_paths.shape == (500, 21)
    assert (wealth_paths > 0).all()
    assert res["var_95_dollar"] > 0
    assert not res["percentiles_summary"].empty
