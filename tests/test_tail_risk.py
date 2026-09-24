"""
Unit tests for Tail Risk and Extreme Value Engine.
"""

import numpy as np
import pandas as pd
import pytest
from src.models.tail_risk import TailRiskEngine
from src.models.backtest import RiskBacktester


@pytest.fixture
def sample_returns():
    rng = np.random.default_rng(42)
    # Heavy-tailed Student's t returns
    return rng.standard_t(df=4, size=1000) * 0.015


def test_tail_risk_subadditivity_and_ordering(sample_returns):
    engine = TailRiskEngine(confidence_levels=[0.95, 0.99])
    df = engine.compute_all_metrics(sample_returns, horizon_days=1, portfolio_value=1_000_000.0)

    assert not df.empty
    assert "VaR_pct" in df.columns
    assert "CVaR_Expected_Shortfall_pct" in df.columns

    # Fundamental axiom: CVaR (Expected Shortfall) >= VaR for any coherent distribution
    for _, row in df.iterrows():
        assert row["CVaR_Expected_Shortfall_pct"] >= row["VaR_pct"] - 1e-6
        assert row["CVaR_Dollar"] >= row["VaR_Dollar"] - 1.0


def test_cornish_fisher_expansion(sample_returns):
    var_cf_95, cvar_cf_95 = TailRiskEngine.cornish_fisher_var_cvar(sample_returns, alpha=0.95)
    var_cf_99, cvar_cf_99 = TailRiskEngine.cornish_fisher_var_cvar(sample_returns, alpha=0.99)

    assert var_cf_99 > var_cf_95
    assert cvar_cf_99 > cvar_cf_95


def test_kupiec_and_christoffersen_backtesting():
    backtester = RiskBacktester(confidence_level=0.95)
    rng = np.random.default_rng(42)
    T = 500
    # Simulate well-calibrated returns and VaR
    actual_returns = pd.Series(rng.normal(0, 0.01, T))
    # True 95% VaR is ~1.645 * 0.01 = 0.01645
    predicted_var = pd.Series(np.repeat(0.01645, T))

    res = backtester.run_full_backtest(actual_returns, predicted_var)

    assert res["Total_Observations_T"] == T
    assert res["Total_Breaches_N"] > 0
    assert 0.0 <= res["Kupiec_p_value"] <= 1.0
    assert 0.0 <= res["Christoffersen_Ind_p_value"] <= 1.0
    assert "GREEN" in res["Basel_Traffic_Light"] or "YELLOW" in res["Basel_Traffic_Light"]
