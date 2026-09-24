"""
Unit tests for ExecutiveRiskReporter and Historical Stress Testing.
"""

import numpy as np
import pandas as pd
import pytest
from src.reporting.risk_report import ExecutiveRiskReporter
from src.models.stress_testing import StressTestEngine


def test_executive_risk_reporter_generation():
    reporter = ExecutiveRiskReporter(author="Gabriel Proaño", institution="AlphaRisk Analytics")

    tail_df = pd.DataFrame([
        {
            "Confidence_Level": "95%",
            "Method": "Historical Simulation",
            "VaR_pct": 0.015,
            "VaR_Dollar": 15000.0,
            "CVaR_Expected_Shortfall_pct": 0.022,
            "CVaR_Dollar": 22000.0,
        }
    ])

    backtest_results = {
        "Basel_Traffic_Light": "GREEN (Acceptable)",
        "Total_Observations_T": 1000,
        "Total_Breaches_N": 48,
        "Expected_Breaches": 50.0,
        "Empirical_Failure_Rate": 0.048,
        "Expected_Failure_Rate": 0.05,
        "Kupiec_LR_Stat": 0.082,
        "Kupiec_p_value": 0.774,
        "Kupiec_Null_Accepted": True,
        "Christoffersen_Ind_LR_Stat": 0.12,
        "Christoffersen_Ind_p_value": 0.72,
        "Independence_Null_Accepted": True,
    }

    port_metrics = {
        "Annualized_Return": 0.12,
        "Annualized_Volatility": 0.14,
        "Sharpe_Ratio": 0.55,
        "Max_Drawdown": -0.18,
    }

    stress_engine = StressTestEngine(random_state=42)
    weights = {"SPY": 0.6, "TLT": 0.4}
    stress_df = stress_engine.simulate_historical_crisis(weights, portfolio_value=1_000_000.0)

    report_md = reporter.generate_markdown_report(
        current_regime="Low Volatility (Bullish / Expansion)",
        regime_durations={"Regime 0": 45.0, "Regime 2": 25.0},
        tail_risk_df=tail_df,
        backtest_results=backtest_results,
        portfolio_metrics=port_metrics,
        portfolio_weights=weights,
        stress_df=stress_df,
        portfolio_value=1_000_000.0,
    )

    assert "Gabriel Proaño" in report_md
    assert "AlphaRisk Analytics" in report_md
    assert "GREEN (Acceptable)" in report_md
    assert "Hierarchical Risk Parity" in report_md


def test_custom_macro_factor_shock():
    stress_engine = StressTestEngine(random_state=42)
    weights = {"SPY": 0.5, "TLT": 0.3, "GLD": 0.2}

    res = stress_engine.simulate_macro_factor_shock(
        weights=weights,
        equity_shock_pct=-0.15,
        rate_shock_bps=100.0,
        vol_shock_pct=0.50,
        portfolio_value=1_000_000.0,
    )

    assert "Simulated_Portfolio_Return_pct" in res
    assert "Simulated_PnL_Dollar" in res
    assert res["Simulated_PnL_Dollar"] < 0  # Severe negative shock expected
    assert np.isclose(res["Ending_Portfolio_Value"], 1_000_000.0 + res["Simulated_PnL_Dollar"])
