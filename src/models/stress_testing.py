"""
Macroeconomic Stress-Testing and Correlated Monte Carlo Simulation Engine.
Implements:
- Cholesky-factorized Correlated Geometric Brownian Motion (GBM) Paths
- Historical Crisis Replays (2008 GFC, 2020 Covid Crash, 2022 Fed Hike Cycle)
- Hypothetical Factor Elasticity Shocks (Fed Policy Shocks & Volatility Spikes)
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class StressTestEngine:
    """
    Simulates portfolio behavior under severe market dislocations and macroeconomic stress.
    """

    HISTORICAL_CRISES = {
        "2008_GFC": {
            "name": "2008 Global Financial Crisis (Lehman Bankruptcy)",
            "duration_days": 60,
            "shocks": {"SPY": -0.38, "QQQ": -0.35, "GLD": +0.05, "TLT": +0.14, "BTC-USD": -0.50},
            "vix_spike": 80.0,
        },
        "2020_COVID": {
            "name": "March 2020 COVID-19 Liquidity Shock",
            "duration_days": 22,
            "shocks": {"SPY": -0.34, "QQQ": -0.28, "GLD": -0.04, "TLT": +0.08, "BTC-USD": -0.45},
            "vix_spike": 82.7,
        },
        "2022_FED_TIGHTENING": {
            "name": "2022 Aggressive Fed Rate Hikes & Inflation Shock",
            "duration_days": 180,
            "shocks": {"SPY": -0.19, "QQQ": -0.33, "GLD": -0.02, "TLT": -0.31, "BTC-USD": -0.65},
            "vix_spike": 34.0,
        },
        "STAGFLATION_COMMODITY": {
            "name": "Hypothetical Geopolitical Stagflation Shock",
            "duration_days": 45,
            "shocks": {"SPY": -0.16, "QQQ": -0.22, "GLD": +0.28, "TLT": -0.14, "BTC-USD": -0.20},
            "vix_spike": 45.0,
        },
    }

    def __init__(self, random_state: int = 42):
        self.rng = np.random.default_rng(random_state)

    def run_correlated_monte_carlo(
        self,
        returns: pd.DataFrame,
        weights: Dict[str, float],
        n_simulations: int = 5000,
        horizon_days: int = 30,
        initial_portfolio_value: float = 1_000_000.0,
    ) -> Dict[str, Union[np.ndarray, pd.DataFrame, float]]:
        """
        Run multi-asset correlated Monte Carlo simulation using Cholesky Decomposition.
        Returns simulated cumulative wealth paths and terminal distribution statistics.
        """
        tickers = list(returns.columns)
        w_vec = np.array([weights.get(t, 0.0) for t in tickers])

        mean_daily = returns.mean().values
        cov_matrix = returns.cov().values

        # Ensure covariance matrix is positive semi-definite
        cov_matrix = (cov_matrix + cov_matrix.T) / 2.0
        min_eig = np.min(np.linalg.eigvals(cov_matrix))
        if min_eig < 1e-8:
            cov_matrix += np.eye(len(tickers)) * (1e-8 - min_eig)

        chol = np.linalg.cholesky(cov_matrix)
        n_assets = len(tickers)

        # Standard normal random variates (simulations, days, assets)
        z = self.rng.standard_normal((n_simulations, horizon_days, n_assets))

        # Correlated shocks: dot product along last axis
        correlated_shocks = np.einsum("sda,ba->sdb", z, chol)

        # Drift adjustment: mu - 0.5 * sigma^2
        diag_var = np.diag(cov_matrix)
        drift = (mean_daily - 0.5 * diag_var).reshape(1, 1, n_assets)
        simulated_log_returns = drift + correlated_shocks

        # Portfolio daily return = sum_i(w_i * (exp(r_{i,t}) - 1))
        simulated_simple_returns = np.exp(simulated_log_returns) - 1.0
        portfolio_daily_returns = np.einsum("sda,a->sd", simulated_simple_returns, w_vec)

        # Cumulative wealth paths
        wealth_paths = np.zeros((n_simulations, horizon_days + 1))
        wealth_paths[:, 0] = initial_portfolio_value

        for d in range(horizon_days):
            wealth_paths[:, d + 1] = wealth_paths[:, d] * (1.0 + portfolio_daily_returns[:, d])

        terminal_wealth = wealth_paths[:, -1]
        terminal_returns = (terminal_wealth - initial_portfolio_value) / initial_portfolio_value

        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        summary_table = pd.DataFrame({
            "Percentile": [f"{p}th" for p in percentiles],
            "Terminal_Portfolio_Value": np.percentile(terminal_wealth, percentiles),
            "Cumulative_Return_pct": np.percentile(terminal_returns, percentiles),
        })

        return {
            "wealth_paths": wealth_paths,
            "percentiles_summary": summary_table,
            "median_terminal_value": float(np.median(terminal_wealth)),
            "var_95_dollar": float(initial_portfolio_value - np.percentile(terminal_wealth, 5)),
            "var_99_dollar": float(initial_portfolio_value - np.percentile(terminal_wealth, 1)),
        }

    def simulate_historical_crisis(
        self,
        weights: Dict[str, float],
        portfolio_value: float = 1_000_000.0,
    ) -> pd.DataFrame:
        """
        Evaluate estimated portfolio impact across famous historical market dislocations.
        """
        results = []
        for crisis_id, crisis_info in self.HISTORICAL_CRISES.items():
            total_shock_pct = 0.0
            asset_breakdown = {}

            for ticker, weight in weights.items():
                shock = crisis_info["shocks"].get(ticker, -0.15)  # default conservative equity-like drop
                impact = weight * shock
                total_shock_pct += impact
                asset_breakdown[ticker] = impact

            pnl_dollar = portfolio_value * total_shock_pct

            results.append({
                "Crisis_Scenario": crisis_info["name"],
                "Duration_Trading_Days": crisis_info["duration_days"],
                "Portfolio_Impact_pct": total_shock_pct,
                "Estimated_PnL_Dollar": pnl_dollar,
                "Ending_Portfolio_Value": portfolio_value + pnl_dollar,
                "Peak_VIX_Level": crisis_info["vix_spike"],
            })

        return pd.DataFrame(results)

    def simulate_macro_factor_shock(
        self,
        weights: Dict[str, float],
        equity_shock_pct: float = -0.10,
        rate_shock_bps: float = 50.0,
        vol_shock_pct: float = 0.30,
        portfolio_value: float = 1_000_000.0,
    ) -> Dict[str, float]:
        """
        Simulate custom macroeconomic factor shocks (Fed interest rate, equity market, volatility).
        Applies standard factor sensitivities (duration for bonds, beta for equities/crypto).
        """
        total_ret = 0.0

        for ticker, w in weights.items():
            t_upper = ticker.upper()
            if "SPY" in t_upper or "VTI" in t_upper:
                asset_ret = equity_shock_pct
            elif "QQQ" in t_upper:
                asset_ret = equity_shock_pct * 1.35 - (rate_shock_bps / 10000.0) * 8.0
            elif "TLT" in t_upper or "BOND" in t_upper:
                # Modified duration approx 16 years for TLT
                asset_ret = -16.0 * (rate_shock_bps / 10000.0)
            elif "GLD" in t_upper or "GOLD" in t_upper:
                asset_ret = (vol_shock_pct * 0.15) - (rate_shock_bps / 10000.0) * 3.0
            elif "BTC" in t_upper or "CRYPTO" in t_upper:
                asset_ret = equity_shock_pct * 2.2 - (rate_shock_bps / 10000.0) * 15.0
            else:
                asset_ret = equity_shock_pct * 0.9

            total_ret += w * asset_ret

        pnl = portfolio_value * total_ret
        return {
            "Simulated_Portfolio_Return_pct": total_ret,
            "Simulated_PnL_Dollar": pnl,
            "Ending_Portfolio_Value": portfolio_value + pnl,
        }
