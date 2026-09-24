"""
Risk Factors and Quantitative Feature Engineering Module.
Computes realized volatility, higher-order statistical moments, drawdowns, and tail metrics.
"""

from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats


class RiskFactorEngine:
    """Calculates quantitative risk factors, volatility estimators, and empirical moments."""

    @staticmethod
    def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """Calculate continuous log-returns: r_t = ln(P_t / P_{t-1})."""
        return np.log(prices / prices.shift(1)).dropna()

    @staticmethod
    def compute_rolling_realized_volatility(
        returns: pd.DataFrame, window: int = 21, annualize: bool = True
    ) -> pd.DataFrame:
        """
        Compute rolling sample standard deviation.
        If annualize=True, multiplies by sqrt(252).
        """
        factor = np.sqrt(252) if annualize else 1.0
        return (returns.rolling(window=window).std() * factor).dropna()

    @staticmethod
    def compute_parkinson_volatility(
        high: pd.DataFrame, low: pd.DataFrame, window: int = 21, annualize: bool = True
    ) -> pd.DataFrame:
        """
        Compute Parkinson extreme-value volatility estimator.
        Sigma^2 = (1 / (4 * ln(2))) * sum(ln(H_t / L_t)^2)
        5x more statistically efficient than standard close-to-close volatility.
        """
        hl_ratio = np.log(high / low)
        hl_sq = hl_ratio**2
        parkinson_var = (1.0 / (4.0 * np.log(2.0))) * hl_sq.rolling(window=window).mean()
        vol = np.sqrt(parkinson_var)
        if annualize:
            vol = vol * np.sqrt(252)
        return vol.dropna()

    @staticmethod
    def compute_drawdown_series(prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute drawdown series: (Price_t - Peak_t) / Peak_t.
        """
        running_max = prices.cummax()
        drawdown = (prices - running_max) / running_max
        return drawdown

    @staticmethod
    def compute_higher_moments(returns: pd.Series, window: int = 63) -> pd.DataFrame:
        """
        Compute rolling Skewness and Excess Kurtosis to monitor fat-tail dynamics.
        """
        rolling_skew = returns.rolling(window=window).apply(lambda x: stats.skew(x, bias=False), raw=False)
        rolling_kurt = returns.rolling(window=window).apply(lambda x: stats.kurtosis(x, bias=False), raw=False)

        return pd.DataFrame({"skewness": rolling_skew, "excess_kurtosis": rolling_kurt}, index=returns.index).dropna()

    @staticmethod
    def compute_downside_deviation(
        returns: pd.Series, mar: float = 0.0, annualize: bool = True
    ) -> float:
        """
        Compute downside semi-deviation below a Minimum Acceptable Return (MAR).
        Used for Sortino ratio and asymmetric downside risk quantification.
        """
        downside = returns[returns < mar] - mar
        if len(downside) == 0:
            return 0.0
        semi_var = np.mean(downside**2)
        dev = np.sqrt(semi_var)
        return dev * np.sqrt(252) if annualize else dev
