"""
Tail Risk and Extreme Value Engine.
Computes Value at Risk (VaR), Conditional VaR / Expected Shortfall (CVaR),
Cornish-Fisher expansion, and parametric distributions (Normal & Student's t).
"""

from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


class TailRiskEngine:
    """
    Quantifies market downside tail risk with multiple statistical methodologies:
    - Non-parametric Historical Simulation
    - Parametric Gaussian
    - Parametric Student's t (fat-tailed)
    - Cornish-Fisher Expansion (adjusted for empirical skewness and excess kurtosis)
    - Conditional VaR / Expected Shortfall (Subadditive coherent risk measure)
    """

    def __init__(self, confidence_levels: Union[float, List[float]] = (0.95, 0.99)):
        if isinstance(confidence_levels, (float, int)):
            confidence_levels = [float(confidence_levels)]
        self.confidence_levels = sorted([float(c) for c in confidence_levels])

    def compute_all_metrics(
        self,
        returns: Union[pd.Series, np.ndarray],
        horizon_days: int = 1,
        portfolio_value: float = 1_000_000.0,
    ) -> pd.DataFrame:
        """
        Calculate comprehensive VaR and CVaR table across all methodologies and confidence levels.
        """
        clean_returns = np.asarray(returns).ravel()
        clean_returns = clean_returns[~np.isnan(clean_returns)]

        records = []
        for alpha in self.confidence_levels:
            # 1. Historical VaR & CVaR
            hist_var, hist_cvar = self.historical_var_cvar(clean_returns, alpha)

            # 2. Parametric Gaussian
            norm_var, norm_cvar = self.parametric_gaussian_var_cvar(clean_returns, alpha)

            # 3. Parametric Student's t
            t_var, t_cvar = self.parametric_student_t_var_cvar(clean_returns, alpha)

            # 4. Cornish-Fisher Semi-parametric
            cf_var, cf_cvar = self.cornish_fisher_var_cvar(clean_returns, alpha)

            scale_factor = np.sqrt(horizon_days)

            for method, (var_val, cvar_val) in [
                ("Historical Simulation", (hist_var, hist_cvar)),
                ("Parametric Gaussian", (norm_var, norm_cvar)),
                ("Parametric Student's t", (t_var, t_cvar)),
                ("Cornish-Fisher Expansion", (cf_var, cf_cvar)),
            ]:
                scaled_var = var_val * scale_factor
                scaled_cvar = cvar_val * scale_factor
                records.append({
                    "Confidence_Level": f"{int(alpha * 100)}%",
                    "Method": method,
                    "Horizon_Days": horizon_days,
                    "VaR_pct": scaled_var,
                    "VaR_Dollar": scaled_var * portfolio_value,
                    "CVaR_Expected_Shortfall_pct": scaled_cvar,
                    "CVaR_Dollar": scaled_cvar * portfolio_value,
                })

        return pd.DataFrame(records)

    @staticmethod
    def historical_var_cvar(returns: np.ndarray, alpha: float = 0.95) -> Tuple[float, float]:
        """
        Historical Simulation VaR and Expected Shortfall:
        VaR_alpha = -Quantile(returns, 1 - alpha)
        CVaR_alpha = -E[returns | returns <= -VaR_alpha]
        """
        cutoff = (1.0 - alpha) * 100.0
        var = -np.percentile(returns, cutoff)
        tail_losses = returns[returns <= -var]
        cvar = -np.mean(tail_losses) if len(tail_losses) > 0 else var
        return float(var), float(cvar)

    @staticmethod
    def parametric_gaussian_var_cvar(returns: np.ndarray, alpha: float = 0.95) -> Tuple[float, float]:
        """
        Parametric Gaussian analytical formula:
        VaR = -(mu + z_{1-alpha} * sigma)
        CVaR = -(mu - sigma * phi(z_{alpha}) / (1 - alpha))
        """
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        z = stats.norm.ppf(1.0 - alpha)
        var = -(mu + z * sigma)
        phi_z = stats.norm.pdf(stats.norm.ppf(alpha))
        cvar = -(mu - sigma * (phi_z / (1.0 - alpha)))
        return float(var), float(cvar)

    @staticmethod
    def parametric_student_t_var_cvar(returns: np.ndarray, alpha: float = 0.95) -> Tuple[float, float]:
        """
        Parametric Student's t distribution fitted via Maximum Likelihood Estimation (MLE).
        Captures fat-tailed leptokurtic distribution.
        """
        params = stats.t.fit(returns)
        df, loc, scale = params
        df = max(df, 2.1)  # Ensure variance exists

        var = -stats.t.ppf(1.0 - alpha, df=df, loc=loc, scale=scale)

        # Numerical integration or simulation for analytical ES of Student-t
        q = stats.t.ppf(1.0 - alpha, df=df)
        cvar_std = (df + q**2) / (df - 1.0) * stats.t.pdf(q, df=df) / (1.0 - alpha)
        cvar = -(loc - scale * cvar_std)

        return float(var), float(cvar)

    @staticmethod
    def cornish_fisher_var_cvar(returns: np.ndarray, alpha: float = 0.95) -> Tuple[float, float]:
        """
        Cornish-Fisher expansion adjusting the normal quantile for skewness (S) and excess kurtosis (K):
        w_alpha = z + (z^2 - 1)*S/6 + (z^3 - 3z)*K/24 - (2z^3 - 5z)*S^2/36
        """
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        skew = stats.skew(returns, bias=False)
        kurt = stats.kurtosis(returns, bias=False)

        z = stats.norm.ppf(1.0 - alpha)
        w = (
            z
            + (z**2 - 1.0) * skew / 6.0
            + (z**3 - 3.0 * z) * kurt / 24.0
            - (2.0 * z**3 - 5.0 * z) * (skew**2) / 36.0
        )
        var = -(mu + w * sigma)

        # Empirical tail conditional expectation given CF VaR threshold
        tail = returns[returns <= -var]
        cvar = -np.mean(tail) if len(tail) > 0 else var * 1.25
        return float(var), float(cvar)
