"""
Regulatory Risk Backtesting Engine.
Implements Basel Committee Standards:
- Kupiec Proportion of Failures (POF) Likelihood Ratio Test
- Christoffersen Independence Test (Breach Clustering)
- Christoffersen Conditional Coverage Joint Test
- Basel Traffic Light System (Green, Yellow, Red Zones)
"""

from typing import Dict, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


class RiskBacktester:
    """
    Evaluates out-of-sample accuracy and independence of Value at Risk models.
    Compliant with the Basel Committee on Banking Supervision (BCBS) regulatory framework.
    """

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.p_expected = 1.0 - confidence_level

    def run_full_backtest(
        self, actual_returns: pd.Series, predicted_var: pd.Series
    ) -> Dict[str, Union[float, int, str, bool]]:
        """
        Execute comprehensive backtest comparing actual returns vs predicted VaR threshold.
        """
        # Align series
        df = pd.DataFrame({"return": actual_returns, "var": predicted_var}).dropna()
        T = len(df)
        if T < 20:
            raise ValueError(f"Insufficient sample size for backtesting: {T} observations. Minimum 20 required.")

        # A breach occurs when Loss > VaR <=> Return < -VaR
        breaches = (df["return"] < -df["var"]).astype(int)
        N = int(breaches.sum())
        p_hat = N / T if T > 0 else 0.0

        # 1. Kupiec POF Test
        lr_pof, pval_pof = self.kupiec_test(T, N, self.p_expected)

        # 2. Christoffersen Independence Test
        lr_ind, pval_ind = self.christoffersen_independence_test(breaches.values)

        # 3. Conditional Coverage Test
        lr_cc = lr_pof + lr_ind
        pval_cc = 1.0 - stats.chi2.cdf(lr_cc, df=2)

        # Basel Traffic Light Zone
        traffic_light = self.basel_traffic_light(T, N, self.confidence_level)

        return {
            "Total_Observations_T": T,
            "Total_Breaches_N": N,
            "Expected_Breaches": float(np.round(T * self.p_expected, 1)),
            "Empirical_Failure_Rate": float(np.round(p_hat, 4)),
            "Expected_Failure_Rate": float(np.round(self.p_expected, 4)),
            "Kupiec_LR_Stat": float(np.round(lr_pof, 4)),
            "Kupiec_p_value": float(np.round(pval_pof, 4)),
            "Kupiec_Null_Accepted": bool(pval_pof > 0.05),
            "Christoffersen_Ind_LR_Stat": float(np.round(lr_ind, 4)),
            "Christoffersen_Ind_p_value": float(np.round(pval_ind, 4)),
            "Independence_Null_Accepted": bool(pval_ind > 0.05),
            "Conditional_Coverage_LR": float(np.round(lr_cc, 4)),
            "Conditional_Coverage_p_val": float(np.round(pval_cc, 4)),
            "Model_Valid_at_5pct": bool(pval_cc > 0.05),
            "Basel_Traffic_Light": traffic_light,
        }

    @staticmethod
    def kupiec_test(T: int, N: int, p: float) -> Tuple[float, float]:
        """
        Likelihood Ratio test for unconditional coverage:
        LR_POF = -2 * ln( (1-p)^{T-N} * p^N / ( (1 - N/T)^{T-N} * (N/T)^N ) )
        Follows Chi-Square distribution with 1 degree of freedom.
        """
        if N == 0:
            lr = -2.0 * T * np.log(1.0 - p)
            pval = 1.0 - stats.chi2.cdf(lr, df=1)
            return float(lr), float(pval)

        p_hat = N / T
        if p_hat >= 1.0:
            return float("inf"), 0.0

        # Log-likelihood ratio
        term1 = (T - N) * np.log((1.0 - p) / (1.0 - p_hat))
        term2 = N * np.log(p / p_hat)
        lr = -2.0 * (term1 + term2)
        lr = max(0.0, lr)

        pval = 1.0 - stats.chi2.cdf(lr, df=1)
        return float(lr), float(pval)

    @staticmethod
    def christoffersen_independence_test(breaches: np.ndarray) -> Tuple[float, float]:
        """
        Tests whether violations are independent against a first-order Markov chain alternative.
        Estimates transition matrix:
        n_{00}: no breach followed by no breach
        n_{01}: no breach followed by breach
        n_{10}: breach followed by no breach
        n_{11}: breach followed by breach
        """
        n00 = np.sum((breaches[:-1] == 0) & (breaches[1:] == 0))
        n01 = np.sum((breaches[:-1] == 0) & (breaches[1:] == 1))
        n10 = np.sum((breaches[:-1] == 1) & (breaches[1:] == 0))
        n11 = np.sum((breaches[:-1] == 1) & (breaches[1:] == 1))

        # Avoid log(0)
        pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
        pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
        pi = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00 + n01 + n10 + n11) > 0 else 0.0

        # Unrestricted likelihood under independence
        if pi0 == 0.0 and pi1 == 0.0:
            return 0.0, 1.0

        try:
            L_null = ((1.0 - pi) ** (n00 + n10)) * (pi ** (n01 + n11))
            L_alt = ((1.0 - pi0) ** n00) * (pi0 ** n01) * ((1.0 - pi1) ** n10) * (pi1 ** n11)

            if L_null <= 0.0 or L_alt <= 0.0:
                lr = 0.0
            else:
                lr = -2.0 * np.log(L_null / L_alt)
                lr = max(0.0, lr)
        except Exception:
            lr = 0.0

        pval = 1.0 - stats.chi2.cdf(lr, df=1)
        return float(lr), float(pval)

    @staticmethod
    def basel_traffic_light(T: int, N: int, confidence: float = 0.95) -> str:
        """
        Categorizes model into Green, Yellow, or Red zone based on cumulative binomial distribution.
        """
        p = 1.0 - confidence
        # Cumulative probability of observing up to N breaches
        cum_prob = stats.binom.cdf(N, T, p)

        if cum_prob < 0.95:
            return "GREEN (Acceptable)"
        elif cum_prob < 0.9999:
            return "YELLOW (Cautionary / Supervisory Review)"
        else:
            return "RED (Model Rejected by Regulator)"
