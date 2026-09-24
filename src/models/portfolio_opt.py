"""
Portfolio Optimization Engine.
Implements Hierarchical Risk Parity (HRP) via unsupervised tree clustering,
Modern Portfolio Theory (Markowitz Minimum Volatility & Max Sharpe), and Equal Weight.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform
from scipy.optimize import minimize


class PortfolioOptimizer:
    """
    Multivariate Portfolio Allocation Engine.
    Features Marcos López de Prado's Hierarchical Risk Parity (HRP) to eliminate
    covariance matrix inversion instabilities, alongside classic Mean-Variance frontiers.
    """

    def __init__(self, risk_free_rate: float = 0.045):
        self.risk_free_rate = risk_free_rate

    def optimize_hrp(self, returns: pd.DataFrame) -> Dict[str, float]:
        """
        Hierarchical Risk Parity (HRP) allocation.
        Steps:
        1. Hierarchical Tree Clustering on correlation distance metric.
        2. Matrix Quasi-Diagonalization (reordering rows/cols).
        3. Recursive Bisection weighting based on inverse cluster variance.
        """
        corr = returns.corr().fillna(0.0)
        cov = returns.cov().fillna(0.0)

        # 1. Distance matrix: d_{i,j} = sqrt(0.5 * (1 - rho_{i,j}))
        dist = np.sqrt(np.clip(0.5 * (1.0 - corr.values), 0.0, 1.0))
        np.fill_diagonal(dist, 0.0)

        # Hierarchical Linkage (Single/Ward)
        condensed_dist = squareform(dist, checks=False)
        link = linkage(condensed_dist, method="single")

        # 2. Quasi-Diagonalization
        sorted_indices = self._get_quasi_diag(link)
        sorted_tickers = [returns.columns[i] for i in sorted_indices]

        # Reorder covariance matrix
        cov_sorted = cov.loc[sorted_tickers, sorted_tickers]

        # 3. Recursive Bisection
        weights = pd.Series(1.0, index=sorted_tickers)
        cluster_list = [sorted_tickers]

        while len(cluster_list) > 0:
            cluster_list = [
                cluster[start:end]
                for cluster in cluster_list
                for start, end in ((0, len(cluster) // 2), (len(cluster) // 2, len(cluster)))
                if len(cluster) > 1
            ]
            for i in range(0, len(cluster_list), 2):
                c1 = cluster_list[i]
                c2 = cluster_list[i + 1]

                v1 = self._get_cluster_variance(cov_sorted, c1)
                v2 = self._get_cluster_variance(cov_sorted, c2)

                alpha = 1.0 - v1 / (v1 + v2) if (v1 + v2) > 0 else 0.5
                weights[c1] *= alpha
                weights[c2] *= 1.0 - alpha

        # Ensure normalized long-only weights in original asset order
        weights = weights.reindex(returns.columns).fillna(0.0)
        weights /= weights.sum()
        return weights.to_dict()

    def optimize_min_volatility(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Markowitz Minimum Variance portfolio with long-only constraints."""
        cov = returns.cov().values * 252.0
        n = len(returns.columns)

        def objective(w):
            return np.sqrt(w.T @ cov @ w)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0) for _ in range(n)]
        init_guess = np.repeat(1.0 / n, n)

        res = minimize(objective, init_guess, method="SLSQP", bounds=bounds, constraints=constraints)
        weights = res.x if res.success else init_guess
        weights /= np.sum(weights)
        return dict(zip(returns.columns, weights))

    def optimize_max_sharpe(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Tangency / Maximum Sharpe Ratio portfolio with long-only constraints."""
        mean_ret = returns.mean().values * 252.0
        cov = returns.cov().values * 252.0
        n = len(returns.columns)

        def neg_sharpe(w):
            port_ret = np.sum(w * mean_ret)
            port_vol = np.sqrt(w.T @ cov @ w)
            if port_vol <= 1e-8:
                return 0.0
            return -(port_ret - self.risk_free_rate) / port_vol

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0) for _ in range(n)]
        init_guess = np.repeat(1.0 / n, n)

        res = minimize(neg_sharpe, init_guess, method="SLSQP", bounds=bounds, constraints=constraints)
        weights = res.x if res.success else init_guess
        weights /= np.sum(weights)
        return dict(zip(returns.columns, weights))

    def optimize_equal_weight(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Equal Weight benchmark (1/N allocation)."""
        n = len(returns.columns)
        w = 1.0 / n if n > 0 else 0.0
        return {col: w for col in returns.columns}

    def compute_portfolio_metrics(
        self, weights: Dict[str, float], returns: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Compute annualized portfolio return, volatility, Sharpe ratio, and Maximum Drawdown.
        """
        w_vec = np.array([weights.get(c, 0.0) for c in returns.columns])
        mean_ret = returns.mean().values * 252.0
        cov = returns.cov().values * 252.0

        port_ret = float(np.sum(w_vec * mean_ret))
        port_vol = float(np.sqrt(w_vec.T @ cov @ w_vec))
        sharpe = (port_ret - self.risk_free_rate) / port_vol if port_vol > 0 else 0.0

        # Cumulative performance series for Drawdown calculation
        daily_port_returns = returns @ w_vec
        cum_ret = (1.0 + daily_port_returns).cumprod()
        running_max = cum_ret.cummax()
        drawdown = (cum_ret - running_max) / running_max
        max_dd = float(drawdown.min())

        return {
            "Annualized_Return": port_ret,
            "Annualized_Volatility": port_vol,
            "Sharpe_Ratio": sharpe,
            "Max_Drawdown": max_dd,
            "Calmar_Ratio": port_ret / abs(max_dd) if abs(max_dd) > 0 else 0.0,
        }

    # Internal helpers for HRP
    @staticmethod
    def _get_quasi_diag(link: np.ndarray) -> List[int]:
        """Recursive traversal of dendrogram to sort cluster items."""
        root = to_tree(link, rd=False)

        def _traverse(node):
            if node.is_leaf():
                return [node.id]
            return _traverse(node.left) + _traverse(node.right)

        return _traverse(root)

    @staticmethod
    def _get_cluster_variance(cov: pd.DataFrame, cluster: List[str]) -> float:
        """Calculate variance of an inverse-variance weighted sub-cluster."""
        sub_cov = cov.loc[cluster, cluster].values
        inv_diag = 1.0 / np.diag(sub_cov)
        inv_diag /= np.sum(inv_diag)
        w = inv_diag.reshape(-1, 1)
        var = float((w.T @ sub_cov @ w).item())
        return var
