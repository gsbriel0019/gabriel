"""
Generates the publication-ready educational Jupyter Notebook for AlphaRisk.
"""

import json
from pathlib import Path

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 📈 AlphaRisk: Quantitative Market Regime Detection & Stress-Testing Platform\n",
                "**Author:** Gabriel Proaño  \n",
                "**Focus:** Financial Econometrics, Latent Regime Modeling (HMM/GMM), Extreme Tail Risk (VaR/CVaR), and Hierarchical Risk Parity (HRP)\n",
                "\n",
                "---\n",
                "\n",
                "## 1. Executive Context & Theoretical Foundations\n",
                "\n",
                "Financial asset returns exhibit significant stylized facts that invalidate standard Gaussian assumptions:\n",
                "1. **Volatility Clustering:** High-volatility days cluster together (Mandelbrot, 1963).\n",
                "2. **Leptokurtosis (Fat Tails):** The probability of extreme drawdown events is orders of magnitude higher than predicted by a normal distribution.\n",
                "3. **Regime Shifts:** Macroeconomic cycles, monetary policy transitions (e.g. Federal Reserve interest rate hikes), and liquidity crises induce structural breaks in the return-generating process.\n",
                "\n",
                "This notebook demonstrates an end-to-end institutional workflow to detect latent market regimes, quantify downside tail risk compliant with the **Basel Committee on Banking Supervision (BCBS)**, allocate risk via **Hierarchical Risk Parity (HRP)**, and stress-test the portfolio under historical crisis scenarios."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys\n",
                "from pathlib import Path\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "# Add project root to sys.path\n",
                "root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n",
                "if str(root) not in sys.path:\n",
                "    sys.path.insert(0, str(root))\n",
                "\n",
                "from src.data.data_loader import DataLoader\n",
                "from src.data.macro_loader import MacroLoader\n",
                "from src.features.risk_factors import RiskFactorEngine\n",
                "from src.models.regime_detector import MarketRegimeDetector\n",
                "from src.models.tail_risk import TailRiskEngine\n",
                "from src.models.portfolio_opt import PortfolioOptimizer\n",
                "from src.models.backtest import RiskBacktester\n",
                "from src.models.stress_testing import StressTestEngine\n",
                "\n",
                "print('✅ AlphaRisk Core Modules imported successfully.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Ingestion & Feature Engineering\n",
                "We fetch continuous historical prices for a multi-asset universe:\n",
                "- **SPY:** S&P 500 ETF (Core US Large Cap Equity)\n",
                "- **QQQ:** Invesco QQQ (Nasdaq 100 Growth / Technology)\n",
                "- **GLD:** SPDR Gold Shares (Safe Haven / Real Asset)\n",
                "- **TLT:** 20+ Year Treasury Bond ETF (Duration / Rates)\n",
                "- **BTC-USD:** Bitcoin (Asymmetric Risk / Liquidity Proxy)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "loader = DataLoader(use_cache=True)\n",
                "universe = ['SPY', 'QQQ', 'GLD', 'TLT', 'BTC-USD']\n",
                "prices = loader.fetch_historical_prices(universe, start_date='2019-01-01')\n",
                "returns = loader.compute_log_returns(prices)\n",
                "\n",
                "print(f'Ingested {len(prices)} trading days across {len(universe)} assets.')\n",
                "returns.describe().T[['mean', 'std', 'min', 'max']]"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Unsupervised Market Regime Detection: Hidden Markov Models (HMM)\n",
                "\n",
                "We model market state transitions as a first-order Markov Chain with Gaussian emissions:\n",
                "$$\\mathbb{P}(S_t = j \\mid S_{t-1} = i) = P_{i,j}$$\n",
                "$$\\mathbf{y}_t \\mid S_t = k \\sim \\mathcal{N}(\\boldsymbol{\\mu}_k, \\boldsymbol{\\Sigma}_k)$$\n",
                "\n",
                "States are sorted by empirical volatility to maintain consistent interpretation:\n",
                "- **Regime 0:** Low Volatility (Bull / Expansion)\n",
                "- **Regime 1:** Moderate Volatility (Rangebound / Mean-Reverting)\n",
                "- **Regime 2:** High Volatility (Market Stress / Liquidity Crisis)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "spy_ret = returns['SPY']\n",
                "spy_vol = RiskFactorEngine.compute_rolling_realized_volatility(spy_ret, window=21)\n",
                "features = pd.concat([spy_ret, spy_vol], axis=1).dropna()\n",
                "features.columns = ['return', 'vol']\n",
                "\n",
                "detector = MarketRegimeDetector(n_regimes=3, model_type='HMM', random_state=42)\n",
                "detector.fit(features)\n",
                "regimes = detector.predict(features)\n",
                "\n",
                "print('=== Transition Probability Matrix P ===')\n",
                "print(detector.get_transition_matrix().to_string())\n",
                "\n",
                "print('\\n=== Expected Regime Persistence (Days) ===')\n",
                "for k, v in detector.calculate_expected_durations().items():\n",
                "    print(f'{k}: {v:.1f} trading days')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Downside Tail Risk & Extreme Value Theory (VaR & CVaR)\n",
                "\n",
                "Value at Risk ($VaR_\\alpha$) represents the maximum loss at confidence level $\\alpha$.\n",
                "**Conditional Value at Risk (CVaR / Expected Shortfall)** satisfies the subadditivity axiom of coherent risk measures:\n",
                "$$CVaR_\\alpha(X) = -\\mathbb{E}[X \\mid X \\le -VaR_\\alpha(X)]$$"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "tail_engine = TailRiskEngine(confidence_levels=[0.95, 0.99])\n",
                "metrics_table = tail_engine.compute_all_metrics(spy_ret, portfolio_value=1_000_000.0)\n",
                "metrics_table"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Basel Regulatory Model Validation (Kupiec & Christoffersen Tests)\n",
                "\n",
                "To validate our risk engine against regulatory standards, we evaluate out-of-sample forecast accuracy using:\n",
                "1. **Kupiec Likelihood Ratio (POF):** Tests whether the empirical frequency of exceptions equals nominal probability ($p = 1 - \\alpha$).\n",
                "2. **Christoffersen Independence Test:** Tests against clustering of exceptions across consecutive days.\n",
                "3. **Basel Traffic Light:** Classifies model into Green, Yellow, or Red zones."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "rolling_var = spy_ret.rolling(window=252).apply(\n",
                "    lambda x: TailRiskEngine.historical_var_cvar(x.values, 0.95)[0], raw=False\n",
                ").dropna()\n",
                "\n",
                "backtester = RiskBacktester(confidence_level=0.95)\n",
                "bt_results = backtester.run_full_backtest(spy_ret.loc[rolling_var.index], rolling_var)\n",
                "\n",
                "for k, v in bt_results.items():\n",
                "    print(f'{k:30}: {v}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Multivariate Portfolio Optimization: Hierarchical Risk Parity (HRP)\n",
                "\n",
                "Traditional Markowitz Mean-Variance optimization maximizes estimation error through covariance inversion $(\\boldsymbol{\\Sigma}^{-1})$.\n",
                "**Hierarchical Risk Parity (López de Prado, 2016)** replaces matrix inversion with machine learning tree clustering:\n",
                "1. Distance metric: $d_{i,j} = \\sqrt{\\frac{1}{2}(1 - \\rho_{i,j})}$\n",
                "2. Quasi-diagonalization of covariance.\n",
                "3. Recursive bisection."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "opt = PortfolioOptimizer(risk_free_rate=0.045)\n",
                "hrp_weights = opt.optimize_hrp(returns)\n",
                "sharpe_weights = opt.optimize_max_sharpe(returns)\n",
                "min_vol_weights = opt.optimize_min_volatility(returns)\n",
                "\n",
                "comparison = pd.DataFrame({\n",
                "    'Hierarchical Risk Parity (HRP)': hrp_weights,\n",
                "    'Maximum Sharpe Ratio': sharpe_weights,\n",
                "    'Minimum Volatility': min_vol_weights,\n",
                "    'Equal Weight Benchmark': opt.optimize_equal_weight(returns)\n",
                "})\n",
                "comparison.style.format('{:.2%}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Macroeconomic Stress-Testing & Historical Crisis Replay\n",
                "We evaluate how the HRP portfolio behaves under severe historical disruptions (2008 Lehman collapse, 2020 COVID shock, 2022 Fed tightening)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "stress_engine = StressTestEngine(random_state=42)\n",
                "crisis_report = stress_engine.simulate_historical_crisis(hrp_weights, portfolio_value=1_000_000.0)\n",
                "crisis_report"
            ]
        }
    ],
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.10.5"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out_path = Path("notebooks/01_alpha_risk_econometrics.ipynb")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print(f"Generated {out_path.resolve()}")
