"""
Executive Risk Reporter Module.
Compiles quantitative metrics, tail risk, regime states, and stress test outcomes
into an institutional-grade executive risk briefing document (Markdown / HTML).
"""

from datetime import datetime
from typing import Dict, Any
import pandas as pd


class ExecutiveRiskReporter:
    """Generates institutional risk memoranda for Chief Risk Officers (CRO) and investment committees."""

    def __init__(self, author: str = "Gabriel Proaño", institution: str = "AlphaRisk Analytics"):
        self.author = author
        self.institution = institution

    def generate_markdown_report(
        self,
        current_regime: str,
        regime_durations: Dict[str, float],
        tail_risk_df: pd.DataFrame,
        backtest_results: Dict[str, Any],
        portfolio_metrics: Dict[str, float],
        portfolio_weights: Dict[str, float],
        stress_df: pd.DataFrame,
        portfolio_value: float = 1_000_000.0,
    ) -> str:
        """
        Produce a full institutional risk report in clean Markdown format.
        """
        date_str = datetime.now().strftime("%B %d, %Y - %H:%M UTC")

        weights_table = pd.DataFrame([
            {"Asset": k, "Weight": f"{v * 100:.2f}%", "Allocation ($)": f"${v * portfolio_value:,.2f}"}
            for k, v in portfolio_weights.items()
        ]).to_markdown(index=False)

        tail_table = tail_risk_df[[
            "Confidence_Level", "Method", "VaR_pct", "VaR_Dollar", "CVaR_Expected_Shortfall_pct", "CVaR_Dollar"
        ]].copy()
        tail_table["VaR_pct"] = tail_table["VaR_pct"].apply(lambda x: f"{x * 100:.2f}%")
        tail_table["VaR_Dollar"] = tail_table["VaR_Dollar"].apply(lambda x: f"${x:,.2f}")
        tail_table["CVaR_Expected_Shortfall_pct"] = tail_table["CVaR_Expected_Shortfall_pct"].apply(lambda x: f"{x * 100:.2f}%")
        tail_table["CVaR_Dollar"] = tail_table["CVaR_Dollar"].apply(lambda x: f"${x:,.2f}")
        tail_md = tail_table.to_markdown(index=False)

        stress_table = stress_df[[
            "Crisis_Scenario", "Duration_Trading_Days", "Portfolio_Impact_pct", "Estimated_PnL_Dollar", "Ending_Portfolio_Value"
        ]].copy()
        stress_table["Portfolio_Impact_pct"] = stress_table["Portfolio_Impact_pct"].apply(lambda x: f"{x * 100:.2f}%")
        stress_table["Estimated_PnL_Dollar"] = stress_table["Estimated_PnL_Dollar"].apply(lambda x: f"${x:,.2f}")
        stress_table["Ending_Portfolio_Value"] = stress_table["Ending_Portfolio_Value"].apply(lambda x: f"${x:,.2f}")
        stress_md = stress_table.to_markdown(index=False)

        traffic_badge = "🟢" if "GREEN" in backtest_results.get("Basel_Traffic_Light", "") else "🟡"

        report = f"""# 🏛️ Institutional Risk Memorandum & Stress-Test Briefing

**Institution:** {self.institution}  
**Lead Quantitative Analyst:** {self.author}  
**Generated On:** {date_str}  
**Portfolio Notional Value:** ${portfolio_value:,.2f}  

---

## 1. Executive Summary & Market Regime Diagnosis

* **Current Market Regime:** **{current_regime}**
* **Expected Regime Persistence:** {regime_durations.get('Regime 0', 'N/A')} days (Bullish) | {regime_durations.get('Regime 2', 'N/A')} days (Crisis)
* **Annualized Portfolio Expected Return:** {portfolio_metrics.get('Annualized_Return', 0.0) * 100:.2f}%
* **Annualized Volatility ($\sigma$):** {portfolio_metrics.get('Annualized_Volatility', 0.0) * 100:.2f}%
* **Sharpe Ratio ($r_f=4.5\%$):** {portfolio_metrics.get('Sharpe_Ratio', 0.0):.2f}
* **Maximum Historical Drawdown:** {portfolio_metrics.get('Max_Drawdown', 0.0) * 100:.2f}%

---

## 2. Portfolio Asset Allocation (Hierarchical Risk Parity - HRP)

{weights_table}

---

## 3. Downside Tail Risk & Extreme Value Metrics (VaR & CVaR)

The subadditive coherent risk metric **Conditional Value at Risk (Expected Shortfall)** represents the expected loss given that a tail breach occurs.

{tail_md}

---

## 4. Basel Regulatory Model Validation (BCBS Standards)

* **Basel Traffic Light Status:** {traffic_badge} **{backtest_results.get('Basel_Traffic_Light', 'N/A')}**
* **Total Sample Window ($T$):** {backtest_results.get('Total_Observations_T', 0)} trading days
* **Observed Breaches ($N$):** {backtest_results.get('Total_Breaches_N', 0)} (Expected: {backtest_results.get('Expected_Breaches', 0.0)})
* **Empirical Breach Rate:** {backtest_results.get('Empirical_Failure_Rate', 0.0) * 100:.2f}% (Nominal Target: {backtest_results.get('Expected_Failure_Rate', 0.0) * 100:.2f}%)
* **Kupiec Likelihood Ratio (Unconditional Coverage):** LR = {backtest_results.get('Kupiec_LR_Stat', 0.0)}, p-value = {backtest_results.get('Kupiec_p_value', 0.0)} {'(Passed ✅)' if backtest_results.get('Kupiec_Null_Accepted') else '(Failed ❌)'}
* **Christoffersen Independence Test (Clustering of Breaches):** LR = {backtest_results.get('Christoffersen_Ind_LR_Stat', 0.0)}, p-value = {backtest_results.get('Christoffersen_Ind_p_value', 0.0)} {'(Passed ✅)' if backtest_results.get('Independence_Null_Accepted') else '(Failed ❌)'}

---

## 5. Macroeconomic Stress-Testing & Historical Crisis Replay

Estimated impact of extreme historical macro events replayed onto current portfolio weights:

{stress_md}

---

*Report generated autonomously by **AlphaRisk Platform** — Developed by Gabriel Proaño.*
"""
        return report
