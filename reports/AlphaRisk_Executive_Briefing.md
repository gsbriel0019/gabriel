# 🏛️ Institutional Risk Memorandum & Stress-Test Briefing

**Institution:** AlphaRisk Quantitative Analytics  
**Lead Quantitative Analyst:** Gabriel Proaño  
**Generated On:** September 23, 2026 - 20:59 UTC  
**Portfolio Notional Value:** $1,000,000.00  

---

## 1. Executive Summary & Market Regime Diagnosis

* **Current Market Regime:** **Low Volatility (Bullish / Expansion)**
* **Expected Regime Persistence:** 57.1 days (Bullish) | 48.9 days (Crisis)
* **Annualized Portfolio Expected Return:** 6.47%
* **Annualized Volatility ($\sigma$):** 10.23%
* **Sharpe Ratio ($r_f=4.5\%$):** 0.19
* **Maximum Historical Drawdown:** -27.90%

---

## 2. Portfolio Asset Allocation (Hierarchical Risk Parity - HRP)

| Asset   | Weight   | Allocation ($)   |
|:--------|:---------|:-----------------|
| BTC-USD | 2.34%    | $23,400.88       |
| GLD     | 25.42%   | $254,228.93      |
| QQQ     | 13.82%   | $138,213.40      |
| SPY     | 26.04%   | $260,372.02      |
| TLT     | 32.38%   | $323,784.78      |

---

## 3. Downside Tail Risk & Extreme Value Metrics (VaR & CVaR)

The subadditive coherent risk metric **Conditional Value at Risk (Expected Shortfall)** represents the expected loss given that a tail breach occurs.

| Confidence_Level   | Method                   | VaR_pct   | VaR_Dollar   | CVaR_Expected_Shortfall_pct   | CVaR_Dollar   |
|:-------------------|:-------------------------|:----------|:-------------|:------------------------------|:--------------|
| 95%                | Historical Simulation    | 1.08%     | $10,790.88   | 1.59%                         | $15,931.67    |
| 95%                | Parametric Gaussian      | 1.03%     | $10,342.28   | 1.30%                         | $13,034.87    |
| 95%                | Parametric Student's t   | 0.79%     | $7,943.52    | 1.65%                         | $16,462.95    |
| 95%                | Cornish-Fisher Expansion | 0.96%     | $9,620.04    | 1.51%                         | $15,137.75    |
| 99%                | Historical Simulation    | 1.83%     | $18,348.75   | 2.61%                         | $26,075.99    |
| 99%                | Parametric Gaussian      | 1.47%     | $14,733.66   | 1.69%                         | $16,917.24    |
| 99%                | Parametric Student's t   | 1.88%     | $18,837.80   | 3.67%                         | $36,737.77    |
| 99%                | Cornish-Fisher Expansion | 3.07%     | $30,713.76   | 3.91%                         | $39,081.41    |

---

## 4. Basel Regulatory Model Validation (BCBS Standards)

* **Basel Traffic Light Status:** 🟢 **GREEN (Acceptable)**
* **Total Sample Window ($T$):** 2206 trading days
* **Observed Breaches ($N$):** 118 (Expected: 110.3)
* **Empirical Breach Rate:** 5.35% (Nominal Target: 5.00%)
* **Kupiec Likelihood Ratio (Unconditional Coverage):** LR = 0.5538, p-value = 0.4568 (Passed ✅)
* **Christoffersen Independence Test (Clustering of Breaches):** LR = 0.0805, p-value = 0.7767 (Passed ✅)

---

## 5. Macroeconomic Stress-Testing & Historical Crisis Replay

Estimated impact of extreme historical macro events replayed onto current portfolio weights:

| Crisis_Scenario                                  |   Duration_Trading_Days | Portfolio_Impact_pct   | Estimated_PnL_Dollar   | Ending_Portfolio_Value   |
|:-------------------------------------------------|------------------------:|:-----------------------|:-----------------------|:-------------------------|
| 2008 Global Financial Crisis (Lehman Bankruptcy) |                      60 | -10.10%                | $-100,975.18           | $899,024.82              |
| March 2020 COVID-19 Liquidity Shock              |                      22 | -12.20%                | $-122,023.01           | $877,976.99              |
| 2022 Aggressive Fed Rate Hikes & Inflation Shock |                     180 | -21.57%                | $-215,749.53           | $784,250.47              |
| Hypothetical Geopolitical Stagflation Shock      |                      45 | -5.09%                 | $-50,892.41            | $949,107.59              |

---

*Report generated autonomously by **AlphaRisk Platform** — Developed by Gabriel Proaño.*
