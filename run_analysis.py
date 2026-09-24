"""
AlphaRisk CLI Pipeline Runner.
Author: Gabriel Proaño
Executes end-to-end quantitative risk workflow directly from terminal and exports executive brief.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
from tabulate import tabulate

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.data.data_loader import DataLoader
from src.features.risk_factors import RiskFactorEngine
from src.models.regime_detector import MarketRegimeDetector
from src.models.tail_risk import TailRiskEngine
from src.models.portfolio_opt import PortfolioOptimizer
from src.models.backtest import RiskBacktester
from src.models.stress_testing import StressTestEngine
from src.reporting.risk_report import ExecutiveRiskReporter


def parse_args():
    parser = argparse.ArgumentParser(description="AlphaRisk Quantitative Platform CLI")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["SPY", "QQQ", "GLD", "TLT", "BTC-USD"],
        help="List of asset tickers for multi-asset universe",
    )
    parser.add_argument("--benchmark", default="SPY", help="Benchmark ticker symbol")
    parser.add_argument("--start-date", default="2018-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=1_000_000.0, help="Portfolio notional capital in USD")
    parser.add_argument("--regimes", type=int, default=3, help="Number of latent market regimes (HMM)")
    parser.add_argument("--output-report", default="reports/AlphaRisk_Executive_Briefing.md", help="Export report path")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 75)
    print("  📈 ALPHARISK: QUANTITATIVE MARKET REGIME & STRESS-TESTING PLATFORM")
    print("  Lead Quantitative Analyst: Gabriel Proaño")
    print("=" * 75)
    print(f"[*] Universe: {', '.join(args.tickers)} | Benchmark: {args.benchmark}")
    print(f"[*] Capital: ${args.capital:,.2f} | Time Horizon: {args.start_date} to Present\n")

    # 1. Ingestion
    loader = DataLoader(cache_dir="data/cache", use_cache=True)
    prices = loader.fetch_historical_prices(args.tickers, start_date=args.start_date)
    returns = loader.compute_log_returns(prices)
    print(f"[+] Loaded {len(prices)} trading observations across {len(args.tickers)} assets.")

    # 2. Market Regime Detection
    bm_ret = returns[args.benchmark]
    bm_vol = RiskFactorEngine.compute_rolling_realized_volatility(bm_ret, window=21)
    features = pd.concat([bm_ret, bm_vol], axis=1).dropna()
    features.columns = ["return", "vol"]

    regime_detector = MarketRegimeDetector(n_regimes=args.regimes, model_type="HMM", random_state=42)
    regime_detector.fit(features)
    predicted_regimes = regime_detector.predict(features)
    current_regime_id = int(predicted_regimes[-1])
    current_regime_name = regime_detector.REGIME_NAMES.get(current_regime_id, f"Regime {current_regime_id}")
    regime_durations = regime_detector.calculate_expected_durations()

    print(f"\n[+] CURRENT MARKET REGIME: {current_regime_name}")
    print(f"    Expected Persistence: {regime_durations.get(f'Regime {current_regime_id}', 'N/A')} trading days")

    # 3. Portfolio Allocation (HRP)
    opt = PortfolioOptimizer(risk_free_rate=0.045)
    hrp_weights = opt.optimize_hrp(returns)
    port_metrics = opt.compute_portfolio_metrics(hrp_weights, returns)

    print("\n[+] OPTIMAL ALLOCATION (Hierarchical Risk Parity - HRP):")
    weights_summary = [
        [ticker, f"{w * 100:.2f}%", f"${w * args.capital:,.2f}"]
        for ticker, w in sorted(hrp_weights.items(), key=lambda x: x[1], reverse=True)
    ]
    print(tabulate(weights_summary, headers=["Asset", "Weight (%)", "Allocation ($)"], tablefmt="grid"))
    print(f"    Expected Return: {port_metrics['Annualized_Return'] * 100:.2f}% (Ann.)")
    print(f"    Volatility:      {port_metrics['Annualized_Volatility'] * 100:.2f}% (Ann.)")
    print(f"    Sharpe Ratio:    {port_metrics['Sharpe_Ratio']:.2f}")
    print(f"    Max Drawdown:    {port_metrics['Max_Drawdown'] * 100:.2f}%")

    # 4. Tail Risk Analysis
    port_daily_ret = (returns * pd.Series(hrp_weights)).sum(axis=1)
    tail_engine = TailRiskEngine(confidence_levels=[0.95, 0.99])
    tail_df = tail_engine.compute_all_metrics(port_daily_ret, portfolio_value=args.capital)

    print("\n[+] DOWNSIDE TAIL RISK (VaR & Conditional VaR / Expected Shortfall):")
    tail_table_display = tail_df[tail_df["Confidence_Level"] == "95%"][
        ["Method", "VaR_pct", "VaR_Dollar", "CVaR_Expected_Shortfall_pct", "CVaR_Dollar"]
    ].copy()
    tail_table_display["VaR_pct"] = tail_table_display["VaR_pct"].apply(lambda x: f"{x * 100:.2f}%")
    tail_table_display["VaR_Dollar"] = tail_table_display["VaR_Dollar"].apply(lambda x: f"${x:,.2f}")
    tail_table_display["CVaR_Expected_Shortfall_pct"] = tail_table_display["CVaR_Expected_Shortfall_pct"].apply(lambda x: f"{x * 100:.2f}%")
    tail_table_display["CVaR_Dollar"] = tail_table_display["CVaR_Dollar"].apply(lambda x: f"${x:,.2f}")
    print(tabulate(tail_table_display.values, headers=list(tail_table_display.columns), tablefmt="grid"))

    # 5. Regulatory Backtest
    rolling_var = port_daily_ret.rolling(window=252).apply(
        lambda x: TailRiskEngine.historical_var_cvar(x.values, 0.95)[0], raw=False
    ).dropna()
    backtester = RiskBacktester(confidence_level=0.95)
    bt_res = backtester.run_full_backtest(port_daily_ret.loc[rolling_var.index], rolling_var)

    print(f"\n[+] BASEL REGULATORY BACKTEST STATUS: {bt_res['Basel_Traffic_Light']}")
    print(f"    Observations: {bt_res['Total_Observations_T']} | Breaches: {bt_res['Total_Breaches_N']} (Expected: {bt_res['Expected_Breaches']})")
    print(f"    Kupiec POF LR: {bt_res['Kupiec_LR_Stat']} (p={bt_res['Kupiec_p_value']}) -> {'PASSED' if bt_res['Kupiec_Null_Accepted'] else 'FAILED'}")
    print(f"    Christoffersen: {bt_res['Christoffersen_Ind_LR_Stat']} (p={bt_res['Christoffersen_Ind_p_value']}) -> {'INDEPENDENT' if bt_res['Independence_Null_Accepted'] else 'CLUSTERED'}")

    # 6. Stress Testing
    stress_engine = StressTestEngine(random_state=42)
    crisis_df = stress_engine.simulate_historical_crisis(hrp_weights, portfolio_value=args.capital)

    print("\n[+] HISTORICAL CRISIS STRESS REPLAY:")
    stress_display = crisis_df[["Crisis_Scenario", "Portfolio_Impact_pct", "Estimated_PnL_Dollar", "Ending_Portfolio_Value"]].copy()
    stress_display["Portfolio_Impact_pct"] = stress_display["Portfolio_Impact_pct"].apply(lambda x: f"{x * 100:.2f}%")
    stress_display["Estimated_PnL_Dollar"] = stress_display["Estimated_PnL_Dollar"].apply(lambda x: f"${x:,.2f}")
    stress_display["Ending_Portfolio_Value"] = stress_display["Ending_Portfolio_Value"].apply(lambda x: f"${x:,.2f}")
    print(tabulate(stress_display.values, headers=list(stress_display.columns), tablefmt="grid"))

    # 7. Export Institutional Report
    out_file = Path(args.output_report)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    reporter = ExecutiveRiskReporter(author="Gabriel Proaño", institution="AlphaRisk Quantitative Analytics")
    report_md = reporter.generate_markdown_report(
        current_regime=current_regime_name,
        regime_durations=regime_durations,
        tail_risk_df=tail_df,
        backtest_results=bt_res,
        portfolio_metrics=port_metrics,
        portfolio_weights=hrp_weights,
        stress_df=crisis_df,
        portfolio_value=args.capital,
    )
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\n[💾] Institutional Risk Memorandum generated and saved to: {out_file.resolve()}")
    print("=" * 75)


if __name__ == "__main__":
    main()
