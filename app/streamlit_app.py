"""
AlphaRisk: Quantitative Market Regime Detection & Macro Stress-Testing Platform.
Author: Gabriel Proaño
Interactive Streamlit Application.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.data_loader import DataLoader
from src.data.macro_loader import MacroLoader
from src.features.risk_factors import RiskFactorEngine
from src.models.regime_detector import MarketRegimeDetector
from src.models.tail_risk import TailRiskEngine
from src.models.portfolio_opt import PortfolioOptimizer
from src.models.backtest import RiskBacktester
from src.models.stress_testing import StressTestEngine
from src.reporting.risk_report import ExecutiveRiskReporter

# Page configuration
st.set_page_config(
    page_title="AlphaRisk | Quantitative Market Regime & Stress-Testing",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #334155;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_market_data(tickers, start_date):
    loader = DataLoader(cache_dir=ROOT_DIR / "data" / "cache", use_cache=True)
    prices = loader.fetch_historical_prices(tickers=tickers, start_date=start_date)
    returns = loader.compute_log_returns(prices)
    return prices, returns


@st.cache_data(show_spinner=False)
def load_macro_data(start_date):
    macro = MacroLoader(cache_dir=ROOT_DIR / "data" / "cache", use_cache=True)
    return macro.fetch_macro_indicators(start_date=start_date)


def main():
    st.title("📈 AlphaRisk: Market Regime & Stress-Testing Platform")
    st.caption("Quantitative Risk Intelligence, Extreme Value Theory & Portfolio Allocation | **Lead Analyst: Gabriel Proaño**")

    # Sidebar Controls
    st.sidebar.header("⚙️ Model Configuration")

    default_universe = ["SPY", "QQQ", "GLD", "TLT", "BTC-USD"]
    selected_tickers = st.sidebar.multiselect(
        "Asset Universe",
        options=["SPY", "QQQ", "GLD", "TLT", "BTC-USD", "NVDA", "AAPL", "MSFT", "EEM", "IEF"],
        default=default_universe,
    )

    if not selected_tickers or len(selected_tickers) < 2:
        st.warning("Please select at least 2 assets for portfolio optimization.")
        selected_tickers = ["SPY", "TLT"]

    benchmark_ticker = st.sidebar.selectbox("Market Benchmark", options=selected_tickers, index=0)
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2018-01-01")).strftime("%Y-%m-%d")

    portfolio_capital = st.sidebar.number_input(
        "Portfolio Capital ($)", min_value=10_000.0, max_value=100_000_000.0, value=1_000_000.0, step=50_000.0
    )

    alloc_method = st.sidebar.selectbox(
        "Allocation Model",
        options=["Hierarchical Risk Parity (HRP)", "Maximum Sharpe Ratio", "Minimum Volatility", "Equal Weight (1/N)"],
        index=0,
    )

    st.sidebar.subheader("Regime Classification")
    model_type = st.sidebar.radio("Model Class", options=["HMM", "GMM"], index=0, horizontal=True)
    n_regimes = st.sidebar.slider("Number of Latent Regimes", min_value=2, max_value=4, value=3)

    confidence_level = st.sidebar.select_slider(
        "VaR Confidence Level (1 - α)", options=[0.90, 0.95, 0.99], value=0.95
    )

    # Load Data
    with st.spinner("Synchronizing financial market data and computing risk factors..."):
        prices, returns = load_market_data(selected_tickers, start_date)
        macro_df = load_macro_data(start_date)

    if prices.empty or returns.empty:
        st.error("Error retrieving price series. Check ticker symbols or internet connection.")
        return

    # Benchmark returns and realized volatility for Regime Detection
    bm_returns = returns[benchmark_ticker]
    bm_prices = prices[benchmark_ticker]
    bm_realized_vol = RiskFactorEngine.compute_rolling_realized_volatility(bm_returns, window=21)

    regime_features = pd.concat([bm_returns, bm_realized_vol], axis=1).dropna()
    regime_features.columns = ["log_return", "realized_vol"]

    # Fit Regime Detector
    regime_detector = MarketRegimeDetector(
        n_regimes=n_regimes, model_type=model_type, covariance_type="full", random_state=42
    )
    regime_detector.fit(regime_features)
    predicted_regimes = regime_detector.predict(regime_features)
    regime_series = pd.Series(predicted_regimes, index=regime_features.index, name="Regime")
    current_regime_id = int(regime_series.iloc[-1])
    current_regime_name = regime_detector.REGIME_NAMES.get(current_regime_id, f"Regime {current_regime_id}")
    regime_durations = regime_detector.calculate_expected_durations()

    # Portfolio Optimization
    opt = PortfolioOptimizer(risk_free_rate=0.045)
    if "HRP" in alloc_method:
        weights = opt.optimize_hrp(returns)
    elif "Sharpe" in alloc_method:
        weights = opt.optimize_max_sharpe(returns)
    elif "Min" in alloc_method:
        weights = opt.optimize_min_volatility(returns)
    else:
        weights = opt.optimize_equal_weight(returns)

    port_metrics = opt.compute_portfolio_metrics(weights, returns)
    portfolio_daily_returns = (returns * pd.Series(weights)).sum(axis=1)

    # Tail Risk & Extreme Value Analysis
    tail_engine = TailRiskEngine(confidence_levels=[0.95, 0.99])
    tail_df = tail_engine.compute_all_metrics(portfolio_daily_returns, portfolio_value=portfolio_capital)

    # Backtesting
    rolling_var_95 = portfolio_daily_returns.rolling(window=252).apply(
        lambda x: TailRiskEngine.historical_var_cvar(x.values, 0.95)[0], raw=False
    ).dropna()
    backtester = RiskBacktester(confidence_level=0.95)
    aligned_returns = portfolio_daily_returns.loc[rolling_var_95.index]
    backtest_res = backtester.run_full_backtest(aligned_returns, rolling_var_95)

    # Stress Testing Engine
    stress_engine = StressTestEngine(random_state=42)
    crisis_df = stress_engine.simulate_historical_crisis(weights, portfolio_value=portfolio_capital)

    # Top KPI Metrics Header
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        color_dot = "🟢" if current_regime_id == 0 else ("🟡" if current_regime_id == 1 else "🔴")
        st.metric("Market Regime", f"{color_dot} Regime {current_regime_id}", current_regime_name.split("(")[0])
    with col2:
        st.metric("Expected Return (Ann.)", f"{port_metrics['Annualized_Return'] * 100:.1f}%")
    with col3:
        st.metric("Annualized Volatility (σ)", f"{port_metrics['Annualized_Volatility'] * 100:.1f}%")
    with col4:
        st.metric("Sharpe Ratio (rf=4.5%)", f"{port_metrics['Sharpe_Ratio']:.2f}")
    with col5:
        st.metric("Max Historical Drawdown", f"{port_metrics['Max_Drawdown'] * 100:.1f}%")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌐 Market Regimes",
        "⚖️ Portfolio Allocation (HRP)",
        "📉 Extreme Tail Risk (VaR/CVaR)",
        "⚡ Macro Stress-Testing & Monte Carlo",
        "📄 Executive Risk Memorandum",
    ])

    # ---------------- TAB 1: MARKET REGIMES ----------------
    with tab1:
        st.subheader("Unsupervised Market Regime Dynamics & Hidden States")
        st.markdown(
            f"The **{model_type}** classifies market conditions into **{n_regimes} volatility-sorted regimes**. "
            "Latent states are dynamically updated using log-returns and 21-day realized volatility."
        )

        # Plot Benchmark Price with Regimes
        aligned_prices = bm_prices.loc[regime_series.index]
        fig_regime = go.Figure()
        fig_regime.add_trace(go.Scatter(
            x=aligned_prices.index, y=aligned_prices.values, mode="lines", name=f"{benchmark_ticker} Price",
            line=dict(color="#94a3b8", width=1.5)
        ))

        # Add regime background shading
        colors = [MarketRegimeDetector.REGIME_COLORS.get(r, "#64748b") for r in regime_series.values]
        fig_regime.add_trace(go.Scatter(
            x=aligned_prices.index, y=aligned_prices.values, mode="markers",
            marker=dict(color=colors, size=4, opacity=0.8),
            name="Regime State"
        ))

        fig_regime.update_layout(
            title=f"{benchmark_ticker} Price Series Classified by Latent Regime ({model_type})",
            template="plotly_dark",
            height=450,
            xaxis_title="Date",
            yaxis_title="Asset Price ($)",
        )
        st.plotly_chart(fig_regime, use_container_width=True)

        col_trans, col_info = st.columns(2)
        with col_trans:
            st.markdown("#### Transition Probability Matrix $\mathbf{P}$")
            trans_df = regime_detector.get_transition_matrix()
            if trans_df is not None:
                st.dataframe(trans_df.style.format("{:.3f}").background_gradient(cmap="Blues"), use_container_width=True)

            st.markdown("#### Expected Regime Duration")
            dur_cols = st.columns(len(regime_durations))
            for i, (k, dur) in enumerate(regime_durations.items()):
                with dur_cols[i]:
                    st.metric(k, f"{dur:.1f} days", "Average Persistence")

        with col_info:
            st.markdown("#### Statistical Model Validation (AIC / BIC Selection)")
            ic_df = MarketRegimeDetector.select_optimal_regimes(regime_features, max_regimes=5)
            fig_ic = px.line(
                ic_df, y=["AIC", "BIC"], markers=True, title="Information Criteria vs Number of Regimes (k)",
                template="plotly_dark"
            )
            fig_ic.update_layout(height=280)
            st.plotly_chart(fig_ic, use_container_width=True)
            st.caption("Lower AIC/BIC indicates superior balance between goodness-of-fit and model parsimony.")

    # ---------------- TAB 2: PORTFOLIO ALLOCATION ----------------
    with tab2:
        st.subheader(f"Portfolio Weights Allocation: {alloc_method}")
        col_w1, col_w2 = st.columns([1, 2])

        with col_w1:
            weights_df = pd.DataFrame([
                {"Asset": k, "Weight": v, "Capital ($)": v * portfolio_capital}
                for k, v in weights.items()
            ]).sort_values(by="Weight", ascending=False)

            fig_pie = px.pie(
                weights_df, values="Weight", names="Asset", hole=0.4,
                title=f"Optimal Weights ({alloc_method})", template="plotly_dark"
            )
            fig_pie.update_traces(textinfo="label+percent")
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_w2:
            st.markdown("#### Cumulative Wealth Evolution & Underwater Drawdown")
            cum_growth = (1.0 + portfolio_daily_returns).cumprod() * portfolio_capital
            running_max = cum_growth.cummax()
            dd_series = (cum_growth - running_max) / running_max

            fig_perf = go.Figure()
            fig_perf.add_trace(go.Scatter(
                x=cum_growth.index, y=cum_growth.values, mode="lines", name="Portfolio Value ($)",
                line=dict(color="#38bdf8", width=2)
            ))
            fig_perf.update_layout(title="Cumulative Portfolio Growth", template="plotly_dark", height=240)
            st.plotly_chart(fig_perf, use_container_width=True)

            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=dd_series.index, y=dd_series.values, mode="lines", fill="tozeroy",
                name="Drawdown", line=dict(color="#f87171", width=1.5)
            ))
            fig_dd.update_layout(title="Underwater Drawdown Timeline", template="plotly_dark", height=200, yaxis_tickformat=".1%")
            st.plotly_chart(fig_dd, use_container_width=True)

    # ---------------- TAB 3: TAIL RISK (VAR / CVAR) ----------------
    with tab3:
        st.subheader("Downside Extreme Value Theory & Basel Backtesting")
        st.markdown(
            "Value at Risk (VaR) and **Conditional Value at Risk (Expected Shortfall - CVaR)**. "
            "CVaR fulfills the axiomatic properties of a coherent risk measure (subadditivity: $ES(X+Y) \le ES(X) + ES(Y)$)."
        )

        st.dataframe(
            tail_df.style.format({
                "VaR_pct": "{:.2%}",
                "VaR_Dollar": "${:,.2f}",
                "CVaR_Expected_Shortfall_pct": "{:.2%}",
                "CVaR_Dollar": "${:,.2f}",
            }),
            use_container_width=True,
        )

        col_hist, col_backtest = st.columns([3, 2])
        with col_hist:
            var_95_daily = float(tail_df[(tail_df['Confidence_Level']=='95%') & (tail_df['Method']=='Historical Simulation')]['VaR_pct'].values[0])
            cvar_95_daily = float(tail_df[(tail_df['Confidence_Level']=='95%') & (tail_df['Method']=='Historical Simulation')]['CVaR_Expected_Shortfall_pct'].values[0])

            fig_dist = px.histogram(
                portfolio_daily_returns, nbins=80, title="Daily Return Distribution with Tail Risk Cutoffs",
                template="plotly_dark", labels={"value": "Daily Return"}
            )
            fig_dist.add_vline(x=-var_95_daily, line_dash="dash", line_color="#eab308", annotation_text="VaR 95%")
            fig_dist.add_vline(x=-cvar_95_daily, line_dash="dash", line_color="#ef4444", annotation_text="CVaR 95% (Expected Shortfall)")
            fig_dist.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig_dist, use_container_width=True)

        with col_backtest:
            st.markdown("#### Basel Committee Regulatory Backtest")
            t_light = backtest_res["Basel_Traffic_Light"]
            badge_icon = "🟢" if "GREEN" in t_light else "🟡"
            st.info(f"{badge_icon} **Status: {t_light}**")

            b_data = [
                {"Test": "Total Trading Days (T)", "Value": str(backtest_res["Total_Observations_T"])},
                {"Test": "Empirical Breaches (N)", "Value": str(backtest_res["Total_Breaches_N"])},
                {"Test": "Expected Breaches", "Value": str(backtest_res["Expected_Breaches"])},
                {"Test": "Kupiec POF LR Stat", "Value": f"{backtest_res['Kupiec_LR_Stat']} (p={backtest_res['Kupiec_p_value']})"},
                {"Test": "Kupiec Test Result", "Value": "PASSED ✅" if backtest_res['Kupiec_Null_Accepted'] else "REJECTED ❌"},
                {"Test": "Christoffersen Indep. LR", "Value": f"{backtest_res['Christoffersen_Ind_LR_Stat']} (p={backtest_res['Christoffersen_Ind_p_value']})"},
                {"Test": "Independence Result", "Value": "PASSED ✅" if backtest_res['Independence_Null_Accepted'] else "CLUSTERING ⚠️"},
            ]
            st.table(pd.DataFrame(b_data))

    # ---------------- TAB 4: STRESS TESTING & MONTE CARLO ----------------
    with tab4:
        st.subheader("Macroeconomic Stress-Testing & Correlated Monte Carlo Simulation")

        col_mc, col_custom = st.columns([3, 2])
        with col_mc:
            st.markdown("#### 30-Day Correlated Monte Carlo Fan Chart (5,000 Paths)")
            mc_res = stress_engine.run_correlated_monte_carlo(
                returns, weights, n_simulations=2000, horizon_days=30, initial_portfolio_value=portfolio_capital
            )
            wealth_paths = mc_res["wealth_paths"]

            fig_mc = go.Figure()
            # Plot sample paths
            sample_indices = np.random.choice(wealth_paths.shape[0], size=50, replace=False)
            for idx in sample_indices:
                fig_mc.add_trace(go.Scatter(
                    y=wealth_paths[idx], mode="lines", line=dict(color="rgba(148, 163, 184, 0.15)", width=1),
                    showlegend=False
                ))

            # Add percentiles
            days = np.arange(wealth_paths.shape[1])
            p5 = np.percentile(wealth_paths, 5, axis=0)
            p50 = np.percentile(wealth_paths, 50, axis=0)
            p95 = np.percentile(wealth_paths, 95, axis=0)

            fig_mc.add_trace(go.Scatter(x=days, y=p95, mode="lines", name="95th Percentile (Bull)", line=dict(color="#22c55e", width=2)))
            fig_mc.add_trace(go.Scatter(x=days, y=p50, mode="lines", name="Median Path", line=dict(color="#38bdf8", width=2)))
            fig_mc.add_trace(go.Scatter(x=days, y=p5, mode="lines", name="5th Percentile (Tail Risk)", line=dict(color="#ef4444", width=2)))

            fig_mc.update_layout(
                title=f"Terminal Value Range (Median: ${mc_res['median_terminal_value']:,.0f})",
                template="plotly_dark", height=380, xaxis_title="Trading Days Ahead", yaxis_title="Portfolio Value ($)"
            )
            st.plotly_chart(fig_mc, use_container_width=True)

        with col_custom:
            st.markdown("#### Custom Macro Shock Simulator")
            shock_eq = st.slider("Equity Market Shock (%)", min_value=-35.0, max_value=15.0, value=-10.0, step=1.0)
            shock_rate = st.slider("Fed Interest Rate Shock (bps)", min_value=-200.0, max_value=300.0, value=75.0, step=25.0)
            shock_vix = st.slider("Volatility Index Shock (VIX %)", min_value=-20.0, max_value=120.0, value=30.0, step=5.0)

            custom_res = stress_engine.simulate_macro_factor_shock(
                weights, equity_shock_pct=shock_eq / 100.0, rate_shock_bps=shock_rate,
                vol_shock_pct=shock_vix / 100.0, portfolio_value=portfolio_capital
            )

            pnl_val = custom_res["Simulated_PnL_Dollar"]
            pnl_color = "#22c55e" if pnl_val >= 0 else "#ef4444"
            st.markdown(
                f"""
                <div class="metric-card">
                    <h4>Simulated Scenario Outcome</h4>
                    <p style="font-size: 24px; font-weight: bold; color: {pnl_color};">
                        {custom_res['Simulated_Portfolio_Return_pct'] * 100:+.2f}% (${pnl_val:+,.2f})
                    </p>
                    <p>Ending Capital: <strong>${custom_res['Ending_Portfolio_Value']:,.2f}</strong></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("#### Historical Crisis Replay on Current Portfolio")
        st.dataframe(
            crisis_df.style.format({
                "Portfolio_Impact_pct": "{:.2%}",
                "Estimated_PnL_Dollar": "${:,.2f}",
                "Ending_Portfolio_Value": "${:,.2f}",
                "Peak_VIX_Level": "{:.1f}",
            }),
            use_container_width=True,
        )

    # ---------------- TAB 5: EXECUTIVE RISK MEMO ----------------
    with tab5:
        st.subheader("Institutional Risk Memorandum")
        reporter = ExecutiveRiskReporter(author="Gabriel Proaño", institution="AlphaRisk Quantitative Analytics")
        memo_markdown = reporter.generate_markdown_report(
            current_regime=current_regime_name,
            regime_durations=regime_durations,
            tail_risk_df=tail_df,
            backtest_results=backtest_res,
            portfolio_metrics=port_metrics,
            portfolio_weights=weights,
            stress_df=crisis_df,
            portfolio_value=portfolio_capital,
        )

        st.download_button(
            label="📥 Download Executive Risk Memo (.md)",
            data=memo_markdown,
            file_name=f"AlphaRisk_Memorandum_{pd.Timestamp.now().strftime('%Y%m%d')}.md",
            mime="text/markdown",
        )

        st.markdown(memo_markdown)


if __name__ == "__main__":
    main()
