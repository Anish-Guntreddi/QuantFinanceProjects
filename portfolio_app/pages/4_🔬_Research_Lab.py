"""
Research Lab — Interactive standalone simulation modules.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Research Lab", page_icon="🔬", layout="wide")

# CSS
_ASSETS = Path(__file__).parent.parent / "assets"
css_file = _ASSETS / "style.css"
if css_file.exists():
    st.markdown(f"<style>{css_file.read_text()}</style>", unsafe_allow_html=True)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from components.theme import plotly_layout, ACCENT, PROFIT_GREEN, LOSS_RED, TEXT_MUTED
from components.metrics_panel import render_metrics_panel
from components.equity_curve import render_equity_curve

st.markdown(
    '<p class="hero-title" style="margin-bottom: 0.25rem;">Research Lab</p>'
    '<h1 style="margin-top: 0;">Interactive Simulations</h1>'
    '<p style="color: #9ca3af; margin-bottom: 1.5rem;">Standalone modules demonstrating quantitative methods. '
    'All simulations run in real-time.</p>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Module tabs
# ---------------------------------------------------------------------------
tab_mc, tab_opt, tab_bt, tab_vol, tab_risk = st.tabs([
    "Monte Carlo", "Portfolio Optimizer", "Backtest Engine", "Vol Surface", "Risk Calculator"
])

# ===========================================================================
# TAB 1: Monte Carlo
# ===========================================================================
with tab_mc:
    from simulations.monte_carlo import run_monte_carlo

    col1, col2 = st.columns([1, 2])
    with col1:
        mc_model = st.selectbox("Model", ["GBM", "Heston", "Jump-Diffusion"], key="mc_model")
        mc_s0 = st.number_input("Initial Price", value=100.0, min_value=1.0, key="mc_s0")
        mc_mu = st.slider("Drift (μ)", 0.0, 0.20, 0.08, 0.01, key="mc_mu")
        mc_sigma = st.slider("Volatility (σ)", 0.05, 0.60, 0.20, 0.01, key="mc_sigma")
        mc_paths = st.slider("Paths", 100, 5000, 1000, 100, key="mc_paths")
        mc_horizon = st.selectbox("Horizon", ["3M", "6M", "1Y", "2Y"], index=2, key="mc_horizon")
        mc_months = {"3M": 3, "6M": 6, "1Y": 12, "2Y": 24}[mc_horizon]
        run_mc = st.button("▶ Run Monte Carlo", type="primary", width="stretch", key="run_mc")

    with col2:
        if run_mc:
            with st.spinner("Simulating..."):
                result = run_monte_carlo(
                    model=mc_model, S0=mc_s0, mu=mc_mu, sigma=mc_sigma,
                    T_months=mc_months, n_paths=mc_paths,
                )

            # Fan chart
            fan = result["fan"]
            n = result["n_steps"]
            x = list(range(n))

            fig = go.Figure()
            # 5-95 band
            fig.add_trace(go.Scatter(x=x, y=fan[95], line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=x, y=fan[5], fill="tonexty", fillcolor="rgba(245,158,11,0.08)",
                                     line=dict(width=0), name="5th-95th"))
            # 25-75 band
            fig.add_trace(go.Scatter(x=x, y=fan[75], line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=x, y=fan[25], fill="tonexty", fillcolor="rgba(245,158,11,0.18)",
                                     line=dict(width=0), name="25th-75th"))
            # Median
            fig.add_trace(go.Scatter(x=x, y=fan[50], line=dict(color=ACCENT, width=2), name="Median"))
            # VaR line
            var_val = mc_s0 * (1 + result["var_95"])
            fig.add_hline(y=var_val, line_dash="dash", line_color=LOSS_RED, annotation_text=f"VaR 95%: ${var_val:.1f}")

            fig.update_layout(**plotly_layout(f"{mc_model} — {mc_paths} paths, {mc_horizon}", height=400))
            fig.update_xaxes(title_text="Trading Days")
            fig.update_yaxes(title_text="Price")
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

            # Stats
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean Return", f"{result['mean_return']*100:+.2f}%")
            c2.metric("Std Dev", f"{result['std_return']*100:.2f}%")
            c3.metric("VaR (95%)", f"{result['var_95']*100:+.2f}%")
            c4.metric("CVaR (95%)", f"{result['cvar_95']*100:+.2f}%")
        else:
            st.info("Configure parameters and click **Run Monte Carlo** to simulate.")

# ===========================================================================
# TAB 2: Portfolio Optimizer
# ===========================================================================
with tab_opt:
    from simulations.portfolio_optimizer import run_optimization

    col1, col2 = st.columns([1, 2])
    with col1:
        opt_assets = st.multiselect(
            "Assets", ["SPY", "AGG", "GLD", "VNQ", "EFA", "QQQ", "TLT", "IWM", "EEM", "DBC"],
            default=["SPY", "AGG", "GLD", "VNQ", "EFA"], key="opt_assets",
        )
        opt_method = st.selectbox("Method", ["Mean-Variance", "Risk Parity", "Min Variance"], key="opt_method")
        opt_max_w = st.slider("Max Weight", 0.1, 1.0, 0.4, 0.05, key="opt_max_w")
        run_opt = st.button("▶ Optimize", type="primary", width="stretch", key="run_opt")

    with col2:
        if run_opt and len(opt_assets) >= 2:
            with st.spinner("Optimizing..."):
                result = run_optimization(symbols=opt_assets, method=opt_method, max_weight=opt_max_w)

            if "error" in result:
                st.error(result["error"])
            else:
                # Weights bar chart
                weights = result["weights"]
                fig_w = go.Figure(go.Bar(
                    x=list(weights.keys()),
                    y=[v * 100 for v in weights.values()],
                    marker_color=ACCENT,
                    text=[f"{v*100:.1f}%" for v in weights.values()],
                    textposition="outside",
                    textfont=dict(family="JetBrains Mono, monospace", size=11),
                ))
                fig_w.update_layout(**plotly_layout(f"Optimal Weights — {opt_method}", height=300))
                fig_w.update_yaxes(title_text="Weight (%)")
                st.plotly_chart(fig_w, width="stretch", config={"displayModeBar": False})

                # Efficient frontier
                frontier = result.get("frontier", {})
                if frontier.get("risk") and frontier.get("return"):
                    fig_ef = go.Figure()
                    fig_ef.add_trace(go.Scatter(
                        x=frontier["risk"], y=frontier["return"],
                        mode="lines", name="Efficient Frontier",
                        line=dict(color=ACCENT, width=2),
                    ))
                    fig_ef.add_trace(go.Scatter(
                        x=[result["volatility"] * 100],
                        y=[result["expected_return"] * 100],
                        mode="markers", name="Optimal",
                        marker=dict(color=PROFIT_GREEN, size=12, symbol="star"),
                    ))
                    fig_ef.update_layout(**plotly_layout("Efficient Frontier", height=350))
                    fig_ef.update_xaxes(title_text="Annualized Volatility (%)")
                    fig_ef.update_yaxes(title_text="Annualized Return (%)")
                    st.plotly_chart(fig_ef, width="stretch", config={"displayModeBar": False})

                c1, c2, c3 = st.columns(3)
                c1.metric("Expected Return", f"{result['expected_return']*100:.2f}%")
                c2.metric("Volatility", f"{result['volatility']*100:.2f}%")
                c3.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.2f}")
        elif run_opt:
            st.warning("Select at least 2 assets.")
        else:
            st.info("Configure parameters and click **Optimize** to run.")

# ===========================================================================
# TAB 3: Backtest Engine
# ===========================================================================
with tab_bt:
    from simulations.backtest_runner import run_backtest

    col1, col2 = st.columns([1, 2])
    with col1:
        bt_strategy = st.selectbox("Strategy", ["Momentum", "Mean Reversion", "MA Crossover"], key="bt_strat")
        bt_symbol = st.selectbox("Asset", ["SPY", "QQQ", "AAPL", "MSFT", "GLD", "BTC-USD"], key="bt_sym")
        bt_start = st.date_input("Start", value=None, key="bt_start")
        bt_end = st.date_input("End", value=None, key="bt_end")

        if bt_strategy == "Momentum":
            bt_lookback = st.slider("Lookback (days)", 5, 100, 20, key="bt_lb")
        elif bt_strategy == "Mean Reversion":
            bt_lookback = st.slider("Lookback (days)", 5, 100, 20, key="bt_lb_mr")
            bt_entry_z = st.slider("Entry Z-Score", 1.0, 3.0, 2.0, 0.1, key="bt_ez")
            bt_exit_z = st.slider("Exit Z-Score", 0.0, 1.5, 0.5, 0.1, key="bt_xz")
        else:
            bt_ma_fast = st.slider("Fast MA", 5, 50, 10, key="bt_maf")
            bt_ma_slow = st.slider("Slow MA", 20, 200, 50, key="bt_mas")

        run_bt = st.button("▶ Run Backtest", type="primary", width="stretch", key="run_bt")

    with col2:
        if run_bt:
            start_str = str(bt_start) if bt_start else "2020-01-01"
            end_str = str(bt_end) if bt_end else "2024-12-31"

            kwargs = dict(strategy=bt_strategy, symbol=bt_symbol, start=start_str, end=end_str)
            if bt_strategy == "Momentum":
                kwargs["lookback"] = bt_lookback
            elif bt_strategy == "Mean Reversion":
                kwargs["lookback"] = bt_lookback
                kwargs["entry_zscore"] = bt_entry_z
                kwargs["exit_zscore"] = bt_exit_z
            else:
                kwargs["ma_fast"] = bt_ma_fast
                kwargs["ma_slow"] = bt_ma_slow

            with st.spinner("Running backtest..."):
                result = run_backtest(**kwargs)

            if "error" in result:
                st.error(result["error"])
            else:
                render_equity_curve(result, title=f"{bt_strategy} on {bt_symbol}")
                render_metrics_panel(result["metrics"], cols=4)
        else:
            st.info("Configure parameters and click **Run Backtest** to simulate.")

# ===========================================================================
# TAB 4: Vol Surface
# ===========================================================================
with tab_vol:
    col1, col2 = st.columns([1, 2])
    with col1:
        vol_spot = st.number_input("Spot Price", value=100.0, min_value=1.0, key="vol_spot")
        vol_rf = st.slider("Risk-Free Rate", 0.0, 0.10, 0.04, 0.005, key="vol_rf")
        vol_base = st.slider("Base Vol (ATM)", 0.05, 0.60, 0.20, 0.01, key="vol_base")
        vol_skew = st.slider("Skew", -0.5, 0.0, -0.15, 0.01, key="vol_skew")
        vol_smile = st.slider("Smile (Convexity)", 0.0, 0.5, 0.10, 0.01, key="vol_smile")
        run_vol = st.button("▶ Build Surface", type="primary", width="stretch", key="run_vol")

    with col2:
        if run_vol:
            # SVI-like parameterization
            strikes = np.linspace(0.7 * vol_spot, 1.3 * vol_spot, 30)
            tenors = np.array([0.08, 0.17, 0.25, 0.5, 0.75, 1.0])  # ~1M to 1Y
            tenor_labels = ["1M", "2M", "3M", "6M", "9M", "1Y"]

            moneyness = np.log(strikes / vol_spot)

            Z = np.zeros((len(tenors), len(strikes)))
            for i, T in enumerate(tenors):
                term_adj = vol_base * (1 + 0.1 * (1 / T - 1))  # term structure
                for j, m in enumerate(moneyness):
                    Z[i, j] = term_adj + vol_skew * m / np.sqrt(T) + vol_smile * m**2

            Z = np.clip(Z, 0.01, 2.0)

            fig = go.Figure(data=[go.Surface(
                x=strikes, y=tenors, z=Z,
                colorscale=[[0, "#3b82f6"], [0.5, "#f59e0b"], [1, "#ef4444"]],
                showscale=False,
                hovertemplate="Strike: %{x:.1f}<br>Tenor: %{y:.2f}Y<br>IV: %{z:.1%}<extra></extra>",
            )])

            fig.update_layout(
                title=dict(text="Implied Volatility Surface", font=dict(family="DM Serif Display", size=16, color="#e5e7eb")),
                scene=dict(
                    xaxis=dict(title="Strike", backgroundcolor="rgba(0,0,0,0)",
                               gridcolor="rgba(31,41,55,0.5)", tickfont=dict(family="JetBrains Mono", size=9, color="#6b7280")),
                    yaxis=dict(title="Tenor (Y)", backgroundcolor="rgba(0,0,0,0)",
                               gridcolor="rgba(31,41,55,0.5)", tickfont=dict(family="JetBrains Mono", size=9, color="#6b7280")),
                    zaxis=dict(title="Implied Vol", backgroundcolor="rgba(0,0,0,0)",
                               gridcolor="rgba(31,41,55,0.5)", tickformat=".0%",
                               tickfont=dict(family="JetBrains Mono", size=9, color="#6b7280")),
                    bgcolor="rgba(0,0,0,0)",
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                height=500,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

            # Cross-sections
            st.markdown("**Smile Cross-Sections**")
            fig_smile = go.Figure()
            colors_list = ["#f59e0b", "#3b82f6", "#10b981", "#f43f5e", "#8b5cf6", "#06b6d4"]
            for i, (T, label) in enumerate(zip(tenors, tenor_labels)):
                fig_smile.add_trace(go.Scatter(
                    x=strikes.tolist(), y=Z[i].tolist(),
                    name=label, line=dict(color=colors_list[i % len(colors_list)], width=1.5),
                ))
            fig_smile.update_layout(**plotly_layout("Volatility Smile by Tenor", height=300))
            fig_smile.update_xaxes(title_text="Strike")
            fig_smile.update_yaxes(title_text="Implied Vol", tickformat=".0%")
            st.plotly_chart(fig_smile, width="stretch", config={"displayModeBar": False})
        else:
            st.info("Configure parameters and click **Build Surface** to generate.")

# ===========================================================================
# TAB 5: Risk Calculator
# ===========================================================================
with tab_risk:
    col1, col2 = st.columns([1, 2])
    with col1:
        risk_assets = st.multiselect(
            "Portfolio Assets", ["SPY", "AGG", "GLD", "VNQ", "EFA", "QQQ", "TLT"],
            default=["SPY", "AGG", "GLD"], key="risk_assets",
        )
        risk_method = st.selectbox("VaR Method", ["Historical", "Parametric", "Monte Carlo"], key="risk_method")
        risk_conf = st.select_slider("Confidence", [0.90, 0.95, 0.99], value=0.95, key="risk_conf")
        risk_horizon = st.selectbox("Horizon", ["1D", "5D", "10D", "21D"], index=1, key="risk_horizon")
        risk_portfolio_val = st.number_input("Portfolio Value ($)", value=1_000_000, step=100_000, key="risk_pv")
        run_risk = st.button("▶ Calculate Risk", type="primary", width="stretch", key="run_risk")

    with col2:
        if run_risk and len(risk_assets) >= 1:
            from simulations.backtest_runner import fetch_data
            import pandas as pd

            horizon_days = {"1D": 1, "5D": 5, "10D": 10, "21D": 21}[risk_horizon]
            n_assets = len(risk_assets)
            equal_weight = 1.0 / n_assets

            # Fetch and compute portfolio returns
            with st.spinner("Fetching data..."):
                all_returns = []
                for sym in risk_assets:
                    data = fetch_data(sym, "2019-01-01", "2024-12-31")
                    if not data.empty:
                        close = data["Close"].squeeze()
                        all_returns.append(close.pct_change().dropna())

            if all_returns:
                df = pd.concat(all_returns, axis=1).dropna()
                df.columns = risk_assets[:df.shape[1]]
                port_returns = (df * equal_weight).sum(axis=1)

                # Scale to horizon
                if horizon_days > 1:
                    port_returns = port_returns.rolling(horizon_days).sum().dropna()

                returns_arr = port_returns.values

                if risk_method == "Historical":
                    var = float(np.percentile(returns_arr, (1 - risk_conf) * 100))
                    cvar = float(returns_arr[returns_arr <= var].mean())
                elif risk_method == "Parametric":
                    from scipy.stats import norm
                    mu = returns_arr.mean()
                    sigma = returns_arr.std()
                    var = float(norm.ppf(1 - risk_conf, mu, sigma))
                    cvar = float(mu - sigma * norm.pdf(norm.ppf(1 - risk_conf)) / (1 - risk_conf))
                else:  # Monte Carlo
                    rng = np.random.default_rng(42)
                    mu = returns_arr.mean()
                    sigma = returns_arr.std()
                    sim = rng.normal(mu, sigma, 50000)
                    var = float(np.percentile(sim, (1 - risk_conf) * 100))
                    cvar = float(sim[sim <= var].mean())

                var_dollar = var * risk_portfolio_val
                cvar_dollar = cvar * risk_portfolio_val

                # Metrics
                c1, c2, c3, c4 = st.columns(4)
                c1.metric(f"VaR ({risk_conf:.0%})", f"${var_dollar:,.0f}")
                c2.metric(f"CVaR ({risk_conf:.0%})", f"${cvar_dollar:,.0f}")
                c3.metric("VaR %", f"{var*100:+.2f}%")
                c4.metric("CVaR %", f"{cvar*100:+.2f}%")

                # Distribution chart
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=(returns_arr * 100).tolist(),
                    nbinsx=80,
                    marker_color="rgba(245,158,11,0.4)",
                    name="Returns",
                ))
                fig.add_vline(x=var * 100, line_dash="dash", line_color=LOSS_RED,
                             annotation_text=f"VaR: {var*100:.2f}%")
                fig.add_vline(x=cvar * 100, line_dash="dot", line_color="#f43f5e",
                             annotation_text=f"CVaR: {cvar*100:.2f}%")

                fig.update_layout(**plotly_layout(
                    f"Portfolio Return Distribution — {risk_horizon} horizon, {risk_method}", height=350))
                fig.update_xaxes(title_text="Return (%)")
                fig.update_yaxes(title_text="Frequency")
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

                # Stress scenarios
                st.markdown("**Stress Scenarios**")
                stress = {
                    "Normal Day": float(np.percentile(returns_arr, 50)),
                    "1-Sigma Down": float(returns_arr.mean() - returns_arr.std()),
                    "2-Sigma Down": float(returns_arr.mean() - 2 * returns_arr.std()),
                    "Worst Day": float(returns_arr.min()),
                    "Best Day": float(returns_arr.max()),
                }
                stress_html = "<table style='width:100%; border-collapse:collapse; background:#111827;'>"
                for scenario, ret in stress.items():
                    color = "#10b981" if ret > 0 else "#ef4444"
                    dollar = ret * risk_portfolio_val
                    stress_html += (
                        f"<tr><td style='padding:8px; font-family:DM Sans; color:#e5e7eb; border-bottom:1px solid #1f2937;'>{scenario}</td>"
                        f"<td style='padding:8px; text-align:right; font-family:JetBrains Mono; color:{color}; border-bottom:1px solid #1f2937;'>{ret*100:+.2f}%</td>"
                        f"<td style='padding:8px; text-align:right; font-family:JetBrains Mono; color:{color}; border-bottom:1px solid #1f2937;'>${dollar:+,.0f}</td></tr>"
                    )
                stress_html += "</table>"
                st.markdown(stress_html, unsafe_allow_html=True)
            else:
                st.error("Could not fetch data for selected assets.")
        elif run_risk:
            st.warning("Select at least 1 asset.")
        else:
            st.info("Configure parameters and click **Calculate Risk** to run.")
