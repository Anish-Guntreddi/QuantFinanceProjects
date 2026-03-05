"""Research Lab — 5 standalone interactive modules."""

import streamlit as st
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.theme import load_css, get_plotly_layout, PROFIT_COLOR, LOSS_COLOR
import plotly.graph_objects as go

load_css()

st.markdown("#### 🔬 Research Lab")
st.markdown("Interactive quantitative tools — adjust parameters and see results in real time.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Monte Carlo", "Portfolio Optimizer", "Backtest Engine", "Vol Surface", "Risk Calculator"])

# ==================== MODULE 1: Monte Carlo ====================
with tab1:
    st.markdown("##### Monte Carlo Simulator")
    c1, c2, c3 = st.columns(3)
    with c1:
        mc_mu = st.slider("Expected Return (%)", 0.0, 30.0, 8.0, 1.0, key="mc_mu") / 100
        mc_sigma = st.slider("Volatility (%)", 5.0, 60.0, 20.0, 1.0, key="mc_sigma") / 100
    with c2:
        mc_paths = st.slider("Paths", 100, 10000, 1000, 100, key="mc_paths")
        mc_horizon = st.selectbox("Horizon", ["3M", "6M", "1Y", "2Y"], index=2, key="mc_horizon")
    with c3:
        mc_initial = st.number_input("Initial Value ($)", 10000, 10000000, 100000, 10000, key="mc_init")
        mc_confidence = st.slider("VaR Confidence (%)", 90, 99, 95, 1, key="mc_conf")

    if st.button("Run Simulation", key="mc_run"):
        horizon_days = {"3M": 63, "6M": 126, "1Y": 252, "2Y": 504}[mc_horizon]
        np.random.seed(42)
        dt = 1 / 252
        paths = np.zeros((mc_paths, horizon_days + 1))
        paths[:, 0] = mc_initial
        for t in range(1, horizon_days + 1):
            z = np.random.randn(mc_paths)
            paths[:, t] = paths[:, t-1] * np.exp((mc_mu - 0.5 * mc_sigma**2) * dt + mc_sigma * np.sqrt(dt) * z)

        terminal = paths[:, -1]
        var = np.percentile(terminal, 100 - mc_confidence) - mc_initial
        cvar = terminal[terminal <= np.percentile(terminal, 100 - mc_confidence)].mean() - mc_initial

        # Fan chart
        percentiles = [5, 25, 50, 75, 95]
        pct_values = {p: np.percentile(paths, p, axis=0) for p in percentiles}
        days = list(range(horizon_days + 1))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=days, y=pct_values[5], line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=days, y=pct_values[95], fill="tonexty", fillcolor="rgba(0,212,170,0.1)",
                                 line=dict(width=0), name="5-95%"))
        fig.add_trace(go.Scatter(x=days, y=pct_values[25], line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=days, y=pct_values[75], fill="tonexty", fillcolor="rgba(0,212,170,0.2)",
                                 line=dict(width=0), name="25-75%"))
        fig.add_trace(go.Scatter(x=days, y=pct_values[50], line=dict(color=PROFIT_COLOR, width=2), name="Median"))
        fig.update_layout(**get_plotly_layout(title="Monte Carlo Paths", height=400, xaxis_title="Days", yaxis_title="Portfolio Value"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric(f"VaR ({mc_confidence}%)", f"${var:,.0f}")
        mc2.metric(f"CVaR ({mc_confidence}%)", f"${cvar:,.0f}")
        mc3.metric("Median Terminal", f"${np.median(terminal):,.0f}")
        mc4.metric("Expected Return", f"{(np.mean(terminal)/mc_initial - 1)*100:.1f}%")

# ==================== MODULE 2: Portfolio Optimizer ====================
with tab2:
    st.markdown("##### Portfolio Optimizer")
    default_tickers = ["SPY", "AGG", "GLD", "EFA", "VNQ"]
    po_tickers = st.multiselect("Assets", ["SPY", "AGG", "GLD", "EFA", "VNQ", "TLT", "QQQ", "IWM", "DBC", "TIPS"],
                                default=default_tickers, key="po_tickers")
    po_method = st.selectbox("Method", ["Equal Weight", "Min Variance", "Max Sharpe", "Risk Parity"], key="po_method")
    po_max_w = st.slider("Max Weight", 0.1, 1.0, 0.4, 0.05, key="po_maxw")

    if st.button("Optimize", key="po_run") and len(po_tickers) >= 2:
        import yfinance as yf

        @st.cache_data(ttl=3600)
        def fetch_returns(tickers):
            data = yf.download(tickers, start="2020-01-01", progress=False)["Close"]
            return data.pct_change().dropna()

        with st.spinner("Fetching data..."):
            returns = fetch_returns(po_tickers)

        mu = returns.mean() * 252
        cov = returns.cov() * 252
        n = len(po_tickers)

        if po_method == "Equal Weight":
            weights = np.ones(n) / n
        elif po_method == "Min Variance":
            try:
                import cvxpy as cp
                w = cp.Variable(n)
                prob = cp.Problem(cp.Minimize(cp.quad_form(w, cov.values)),
                                  [cp.sum(w) == 1, w >= 0, w <= po_max_w])
                prob.solve(solver=cp.OSQP)
                weights = w.value
            except Exception:
                weights = np.ones(n) / n
        elif po_method == "Max Sharpe":
            try:
                import cvxpy as cp
                w = cp.Variable(n)
                ret = mu.values @ w
                risk = cp.quad_form(w, cov.values)
                prob = cp.Problem(cp.Maximize(ret - 0.5 * risk),
                                  [cp.sum(w) == 1, w >= 0, w <= po_max_w])
                prob.solve(solver=cp.OSQP)
                weights = w.value
            except Exception:
                weights = np.ones(n) / n
        elif po_method == "Risk Parity":
            weights = 1 / (returns.std().values * np.sqrt(252))
            weights = weights / weights.sum()
            weights = np.minimum(weights, po_max_w)
            weights = weights / weights.sum()

        port_ret = float(mu.values @ weights)
        port_vol = float(np.sqrt(weights @ cov.values @ weights))
        port_sharpe = port_ret / port_vol if port_vol > 0 else 0

        p1, p2, p3 = st.columns(3)
        p1.metric("Return", f"{port_ret:.1%}")
        p2.metric("Volatility", f"{port_vol:.1%}")
        p3.metric("Sharpe", f"{port_sharpe:.2f}")

        fig = go.Figure(go.Bar(x=po_tickers, y=weights * 100,
                               marker_color=[PROFIT_COLOR if w > 0.01 else "#6B7280" for w in weights]))
        fig.update_layout(**get_plotly_layout(title="Optimal Weights (%)", height=350, yaxis_title="Weight %"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ==================== MODULE 3: Backtest Engine ====================
with tab3:
    st.markdown("##### Simple Backtest Engine")
    bt_ticker = st.selectbox("Asset", ["SPY", "QQQ", "AAPL", "MSFT", "AMZN", "GLD"], key="bt_ticker")
    bt_strategy = st.selectbox("Strategy", ["SMA Crossover", "RSI Mean Reversion", "Bollinger Bands"], key="bt_strat")
    bt_fast = st.slider("Fast Period", 5, 50, 20, 5, key="bt_fast")
    bt_slow = st.slider("Slow Period", 20, 200, 50, 10, key="bt_slow")

    if st.button("Run Backtest", key="bt_run"):
        import yfinance as yf

        @st.cache_data(ttl=3600)
        def fetch_price(ticker):
            return yf.download(ticker, start="2020-01-01", progress=False)["Close"]

        with st.spinner("Running backtest..."):
            price = fetch_price(bt_ticker)
            returns = price.pct_change().dropna()

            if bt_strategy == "SMA Crossover":
                fast = price.rolling(bt_fast).mean()
                slow = price.rolling(bt_slow).mean()
                signal = (fast > slow).astype(float)
            elif bt_strategy == "RSI Mean Reversion":
                delta = price.diff()
                gain = delta.where(delta > 0, 0).rolling(bt_fast).mean()
                loss = -delta.where(delta < 0, 0).rolling(bt_fast).mean()
                rsi = 100 - 100 / (1 + gain / (loss + 1e-10))
                signal = pd.Series(0.0, index=price.index)
                signal[rsi < 30] = 1.0
                signal[rsi > 70] = 0.0
                signal = signal.ffill()
            else:
                mean = price.rolling(bt_fast).mean()
                std = price.rolling(bt_fast).std()
                signal = pd.Series(0.0, index=price.index)
                signal[price < mean - 2 * std] = 1.0
                signal[price > mean + 2 * std] = 0.0
                signal = signal.ffill()

            strat_ret = signal.shift(1) * returns
            strat_ret = strat_ret.dropna()
            equity = (1 + strat_ret).cumprod() * 100000
            bench = (1 + returns.loc[strat_ret.index]).cumprod() * 100000

            # Metrics
            sr = strat_ret.mean() / strat_ret.std() * np.sqrt(252) if strat_ret.std() > 0 else 0
            total_ret = equity.iloc[-1] / equity.iloc[0] - 1
            dd = equity / equity.cummax() - 1
            max_dd = dd.min()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sharpe", f"{sr:.2f}")
        m2.metric("Return", f"{total_ret:.1%}")
        m3.metric("Max DD", f"{max_dd:.1%}")
        m4.metric("Trades", f"{int(signal.diff().abs().sum())}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=equity.index, y=equity.values, name="Strategy", line=dict(color=PROFIT_COLOR, width=2)))
        fig.add_trace(go.Scatter(x=bench.index, y=bench.values, name="Buy & Hold", line=dict(color="#6B7280", width=1, dash="dot")))
        fig.update_layout(**get_plotly_layout(title=f"{bt_strategy} on {bt_ticker}", height=400))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ==================== MODULE 4: Vol Surface ====================
with tab4:
    st.markdown("##### Volatility Surface Builder")
    vs_spot = st.number_input("Spot Price", 50.0, 500.0, 100.0, 10.0, key="vs_spot")
    vs_base_vol = st.slider("Base Vol (%)", 5.0, 60.0, 20.0, 1.0, key="vs_bvol") / 100
    vs_skew = st.slider("Skew", -0.5, 0.0, -0.15, 0.05, key="vs_skew")

    if st.button("Build Surface", key="vs_run"):
        strikes = np.linspace(0.7 * vs_spot, 1.3 * vs_spot, 25)
        maturities = np.array([0.083, 0.25, 0.5, 1.0, 2.0])
        log_m = np.log(strikes / vs_spot)

        Z = np.zeros((len(maturities), len(strikes)))
        for i, T in enumerate(maturities):
            Z[i] = vs_base_vol + vs_skew * log_m / np.sqrt(T) + 0.4 * log_m ** 2

        fig = go.Figure(data=[go.Surface(
            z=Z * 100, x=strikes, y=maturities,
            colorscale=[[0, "#1E90FF"], [0.5, "#00D4AA"], [1, "#FF6B35"]],
            contours_z=dict(show=True, usecolormap=True, project_z=True),
        )])
        fig.update_layout(
            title="Implied Volatility Surface (%)", height=500,
            scene=dict(
                xaxis_title="Strike", yaxis_title="Maturity (yr)", zaxis_title="IV (%)",
                bgcolor="#1A1F2E",
                xaxis=dict(gridcolor="#2D3748"), yaxis=dict(gridcolor="#2D3748"), zaxis=dict(gridcolor="#2D3748"),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="JetBrains Mono", color="#E0E0E0"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ==================== MODULE 5: Risk Calculator ====================
with tab5:
    st.markdown("##### Risk Calculator")
    rc_tickers = st.multiselect("Portfolio Assets", ["SPY", "AGG", "GLD", "EFA", "VNQ", "QQQ", "TLT"],
                                default=["SPY", "AGG", "GLD"], key="rc_tickers")
    rc_conf = st.slider("Confidence Level (%)", 90, 99, 95, 1, key="rc_conf")
    rc_horizon = st.selectbox("Horizon", ["1D", "5D", "10D", "21D"], index=0, key="rc_horizon")

    if st.button("Calculate Risk", key="rc_run") and len(rc_tickers) >= 1:
        import yfinance as yf

        @st.cache_data(ttl=3600)
        def fetch_returns_rc(tickers):
            data = yf.download(tickers, start="2020-01-01", progress=False)["Close"]
            return data.pct_change().dropna()

        with st.spinner("Computing risk metrics..."):
            returns = fetch_returns_rc(rc_tickers)
            n = len(rc_tickers)
            weights = np.ones(n) / n
            port_returns = (returns * weights).sum(axis=1)

            horizon_map = {"1D": 1, "5D": 5, "10D": 10, "21D": 21}
            h = horizon_map[rc_horizon]
            if h > 1:
                port_returns_h = port_returns.rolling(h).sum().dropna()
            else:
                port_returns_h = port_returns

            # Historical VaR
            hist_var = np.percentile(port_returns_h, 100 - rc_conf)
            hist_cvar = port_returns_h[port_returns_h <= hist_var].mean()

            # Parametric VaR
            from scipy import stats
            mu_p = port_returns_h.mean()
            sig_p = port_returns_h.std()
            param_var = mu_p + sig_p * stats.norm.ppf((100 - rc_conf) / 100)

            # Monte Carlo VaR
            np.random.seed(42)
            mc_returns = np.random.normal(mu_p, sig_p, 10000)
            mc_var = np.percentile(mc_returns, 100 - rc_conf)

        r1, r2, r3 = st.columns(3)
        r1.metric("Historical VaR", f"{hist_var:.2%}")
        r2.metric("Parametric VaR", f"{param_var:.2%}")
        r3.metric("Monte Carlo VaR", f"{mc_var:.2%}")

        r4, r5, r6 = st.columns(3)
        r4.metric("CVaR (ES)", f"{hist_cvar:.2%}")
        r5.metric("Portfolio Vol (ann.)", f"{sig_p * np.sqrt(252/h):.2%}")
        r6.metric("Horizon", rc_horizon)

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=port_returns_h, nbinsx=80, name="Returns",
                                   marker_color="rgba(0,212,170,0.4)", marker_line_color=PROFIT_COLOR))
        fig.add_vline(x=hist_var, line_dash="dash", line_color=LOSS_COLOR, annotation_text=f"VaR {rc_conf}%")
        fig.add_vline(x=hist_cvar, line_dash="dot", line_color="#FFA502", annotation_text="CVaR")
        fig.update_layout(**get_plotly_layout(
            title=f"Return Distribution ({rc_horizon} horizon)", height=350,
            xaxis_title="Return", yaxis_title="Frequency",
        ))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
