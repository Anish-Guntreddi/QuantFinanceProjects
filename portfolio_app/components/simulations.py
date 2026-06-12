"""
Interactive quant finance simulations — one function per project demo type.
All seeded via np.random.default_rng(seed). Pure numpy + plotly.
Dispatched by model_card['demo_type'] string.
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from components.model_loader import PLOTLY_COLORS

# ── Shared layout helper ───────────────────────────────────────────────────


def _layout(title="", height=380):
    return go.Layout(
        title=dict(text=title, font=dict(family="JetBrains Mono", size=13, color="#fafaf9")),
        font=dict(family="Source Sans 3", size=12, color="#a8a29e"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=50, r=20, t=40, b=40),
        showlegend=True,
        legend=dict(font=dict(family="JetBrains Mono", size=10)),
    )


# ── sim_qbacktest ──────────────────────────────────────────────────────────

def sim_qbacktest():
    """
    GBM price path + MA crossover signals.
    KEY FEATURE: side-by-side equity curves:
      - "T+1 open fills (honest)" — signal generated on bar t, fill on bar t+1
      - "Same-bar close fills (look-ahead)" — fill on the same bar that generated the signal
    Shows the Sharpe gap to demonstrate why T+1 discipline matters.
    """
    st.markdown("#### MA Crossover: T+1 Honest vs. Same-Bar Look-Ahead")
    st.caption(
        "Demonstrates the look-ahead bias that inflates backtest Sharpe when fills are "
        "applied to the same bar that generated the signal instead of the next open."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        seed = st.slider("Seed", 0, 99, 42, key="bt_seed")
    with col2:
        cost_bps = st.slider("Cost (bps)", 0, 50, 5, key="bt_cost")
    with col3:
        fast = st.slider("Fast MA window", 5, 50, 10, key="bt_fast")
    with col4:
        slow = st.slider("Slow MA window", 20, 200, 50, key="bt_slow")

    if fast >= slow:
        st.warning("Fast window must be smaller than slow window.")
        return

    rng = np.random.default_rng(seed)
    n = 500
    dt = 1 / 252
    mu, sigma = 0.08, 0.18
    log_rets = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n)
    price = 100 * np.exp(np.cumsum(log_rets))

    # Moving averages
    ma_fast = np.convolve(price, np.ones(fast) / fast, mode='full')[:n]
    ma_slow = np.convolve(price, np.ones(slow) / slow, mode='full')[:n]

    # Signal: +1 when fast > slow, else -1
    raw_signal = np.where(ma_fast > ma_slow, 1.0, -1.0)
    # Zero out before slow window warms up
    raw_signal[:slow - 1] = 0.0

    cost = cost_bps * 1e-4

    def _equity(signal, shift):
        """Build equity curve given a signal shift (1=T+1, 0=same-bar)."""
        pos = np.roll(signal, shift)
        pos[:shift] = 0.0
        daily_ret = pos * log_rets
        # Subtract cost on position changes
        trade = np.diff(pos, prepend=pos[0])
        daily_ret -= np.abs(trade) * cost
        cum = np.cumprod(1 + daily_ret)
        return cum

    eq_honest = _equity(raw_signal, 1)   # T+1
    eq_lookahead = _equity(raw_signal, 0)  # same-bar

    def _sharpe(signal, shift):
        pos = np.roll(signal, shift)
        pos[:shift] = 0.0
        daily_ret = pos * log_rets
        trade = np.diff(pos, prepend=pos[0])
        daily_ret -= np.abs(trade) * cost
        if daily_ret.std() == 0:
            return 0.0
        return float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))

    sharpe_honest = _sharpe(raw_signal, 1)
    sharpe_la = _sharpe(raw_signal, 0)
    sharpe_gap = sharpe_la - sharpe_honest

    # Metrics row
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Sharpe — T+1 Honest", f"{sharpe_honest:.2f}")
    with m2:
        st.metric("Sharpe — Same-Bar", f"{sharpe_la:.2f}")
    with m3:
        delta_str = f"+{sharpe_gap:.2f}" if sharpe_gap >= 0 else f"{sharpe_gap:.2f}"
        st.metric("Sharpe Gap (Look-Ahead Bias)", delta_str)

    if sharpe_gap > 0.2:
        st.error(
            f"Significant look-ahead bias detected: same-bar fills inflate Sharpe by {sharpe_gap:.2f}. "
            "Always apply signals with a one-bar lag."
        )
    else:
        st.info("Look-ahead gap is modest for this configuration — but always use T+1 fills in production.")

    # Equity curve chart
    fig = go.Figure(layout=_layout("Equity Curves: T+1 vs. Same-Bar Fill", height=380))
    x = list(range(n))
    fig.add_trace(go.Scatter(
        x=x, y=eq_honest.tolist(), name="T+1 open fills (honest)",
        line=dict(color=PLOTLY_COLORS[0], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=eq_lookahead.tolist(), name="Same-bar close fills (look-ahead)",
        line=dict(color=PLOTLY_COLORS[4], width=2, dash='dash'),
    ))
    fig.update_xaxes(title_text="Bar", gridcolor="#292524")
    fig.update_yaxes(title_text="Cumulative Return (× initial)", gridcolor="#292524")
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Price + MA overlay
    fig2 = go.Figure(layout=_layout("Price & Moving Averages", height=280))
    fig2.add_trace(go.Scatter(x=x, y=price.tolist(), name="Price",
                               line=dict(color="#a8a29e", width=1)))
    fig2.add_trace(go.Scatter(x=x, y=ma_fast.tolist(), name=f"MA({fast})",
                               line=dict(color=PLOTLY_COLORS[0], width=1.5)))
    fig2.add_trace(go.Scatter(x=x, y=ma_slow.tolist(), name=f"MA({slow})",
                               line=dict(color=PLOTLY_COLORS[1], width=1.5, dash='dash')))
    fig2.update_xaxes(gridcolor="#292524")
    fig2.update_yaxes(gridcolor="#292524")
    st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})


# ── sim_alpharank ──────────────────────────────────────────────────────────

def sim_alpharank():
    """
    Planted-IC cross-section simulation.
    Shows: scatter of signal vs forward return, decile mean-return bar chart,
    realized IC metric.
    """
    st.markdown("#### Cross-Sectional Alpha: IC & Decile Returns")
    st.caption(
        "Simulates a cross-section with planted information coefficient (IC). "
        "Higher IC → cleaner rank ordering of forward returns."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        seed = st.slider("Seed", 0, 99, 7, key="cs_seed")
    with c2:
        ic_strength = st.slider("IC strength", 0.0, 0.30, 0.05, step=0.01, key="cs_ic")
    with c3:
        n_assets = st.slider("Assets", 50, 500, 200, step=50, key="cs_n")

    rng = np.random.default_rng(seed)
    n_periods = 60

    # Simulate: forward_ret = ic_strength * signal + noise
    signals = rng.standard_normal((n_periods, n_assets))
    noise = rng.standard_normal((n_periods, n_assets)) * np.sqrt(1 - ic_strength**2)
    fwd_rets = ic_strength * signals + noise

    # Realized IC (Pearson correlation averaged across periods)
    ics = []
    for t in range(n_periods):
        s = signals[t]
        r = fwd_rets[t]
        if s.std() > 0 and r.std() > 0:
            ics.append(float(np.corrcoef(s, r)[0, 1]))
    realized_ic = float(np.mean(ics)) if ics else 0.0
    icir = float(np.mean(ics) / np.std(ics)) if ics and np.std(ics) > 0 else 0.0

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Realized IC (mean)", f"{realized_ic:.3f}")
    with m2:
        st.metric("ICIR", f"{icir:.2f}")

    # Scatter: last period signal vs forward return
    last_s = signals[-1]
    last_r = fwd_rets[-1]
    fig_scatter = go.Figure(layout=_layout("Signal vs. Forward Return (last period)", height=340))
    fig_scatter.add_trace(go.Scatter(
        x=last_s.tolist(), y=last_r.tolist(), mode='markers',
        marker=dict(color=PLOTLY_COLORS[0], size=5, opacity=0.6),
        name="Assets",
    ))
    # Trend line
    m_coef = np.polyfit(last_s, last_r, 1)
    x_line = np.linspace(last_s.min(), last_s.max(), 50)
    fig_scatter.add_trace(go.Scatter(
        x=x_line.tolist(), y=(m_coef[0] * x_line + m_coef[1]).tolist(),
        mode='lines', name="OLS fit",
        line=dict(color=PLOTLY_COLORS[1], width=2),
    ))
    fig_scatter.update_xaxes(title_text="Signal", gridcolor="#292524")
    fig_scatter.update_yaxes(title_text="Forward Return", gridcolor="#292524")
    st.plotly_chart(fig_scatter, use_container_width=True, config={'displayModeBar': False})

    # Decile mean returns (averaged across all periods)
    decile_means = []
    for t in range(n_periods):
        ranks = np.argsort(np.argsort(signals[t]))  # rank 0..n-1
        deciles = (ranks * 10 // n_assets).clip(0, 9)
        for d in range(10):
            mask = deciles == d
            if mask.sum() > 0:
                decile_means.append((d, float(fwd_rets[t][mask].mean())))

    d_avg = np.zeros(10)
    d_cnt = np.zeros(10)
    for d, v in decile_means:
        d_avg[d] += v
        d_cnt[d] += 1
    d_avg = np.where(d_cnt > 0, d_avg / d_cnt, 0.0)

    colors_d = [PLOTLY_COLORS[0] if d_avg[i] >= 0 else PLOTLY_COLORS[4] for i in range(10)]
    fig_dec = go.Figure(layout=_layout("Mean Forward Return by Signal Decile", height=300))
    fig_dec.add_trace(go.Bar(
        x=[f"D{i+1}" for i in range(10)],
        y=d_avg.tolist(),
        marker_color=colors_d,
        text=[f"{v:.4f}" for v in d_avg],
        textposition='outside',
        textfont=dict(family="JetBrains Mono", size=10),
        showlegend=False,
    ))
    fig_dec.update_xaxes(title_text="Signal Decile (D1=low, D10=high)", gridcolor="#292524")
    fig_dec.update_yaxes(title_text="Mean Fwd Return", gridcolor="#292524")
    st.plotly_chart(fig_dec, use_container_width=True, config={'displayModeBar': False})


# ── sim_macroregime ────────────────────────────────────────────────────────

def sim_macroregime():
    """
    3-state sticky Markov regime simulation.
    Regime-shaded price path + strategy vs buy-and-hold equity.
    T+1 note: weights applied with one-bar lag.
    """
    st.markdown("#### 3-State Macro Regime Simulation")
    st.caption(
        "Sticky Markov chain drives three regimes (Bull / Bear / Crisis). "
        "Equity weights are applied with a **one-bar lag** (T+1 discipline — no look-ahead)."
    )

    c1, c2 = st.columns(2)
    with c1:
        seed = st.slider("Seed", 0, 99, 13, key="mr_seed")
        stay_prob = st.slider("Regime stay probability", 0.95, 0.999, 0.97,
                               step=0.001, format="%.3f", key="mr_stay")
    with c2:
        w_bull = st.slider("Bull equity weight", 0.0, 1.5, 1.0, step=0.05, key="mr_wbull")
        w_bear = st.slider("Bear equity weight", 0.0, 1.5, 0.4, step=0.05, key="mr_wbear")
        w_crisis = st.slider("Crisis equity weight", 0.0, 1.5, 0.0, step=0.05, key="mr_wcrisis")

    rng = np.random.default_rng(seed)
    n = 600

    # Regime parameters: (mu_daily, sigma_daily)
    REGIME_PARAMS = {
        0: (0.0008, 0.008),   # Bull
        1: (-0.0003, 0.015),  # Bear
        2: (-0.003, 0.035),   # Crisis
    }
    REGIME_NAMES = {0: "Bull", 1: "Bear", 2: "Crisis"}
    REGIME_COLORS = {0: PLOTLY_COLORS[2], 1: PLOTLY_COLORS[1], 2: PLOTLY_COLORS[4]}
    WEIGHTS = {0: w_bull, 1: w_bear, 2: w_crisis}

    # Transition matrix: stay with stay_prob, equal split of rest
    trans_off = (1 - stay_prob) / 2
    P = np.array([
        [stay_prob, trans_off, trans_off],
        [trans_off, stay_prob, trans_off],
        [trans_off, trans_off, stay_prob],
    ])

    # Simulate regime sequence
    regimes = np.zeros(n, dtype=int)
    regimes[0] = 0
    for t in range(1, n):
        regimes[t] = rng.choice(3, p=P[regimes[t - 1]])

    # Simulate daily returns
    rets = np.zeros(n)
    for t in range(n):
        mu, sig = REGIME_PARAMS[regimes[t]]
        rets[t] = rng.normal(mu, sig)

    # Equity weights with T+1 lag
    weights_raw = np.array([WEIGHTS[r] for r in regimes], dtype=float)
    weights_lagged = np.roll(weights_raw, 1)
    weights_lagged[0] = 0.0

    strat_rets = weights_lagged * rets
    bnh_rets = rets

    eq_strat = np.cumprod(1 + strat_rets)
    eq_bnh = np.cumprod(1 + bnh_rets)

    def _sharpe(r):
        if r.std() == 0:
            return 0.0
        return float(r.mean() / r.std() * np.sqrt(252))

    def _max_dd(eq):
        roll_max = np.maximum.accumulate(eq)
        dd = (eq - roll_max) / roll_max
        return float(dd.min())

    # Dwell time stats
    dwell = {r: 0 for r in range(3)}
    for r in regimes:
        dwell[r] += 1

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Strategy Sharpe", f"{_sharpe(strat_rets):.2f}")
        st.metric("B&H Sharpe", f"{_sharpe(bnh_rets):.2f}")
    with m2:
        st.metric("Max Drawdown (strategy)", f"{_max_dd(eq_strat)*100:.1f}%")
        st.metric("Max Drawdown (B&H)", f"{_max_dd(eq_bnh)*100:.1f}%")
    with m3:
        for r in range(3):
            st.metric(f"Bars in {REGIME_NAMES[r]}", f"{dwell[r]:,} ({dwell[r]/n*100:.0f}%)")

    # Price path with regime shading
    price = 100 * eq_bnh
    fig = go.Figure(layout=_layout("Price Path — Regime-Shaded", height=360))

    # Regime shading via shapes
    shapes = []
    i = 0
    while i < n:
        r = regimes[i]
        j = i
        while j < n and regimes[j] == r:
            j += 1
        shapes.append(dict(
            type="rect", xref="x", yref="paper",
            x0=i, x1=j - 1, y0=0, y1=1,
            fillcolor=REGIME_COLORS[r], opacity=0.12, line_width=0,
            layer="below",
        ))
        i = j

    fig.update_layout(shapes=shapes)

    x = list(range(n))
    fig.add_trace(go.Scatter(
        x=x, y=price.tolist(), name="Price (B&H)",
        line=dict(color="#a8a29e", width=1.5), showlegend=True,
    ))

    # Invisible traces for legend colors
    for r in range(3):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(color=REGIME_COLORS[r], size=10, symbol='square'),
            name=REGIME_NAMES[r],
        ))

    fig.update_xaxes(title_text="Bar", gridcolor="#292524")
    fig.update_yaxes(title_text="Price", gridcolor="#292524")
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Equity comparison
    fig2 = go.Figure(layout=_layout("Strategy vs. Buy-and-Hold Equity", height=300))
    fig2.add_trace(go.Scatter(
        x=x, y=eq_strat.tolist(), name="Regime Strategy (T+1)",
        line=dict(color=PLOTLY_COLORS[0], width=2),
    ))
    fig2.add_trace(go.Scatter(
        x=x, y=eq_bnh.tolist(), name="Buy & Hold",
        line=dict(color=PLOTLY_COLORS[1], width=2, dash='dash'),
    ))
    fig2.update_xaxes(gridcolor="#292524")
    fig2.update_yaxes(title_text="Cumulative Return", gridcolor="#292524")
    st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})


# ── sim_volsurfacelab ──────────────────────────────────────────────────────

def sim_volsurfacelab():
    """
    SVI smile explorer.
    w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))
    Shows IV smile + butterfly density g(k); detects butterfly arbitrage.
    """
    st.markdown("#### SVI Volatility Surface Lab")
    st.caption(
        "Explore the Raw SVI parameterization of the total-variance smile. "
        "The butterfly condition g(k) ≥ 0 is verified numerically."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        a = st.slider("a (level)", -0.1, 0.5, 0.04, step=0.01, key="svi_a")
        b = st.slider("b (slope)", 0.01, 1.0, 0.4, step=0.01, key="svi_b")
    with c2:
        rho = st.slider("ρ (skew)", -0.99, 0.99, -0.3, step=0.01, key="svi_rho")
        m = st.slider("m (shift)", -1.0, 1.0, 0.0, step=0.05, key="svi_m")
    with c3:
        sig = st.slider("σ (curvature)", 0.01, 1.0, 0.3, step=0.01, key="svi_sig")
        T = 0.25  # fixed maturity

    k = np.linspace(-1.5, 1.5, 300)

    def w(k_arr):
        return a + b * (rho * (k_arr - m) + np.sqrt((k_arr - m)**2 + sig**2))

    w_vals = w(k)
    # Clamp negative total variance (arbitrage)
    w_vals_safe = np.maximum(w_vals, 1e-8)
    iv = np.sqrt(w_vals_safe / T)

    # Butterfly density: g(k) = (1 - k*w'/(2w))^2 - (w'^2/4)*(1/w + 1/4) + w''/2
    dk = k[1] - k[0]
    w_prime = np.gradient(w_vals, dk)
    w_double = np.gradient(w_prime, dk)

    g = (
        (1 - k * w_prime / (2 * w_vals_safe))**2
        - (w_prime**2 / 4) * (1 / w_vals_safe + 0.25)
        + w_double / 2
    )

    min_g = float(g.min())
    violated = min_g < 0

    if violated:
        bad_k = k[g < 0]
        st.error(
            f"Butterfly arbitrage violated: g(k) < 0 for k ∈ [{bad_k.min():.3f}, {bad_k.max():.3f}]. "
            "Adjust parameters to restore a valid smile."
        )
    else:
        st.success(f"No butterfly arbitrage — min g(k) = {min_g:.4f} ≥ 0.")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("ATM IV (k=0)", f"{float(np.interp(0, k, iv))*100:.2f}%")
    with m2:
        st.metric("Min g(k)", f"{min_g:.4f}")
    with m3:
        st.metric("w(0) — total var", f"{float(np.interp(0, k, w_vals)):.4f}")

    # IV smile panel
    fig = go.Figure(layout=_layout(f"SVI Implied Volatility Smile (T={T:.2f})", height=360))
    fig.add_trace(go.Scatter(
        x=k.tolist(), y=(iv * 100).tolist(), name="IV (%)",
        line=dict(color=PLOTLY_COLORS[0], width=2),
    ))
    fig.update_xaxes(title_text="Log-Moneyness k = log(K/F)", gridcolor="#292524")
    fig.update_yaxes(title_text="Implied Volatility (%)", gridcolor="#292524")
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Butterfly density panel
    fig2 = go.Figure(layout=_layout("Butterfly Density g(k)", height=280))
    fill_color = "rgba(251,113,133,0.15)" if violated else "rgba(52,211,153,0.1)"
    line_color = PLOTLY_COLORS[4] if violated else PLOTLY_COLORS[2]
    fig2.add_trace(go.Scatter(
        x=k.tolist(), y=g.tolist(), name="g(k)",
        line=dict(color=line_color, width=2),
        fill='tozeroy', fillcolor=fill_color,
    ))
    # Zero line
    fig2.add_hline(y=0, line=dict(color="#78716c", width=1, dash="dash"))
    fig2.update_xaxes(title_text="k", gridcolor="#292524")
    fig2.update_yaxes(title_text="g(k)", gridcolor="#292524")
    st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})


# ── sim_defiregimenet ──────────────────────────────────────────────────────

def sim_defiregimenet():
    """
    Multi-token fat-tail simulation.
    Student-t returns with market factor + idiosyncratic components.
    Shows price paths + cross-token correlation heatmap + kurtosis metrics.
    """
    st.markdown("#### Multi-Token Fat-Tail Simulation")
    st.caption(
        "Simulates correlated crypto/DeFi token returns via a Student-t market factor model. "
        "Kurtosis and correlation structure adapt to the degree-of-freedom and factor-weight sliders."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        seed = st.slider("Seed", 0, 99, 22, key="cr_seed")
    with c2:
        df = st.slider("Student-t df (tail heaviness)", 2.5, 30.0, 4.0,
                        step=0.5, key="cr_df")
    with c3:
        mkt_weight = st.slider("Market factor weight", 0.0, 1.0, 0.6,
                                step=0.05, key="cr_mkt")

    n_tokens = st.slider("Tokens", 2, 5, 3, key="cr_ntokens")

    rng = np.random.default_rng(seed)
    n = 400

    TOKEN_NAMES = ["BTC", "ETH", "SOL", "AVAX", "ARB"][:n_tokens]

    def _t_sample(size):
        """Draw Student-t samples via ratio of normals and chi-squared."""
        z = rng.standard_normal(size)
        v = rng.chisquare(df, size) / df
        return z / np.sqrt(v)

    # Market factor
    mkt = _t_sample(n) * 0.04

    prices = {}
    all_rets = np.zeros((n, n_tokens))
    for i, name in enumerate(TOKEN_NAMES):
        idio = _t_sample(n) * 0.05
        ret = mkt_weight * mkt + (1 - mkt_weight) * idio
        all_rets[:, i] = ret
        prices[name] = 100 * np.cumprod(1 + ret)

    # Kurtosis per token
    def _kurt(r):
        mu = r.mean()
        sigma = r.std()
        if sigma == 0:
            return 0.0
        return float(np.mean(((r - mu) / sigma) ** 4))

    kurt_vals = {TOKEN_NAMES[i]: _kurt(all_rets[:, i]) for i in range(n_tokens)}

    # Metrics
    cols_m = st.columns(n_tokens)
    for i, name in enumerate(TOKEN_NAMES):
        with cols_m[i]:
            st.metric(f"{name} Kurtosis", f"{kurt_vals[name]:.2f}")

    # Price paths
    fig = go.Figure(layout=_layout("Token Price Paths", height=340))
    x = list(range(n))
    for i, name in enumerate(TOKEN_NAMES):
        fig.add_trace(go.Scatter(
            x=x, y=prices[name].tolist(), name=name,
            line=dict(color=PLOTLY_COLORS[i % len(PLOTLY_COLORS)], width=2),
        ))
    fig.update_xaxes(title_text="Bar", gridcolor="#292524")
    fig.update_yaxes(title_text="Price", gridcolor="#292524")
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Correlation heatmap
    corr = np.corrcoef(all_rets.T)

    import plotly.express as px
    fig_corr = px.imshow(
        corr,
        x=TOKEN_NAMES, y=TOKEN_NAMES,
        color_continuous_scale=[[0, '#042f2e'], [0.5, '#0d9488'], [1, '#2dd4bf']],
        zmin=-1, zmax=1,
        text_auto=".2f",
    )
    fig_corr.update_layout(
        title=dict(text="Return Correlation Matrix", font=dict(family="JetBrains Mono", size=13, color="#fafaf9")),
        font=dict(family="JetBrains Mono", size=11, color="#fafaf9"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(l=50, r=20, t=40, b=40),
        coloraxis_colorbar=dict(
            tickfont=dict(family="JetBrains Mono", color="#a8a29e"),
            title=dict(text="ρ", font=dict(family="JetBrains Mono", color="#a8a29e")),
        ),
    )
    fig_corr.update_traces(textfont=dict(family="JetBrains Mono", size=11, color="#fafaf9"))
    st.plotly_chart(fig_corr, use_container_width=True, config={'displayModeBar': False})


# ── Dispatcher ────────────────────────────────────────────────────────────

DEMO_MAP = {
    "sim_backtest": sim_qbacktest,
    "sim_crosssection": sim_alpharank,
    "sim_regime": sim_macroregime,
    "sim_svi": sim_volsurfacelab,
    "sim_crypto": sim_defiregimenet,
}


def render_simulation(demo_type):
    """Dispatch to the correct simulation function by demo_type string."""
    fn = DEMO_MAP.get(demo_type)
    if fn is None:
        st.info(f"No simulation registered for demo_type='{demo_type}'.")
        return
    try:
        fn()
    except Exception as exc:
        st.info(f"Simulation unavailable: {exc}")
