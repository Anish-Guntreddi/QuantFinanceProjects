"""Strategy Detail — Deep-dive view with interactive simulation."""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.theme import load_css, CATEGORY_COLORS, CATEGORY_DISPLAY
from components.charts import equity_curve_chart, drawdown_chart, monthly_heatmap, sensitivity_chart
from components.metrics import render_metrics_row
from components.parameter_controls import render_controls
from utils.data_loader import load_all_cards, load_results
from utils.formatting import fmt_pct, fmt_number

load_css()

st.markdown("#### 📊 Strategy Detail")

cards = load_all_cards()
if not cards:
    st.warning("No strategy data found. Run generate_precomputed.py first.")
    st.stop()

# Strategy selector
card_names = {c["project_id"]: c["title"] for c in cards}
selected_id = st.selectbox("Select Strategy", list(card_names.keys()),
                           format_func=lambda x: card_names[x],
                           key="strategy_select")

card = next((c for c in cards if c["project_id"] == selected_id), None)
if not card:
    st.stop()

results = load_results(selected_id)

# Header
cat = card.get("category", "")
color = CATEGORY_COLORS.get(cat, "#00D4AA")
cat_display = CATEGORY_DISPLAY.get(cat, cat)
langs = " ".join(f'`{l}`' for l in card.get("languages", []))

st.markdown(f"""
<div style="border-left: 4px solid {color}; padding-left: 16px; margin: 16px 0;">
    <h2 style="margin: 0;">{card['title']}</h2>
    <div style="color: #9CA3AF; margin-top: 4px;">{cat_display} · {card.get('frequency', 'Daily')} · {langs}</div>
</div>
""", unsafe_allow_html=True)

st.markdown(card.get("long_description", ""))

if results:
    metrics = results.get("metrics", {})

    # Metrics row
    render_metrics_row(metrics, columns=4)
    st.markdown("")

    # Two columns: charts + parameters
    col_left, col_right = st.columns([1.3, 1])

    with col_left:
        eq = results.get("equity_curve", {})
        if eq:
            fig = equity_curve_chart(eq["dates"], eq["values"], eq.get("benchmark_values"), title="Equity Curve")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            fig_dd = drawdown_chart(eq["dates"], eq["values"])
            st.plotly_chart(fig_dd, use_container_width=True, config={"displayModeBar": False})

        monthly = results.get("monthly_returns", {})
        if monthly:
            fig_heat = monthly_heatmap(monthly)
            st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})

    with col_right:
        st.markdown("##### Parameters")
        params = render_controls(card.get("interactive_params", []))

        tier = card.get("simulation_tier", "precomputed")
        if tier == "precomputed":
            st.info("📋 Showing pre-computed results. C++/heavy compute components require full environment.")
        elif tier == "cached_sweep":
            st.info("📋 Results interpolated from pre-computed parameter grid.")

        # Sensitivity
        sens = results.get("parameter_sensitivity", [])
        if sens:
            st.markdown("##### Parameter Sensitivity")
            fig_sens = sensitivity_chart(sens)
            st.plotly_chart(fig_sens, use_container_width=True, config={"displayModeBar": False})

    # Tags
    tags = card.get("tags", [])
    if tags:
        tag_html = " ".join(f'<span class="tag-pill">{t}</span>' for t in tags)
        st.markdown(f"**Key Techniques:** {tag_html}", unsafe_allow_html=True)
else:
    st.warning("No results data available for this strategy.")
