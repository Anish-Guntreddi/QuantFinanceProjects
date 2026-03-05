"""Compare — Cross-strategy comparison."""

import streamlit as st
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.theme import load_css, CATEGORY_COLORS, get_plotly_layout
from components.charts import radar_chart, equity_curve_chart
from utils.data_loader import load_all_cards, load_results
from utils.formatting import fmt_pct, fmt_number
import plotly.graph_objects as go

load_css()

st.markdown("#### 📈 Strategy Comparison")

cards = load_all_cards()
if not cards:
    st.warning("No data found.")
    st.stop()

card_map = {c["project_id"]: c for c in cards}
selected = st.multiselect(
    "Select strategies to compare (2-6)",
    list(card_map.keys()),
    default=list(card_map.keys())[:3],
    format_func=lambda x: card_map[x]["title"],
    max_selections=6,
)

if len(selected) < 2:
    st.info("Select at least 2 strategies to compare.")
    st.stop()

# Load results
results_map = {}
for pid in selected:
    r = load_results(pid)
    if r:
        results_map[pid] = r

if len(results_map) < 2:
    st.warning("Insufficient results data.")
    st.stop()

# Radar chart
st.markdown("##### Performance Radar")
labels = ["Sharpe", "Sortino", "Win Rate", "Calmar", "Return"]
values_dict = {}
for pid in selected:
    r = results_map.get(pid)
    if not r:
        continue
    m = r["metrics"]
    # Normalize to 0-1 scale
    vals = [
        min(max(m.get("sharpe_ratio", 0) / 3, 0), 1),
        min(max(m.get("sortino_ratio", 0) / 4, 0), 1),
        m.get("win_rate", 0.5),
        min(max(m.get("calmar_ratio", 0) / 5, 0), 1),
        min(max(m.get("total_return", 0) / 0.5, 0), 1),
    ]
    values_dict[card_map[pid]["title"][:25]] = vals

fig_radar = radar_chart(labels, values_dict)
st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})

# Equity curves overlay
st.markdown("##### Equity Curves (Rebased to 100)")
colors = ["#00D4AA", "#FF6B35", "#7B68EE", "#1E90FF", "#FFA502", "#FF4757"]
fig_eq = go.Figure()
for i, pid in enumerate(selected):
    r = results_map.get(pid)
    if not r or "equity_curve" not in r:
        continue
    eq = r["equity_curve"]
    vals = np.array(eq["values"])
    rebased = vals / vals[0] * 100
    fig_eq.add_trace(go.Scatter(
        x=eq["dates"], y=rebased,
        name=card_map[pid]["title"][:30],
        line=dict(color=colors[i % len(colors)], width=2),
    ))
fig_eq.update_layout(**get_plotly_layout(
    title="Equity Curves (Rebased)", height=450,
    yaxis_title="Value (base=100)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
))
st.plotly_chart(fig_eq, use_container_width=True, config={"displayModeBar": False})

# Comparison table
st.markdown("##### Metrics Comparison")
metric_keys = ["sharpe_ratio", "sortino_ratio", "max_drawdown", "win_rate", "cagr", "total_return",
               "calmar_ratio", "profit_factor", "annualized_vol", "total_trades"]

table_html = '<table style="width:100%; border-collapse: collapse; font-family: var(--font-mono);">'
table_html += '<tr style="border-bottom: 1px solid #2D3748;">'
table_html += '<th style="text-align: left; padding: 8px; color: #9CA3AF;">Metric</th>'
for pid in selected:
    name = card_map[pid]["title"][:20]
    table_html += f'<th style="text-align: right; padding: 8px; color: #E0E0E0;">{name}</th>'
table_html += '</tr>'

for mk in metric_keys:
    table_html += '<tr style="border-bottom: 1px solid #1A1F2E;">'
    label = mk.replace("_", " ").title()
    table_html += f'<td style="padding: 6px 8px; color: #9CA3AF; font-size: 0.85rem;">{label}</td>'
    vals = []
    for pid in selected:
        r = results_map.get(pid)
        v = r["metrics"].get(mk) if r else None
        vals.append(v)

    best = max((v for v in vals if v is not None), default=None)
    if "drawdown" in mk:
        best = max((v for v in vals if v is not None), default=None)

    for v in vals:
        if v is None:
            table_html += '<td style="text-align: right; padding: 6px 8px; color: #6B7280;">—</td>'
        else:
            if "return" in mk or "drawdown" in mk or "vol" in mk or "rate" in mk:
                display = fmt_pct(v)
            else:
                display = fmt_number(v)
            color = "#E0E0E0"
            if v == best and len(vals) > 1:
                color = "#00D4AA"
            table_html += f'<td style="text-align: right; padding: 6px 8px; color: {color}; font-size: 0.9rem;">{display}</td>'
    table_html += '</tr>'

table_html += '</table>'
st.markdown(table_html, unsafe_allow_html=True)
