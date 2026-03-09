"""
Compare — Side-by-side comparison of 2-4 strategies.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Compare Strategies", page_icon="📈", layout="wide")

# CSS
_ASSETS = Path(__file__).parent.parent / "assets"
css_file = _ASSETS / "style.css"
if css_file.exists():
    st.markdown(f"<style>{css_file.read_text()}</style>", unsafe_allow_html=True)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import get_project_ids, load_card, load_results
from components.theme import plotly_layout, chart_colors, ACCENT, TEXT_MUTED, BORDER
from components.metrics_panel import render_metrics_comparison

# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
project_ids = get_project_ids()

selected = st.multiselect(
    "Select 2-4 strategies to compare",
    project_ids,
    default=project_ids[:2],
    max_selections=4,
    format_func=lambda x: x.replace("_", " ").title(),
)

if len(selected) < 2:
    st.info("Select at least 2 strategies to compare.")
    st.stop()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
cards = {pid: load_card(pid) for pid in selected}
results = {pid: load_results(pid) for pid in selected}
colors = chart_colors(len(selected))
labels = [cards[pid].get("title", pid) for pid in selected]

# ---------------------------------------------------------------------------
# Overlaid equity curves
# ---------------------------------------------------------------------------
st.markdown("### Equity Curves")

fig = go.Figure()
for i, pid in enumerate(selected):
    ec = results[pid].get("equity_curve", {})
    dates = ec.get("dates", [])
    values = ec.get("values", [])
    if dates and values:
        fig.add_trace(go.Scatter(
            x=dates, y=values,
            name=cards[pid].get("title", pid)[:30],
            line=dict(color=colors[i], width=2),
        ))

layout = plotly_layout(title="Normalized Equity Curves", height=450)
fig.update_layout(**layout)
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# Metrics comparison table
# ---------------------------------------------------------------------------
st.markdown("### Performance Metrics")
all_metrics = {pid: results[pid].get("metrics", {}) for pid in selected}
short_labels = [cards[pid].get("title", pid)[:25] for pid in selected]
render_metrics_comparison(all_metrics, short_labels)

# ---------------------------------------------------------------------------
# Radar chart
# ---------------------------------------------------------------------------
st.markdown("### Risk-Return Profile")

radar_metrics = ["sharpe_ratio", "sortino_ratio", "calmar_ratio", "win_rate", "profit_factor"]
radar_labels = ["Sharpe", "Sortino", "Calmar", "Win Rate", "Profit Factor"]

# Normalize each metric to [0, 1] across selected strategies
all_vals = {m: [all_metrics[pid].get(m, 0) for pid in selected] for m in radar_metrics}
norm_vals = {}
for m in radar_metrics:
    vals = all_vals[m]
    mn, mx = min(vals), max(vals)
    rng = mx - mn if mx - mn > 0 else 1
    norm_vals[m] = [(v - mn) / rng for v in vals]

fig_radar = go.Figure()
for i, pid in enumerate(selected):
    r = [norm_vals[m][i] for m in radar_metrics]
    r.append(r[0])  # close the polygon
    fig_radar.add_trace(go.Scatterpolar(
        r=r,
        theta=radar_labels + [radar_labels[0]],
        name=cards[pid].get("title", pid)[:25],
        line=dict(color=colors[i], width=2),
        fill="toself",
        fillcolor=f"rgba({int(colors[i][1:3], 16)},{int(colors[i][3:5], 16)},{int(colors[i][5:7], 16)},0.08)",
    ))

fig_radar.update_layout(
    polar=dict(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(31,41,55,0.5)",
                        tickfont=dict(size=9, color=TEXT_MUTED)),
        angularaxis=dict(gridcolor="rgba(31,41,55,0.5)",
                         tickfont=dict(family="DM Sans, sans-serif", size=11, color="#e5e7eb")),
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#9ca3af"),
    height=400,
    margin=dict(l=60, r=60, t=30, b=30),
    legend=dict(
        font=dict(family="DM Sans, sans-serif", size=11, color=TEXT_MUTED),
        bgcolor="rgba(0,0,0,0)",
    ),
    showlegend=True,
)

st.plotly_chart(fig_radar, width="stretch", config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# Return correlation
# ---------------------------------------------------------------------------
st.markdown("### Return Correlation")

# Build simple daily return series from equity curves and compute correlation
import pandas as pd

return_data = {}
for pid in selected:
    ec = results[pid].get("equity_curve", {})
    values = ec.get("values", [])
    if len(values) > 1:
        arr = np.array(values)
        rets = np.diff(arr) / arr[:-1]
        return_data[cards[pid].get("title", pid)[:20]] = rets

if len(return_data) >= 2:
    min_len = min(len(v) for v in return_data.values())
    df = pd.DataFrame({k: v[:min_len] for k, v in return_data.items()})
    corr = df.corr()

    fig_corr = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate="%{text}",
        textfont=dict(family="JetBrains Mono, monospace", size=12),
        colorscale=[[0, "#ef4444"], [0.5, "#1f2937"], [1, "#10b981"]],
        zmid=0,
        showscale=False,
    ))

    layout = plotly_layout(title="Return Correlation Matrix", height=350)
    fig_corr.update_layout(**layout)
    fig_corr.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_corr, width="stretch", config={"displayModeBar": False})
