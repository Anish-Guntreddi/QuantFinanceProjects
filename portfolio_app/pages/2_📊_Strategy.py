"""
Strategy Detail — Deep-dive into a single project with interactive simulation.
"""

import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Strategy Detail", page_icon="📊", layout="wide")

# Load CSS
_ASSETS = Path(__file__).parent.parent / "assets"
css_file = _ASSETS / "style.css"
if css_file.exists():
    st.markdown(f"<style>{css_file.read_text()}</style>", unsafe_allow_html=True)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import (
    load_card, load_results, get_project_ids,
    CATEGORY_DISPLAY_NAMES, CATEGORY_ICONS,
)
from components.theme import CATEGORY_COLORS
from components.metrics_panel import render_metrics_panel
from components.equity_curve import render_equity_curve
from components.heatmap import render_monthly_heatmap
from components.parameter_controls import render_controls, render_sensitivity_chart
from components.notebook_viewer import render_notebook
from simulations.dispatcher import can_simulate, run_simulation

# ---------------------------------------------------------------------------
# Project selection
# ---------------------------------------------------------------------------
project_ids = get_project_ids()

# Check if navigated from dashboard
default_idx = 0
if "selected_project" in st.session_state:
    pid = st.session_state["selected_project"]
    if pid in project_ids:
        default_idx = project_ids.index(pid)

with st.sidebar:
    selected_id = st.selectbox(
        "Select Strategy",
        project_ids,
        index=default_idx,
        format_func=lambda x: x.replace("_", " ").title(),
    )

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
card = load_card(selected_id)
results = load_results(selected_id)
metrics = results.get("metrics", {})
category = card.get("category", "")
cat_color = CATEGORY_COLORS.get(category, "#f59e0b")
cat_name = CATEGORY_DISPLAY_NAMES.get(category, category)
cat_icon = CATEGORY_ICONS.get(category, "")

# ---------------------------------------------------------------------------
# Header — each element is a separate st.markdown so Streamlit can
# measure heights correctly (avoids overlap with expanders below).
# ---------------------------------------------------------------------------
lang_badges = ''.join(
    f'<span class="badge badge-{"python" if l=="Python" else "cpp"}">{l}</span>'
    for l in card.get("languages", [])
)
st.markdown(
    f'<div style="margin-bottom: 0.5rem;">'
    f'<span class="badge badge-category" style="color: {cat_color}; border: 1px solid {cat_color}33; '
    f'background: {cat_color}15;">{cat_icon} {cat_name}</span> {lang_badges}</div>',
    unsafe_allow_html=True,
)
st.title(card.get("title", ""))
st.caption(card.get("short_description", ""))

# ---------------------------------------------------------------------------
# Metrics panel (full-width row)
# ---------------------------------------------------------------------------
render_metrics_panel(metrics, cols=4)

# ---------------------------------------------------------------------------
# Controls row (compact, above charts)
# ---------------------------------------------------------------------------
tier = card.get("simulation_tier", "precomputed")
interactive_params = card.get("interactive_params", [])

sim_results = None  # Will hold live simulation output if run

if interactive_params:
    with st.expander("Interactive Controls", expanded=False):
        params = render_controls(interactive_params, key_prefix=f"{selected_id}_")

        if can_simulate(selected_id, tier):
            if st.button("▶ Run Simulation", type="primary"):
                with st.spinner("Running simulation..."):
                    sim_results = run_simulation(selected_id, params)
                if sim_results and "error" not in sim_results:
                    st.success("Simulation complete — results updated below.")
                elif sim_results:
                    st.error(f"Simulation error: {sim_results['error']}")
            st.caption("Adjust parameters above, then click Run to re-compute metrics and equity curve.")
        elif tier == "cached_sweep":
            st.caption("Results interpolated from pre-computed parameter sweep.")
        else:
            st.caption("Pre-computed results — live simulation not available for this strategy.")

        bp = results.get("backtest_period", {})
        if bp:
            st.caption(f"Backtest period: {bp.get('start', '')} → {bp.get('end', '')}")

# Use simulation results if available, otherwise fall back to pre-computed
display_results = sim_results if (sim_results and "error" not in sim_results) else results
display_metrics = display_results.get("metrics", metrics)

# Re-render metrics if simulation was run (override the static panel above)
if sim_results and "error" not in sim_results:
    st.subheader("Simulated Results")
    render_metrics_panel(display_metrics, cols=4)

# ---------------------------------------------------------------------------
# Charts (full-width, stacked vertically)
# ---------------------------------------------------------------------------
render_equity_curve(display_results, title="Equity Curve vs Benchmark")

render_monthly_heatmap(display_results.get("monthly_returns", {}))

sensitivity = display_results.get("parameter_sensitivity", [])
if sensitivity:
    st.subheader("Parameter Sensitivity")
    render_sensitivity_chart(sensitivity)

# ---------------------------------------------------------------------------
# About
# ---------------------------------------------------------------------------
st.markdown("---")

with st.expander("About This Strategy", expanded=False):
    desc = card.get("long_description", card.get("short_description", ""))
    st.markdown(
        f'<p style="font-size: 0.95rem; color: #d1d5db; line-height: 1.6; margin-bottom: 1rem;">{desc}</p>',
        unsafe_allow_html=True,
    )

    data_source = card.get("data_source", "")
    freq = card.get("frequency", "")
    asset_class = card.get("asset_class", "")
    if data_source or freq:
        st.markdown(
            f'<p style="font-size: 0.85rem; color: #9ca3af;">'
            f'<strong>Data source:</strong> {data_source} · '
            f'<strong>Frequency:</strong> {freq} · '
            f'<strong>Asset class:</strong> {asset_class}</p>',
            unsafe_allow_html=True,
        )

    techniques = card.get("key_techniques", [])
    if techniques:
        tags = " ".join(f'<span class="badge badge-tag">{t}</span>' for t in techniques)
        st.markdown(
            f'<p style="margin-top: 0.75rem;"><strong style="font-size: 0.85rem; color: #9ca3af;">'
            f'Key Techniques:</strong> {tags}</p>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Notebook viewer
# ---------------------------------------------------------------------------
notebook_path = card.get("notebook_path", "")
if notebook_path:
    render_notebook(notebook_path)
