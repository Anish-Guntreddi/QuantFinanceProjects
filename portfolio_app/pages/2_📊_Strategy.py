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

# Session-state key for this project's simulation results.
# Cleared whenever the user switches to a different project.
_sim_key = f"sim_results_{selected_id}"
if st.session_state.get("_last_sim_project") != selected_id:
    st.session_state.pop(_sim_key, None)
    st.session_state["_last_sim_project"] = selected_id

if interactive_params:
    with st.expander("Interactive Controls", expanded=False):
        params = render_controls(interactive_params, key_prefix=f"{selected_id}_")

        if can_simulate(selected_id, tier):
            if st.button("▶ Run Simulation", type="primary"):
                with st.spinner("Running simulation..."):
                    _result = run_simulation(selected_id, params)
                if _result and "error" not in _result:
                    st.session_state[_sim_key] = _result
                    st.success("Simulation complete — metrics and equity curve updated below.")
                elif _result:
                    st.error(f"Simulation error: {_result['error']}")

            if _sim_key in st.session_state:
                if st.button("✕ Reset to pre-computed", type="secondary"):
                    st.session_state.pop(_sim_key, None)
                    st.rerun()

            st.caption("Adjust parameters above, then click Run to re-compute metrics and equity curve.")
        elif tier == "cached_sweep":
            st.caption("Results interpolated from pre-computed parameter sweep.")
        else:
            st.caption("Pre-computed results — live simulation not available for this strategy.")

        bp = results.get("backtest_period", {})
        if bp:
            st.caption(f"Backtest period: {bp.get('start', '')} → {bp.get('end', '')}")

# Retrieve persisted sim results (survives slider-change reruns)
sim_results = st.session_state.get(_sim_key)

# Always use pre-computed monthly_returns and sensitivity — sim engines don't produce them
display_metrics  = sim_results["metrics"]       if sim_results else metrics
display_equity   = sim_results                  if sim_results else results
display_monthly  = results.get("monthly_returns", {})          # always pre-computed
display_sensitivity = results.get("parameter_sensitivity", []) # always pre-computed

# Show updated metrics panel when simulation has run
if sim_results:
    st.subheader("Simulated Results")
    render_metrics_panel(display_metrics, cols=4)

# ---------------------------------------------------------------------------
# Charts (full-width, stacked vertically)
# ---------------------------------------------------------------------------
render_equity_curve(display_equity, title="Equity Curve vs Benchmark")

render_monthly_heatmap(display_monthly)

if display_sensitivity:
    st.subheader("Parameter Sensitivity")
    render_sensitivity_chart(display_sensitivity)

# ---------------------------------------------------------------------------
# About
# ---------------------------------------------------------------------------
st.markdown("---")

with st.expander("About This Strategy", expanded=True):
    # ── Meta row ────────────────────────────────────────────────────────────
    data_source = card.get("data_source", "")
    freq = card.get("frequency", "")
    asset_class = card.get("asset_class", "")
    has_cpp = card.get("has_cpp", False)
    languages = card.get("languages", [])

    meta_parts = []
    if asset_class:
        meta_parts.append(f"<strong>Asset class:</strong> {asset_class}")
    if freq:
        meta_parts.append(f"<strong>Frequency:</strong> {freq}")
    if data_source:
        meta_parts.append(f"<strong>Data source:</strong> {data_source}")
    if has_cpp:
        meta_parts.append('<strong>C++ core:</strong> Yes')

    if meta_parts:
        st.markdown(
            f'<p style="font-size: 0.82rem; color: #9ca3af; margin-bottom: 0.75rem;">'
            + " &nbsp;·&nbsp; ".join(meta_parts) + "</p>",
            unsafe_allow_html=True,
        )

    # ── Key techniques ───────────────────────────────────────────────────────
    techniques = card.get("key_techniques", [])
    if techniques:
        tags = " ".join(f'<span class="badge badge-tag">{t}</span>' for t in techniques)
        st.markdown(
            f'<div style="margin-bottom: 1.25rem;">'
            f'<span style="font-size: 0.82rem; color: #9ca3af; font-weight: 600; text-transform: uppercase; '
            f'letter-spacing: 0.05em;">Key Techniques &nbsp;</span>{tags}</div>',
            unsafe_allow_html=True,
        )

    # ── Technical documentation ──────────────────────────────────────────────
    technical_details = card.get("technical_details", "")
    if technical_details:
        st.markdown(
            '<div style="'
            'background: #0d1117; '
            'border: 1px solid #1f2937; '
            'border-radius: 8px; '
            'padding: 1.5rem 1.75rem; '
            'margin-top: 0.5rem;'
            '">',
            unsafe_allow_html=True,
        )
        st.markdown(technical_details)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # Fallback to long_description if no technical_details
        desc = card.get("long_description", card.get("short_description", ""))
        st.markdown(
            f'<p style="font-size: 0.95rem; color: #d1d5db; line-height: 1.6;">{desc}</p>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Notebook viewer
# ---------------------------------------------------------------------------
notebook_path = card.get("notebook_path", "")
if notebook_path:
    render_notebook(notebook_path)
