"""
Architecture — System design, tech stack, and design patterns.
"""

import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Architecture", page_icon="🏗️", layout="wide")

# CSS
_ASSETS = Path(__file__).parent.parent / "assets"
css_file = _ASSETS / "style.css"
if css_file.exists():
    st.markdown(f"<style>{css_file.read_text()}</style>", unsafe_allow_html=True)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import get_all_cards, CATEGORY_DISPLAY_NAMES, CATEGORY_ORDER
from components.theme import CATEGORY_COLORS

st.markdown(
    '<p class="hero-title" style="margin-bottom: 0.25rem;">System Architecture</p>'
    '<h1 style="margin-top: 0;">Infrastructure & Design</h1>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# System topology
# ---------------------------------------------------------------------------
st.markdown("### System Topology")

st.markdown("""
```
┌─────────────┐    ┌──────────────────┐    ┌───────────────┐    ┌──────────────┐
│ Market Data  │───▶│  Feed Handler    │───▶│ Signal Engine │───▶│  Strategy    │
│  (yfinance/  │    │  (C++ / asyncio) │    │  (Python)     │    │  Layer       │
│   synthetic) │    └──────────────────┘    └───────────────┘    └──────┬───────┘
└─────────────┘                                                        │
                                                                       ▼
┌─────────────┐    ┌──────────────────┐    ┌───────────────┐    ┌──────────────┐
│  Analytics  │◀───│ Matching Engine  │◀───│  Execution    │◀───│    Risk      │
│  & Metrics  │    │  (C++ / pybind)  │    │  Engine       │    │   Manager    │
└─────────────┘    └──────────────────┘    └───────────────┘    └──────────────┘
```
""")

# ---------------------------------------------------------------------------
# Technology stack
# ---------------------------------------------------------------------------
st.markdown("### Technology Stack")

stack_items = [
    ("Python 3.11+", "Primary language for research, backtesting, ML training, and portfolio analytics", "#3b82f6"),
    ("C++ 17", "Latency-critical components: matching engine, order book, feed handler (pybind11 bindings)", "#f59e0b"),
    ("PyTorch", "Deep learning models: LSTM, Transformer forecasters, RL agents (Stable Baselines3)", "#ef4444"),
    ("scikit-learn", "Classical ML: Random Forest, XGBoost, LightGBM for signal generation and regime detection", "#10b981"),
    ("cvxpy", "Convex optimization: Mean-Variance, Risk Parity, portfolio constraints", "#8b5cf6"),
    ("QuantLib", "Options pricing, vol surface fitting, Greeks computation", "#06b6d4"),
    ("NumPy / Pandas", "Core data manipulation, vectorized backtesting, time series analysis", "#ec4899"),
    ("Plotly / Streamlit", "Interactive visualization and web dashboard", "#f59e0b"),
]

grid_html = '<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.75rem;">'
for name, desc, color in stack_items:
    grid_html += f"""
    <div class="metric-card" style="text-align: left; border-left: 3px solid {color};">
        <div style="font-family: JetBrains Mono, monospace; font-size: 0.95rem; color: {color}; margin-bottom: 0.25rem;">{name}</div>
        <div style="font-family: DM Sans, sans-serif; font-size: 0.8rem; color: #9ca3af;">{desc}</div>
    </div>
    """
grid_html += "</div>"
st.markdown(grid_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# C++ component benchmarks
# ---------------------------------------------------------------------------
st.markdown("### C++ Component Benchmarks")

bench_html = """
<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem;">
    <div class="metric-card">
        <div class="metric-label">Matching Engine</div>
        <div class="metric-value" style="color: #f59e0b;">~150K</div>
        <div style="font-family: DM Sans; font-size: 0.75rem; color: #6b7280;">orders/sec throughput</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Order Book Update</div>
        <div class="metric-value" style="color: #f59e0b;">&lt;1μs</div>
        <div style="font-family: DM Sans; font-size: 0.75rem; color: #6b7280;">p50 latency</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Feed Handler</div>
        <div class="metric-value" style="color: #f59e0b;">~500K</div>
        <div style="font-family: DM Sans; font-size: 0.75rem; color: #6b7280;">msg/sec parse rate</div>
    </div>
</div>
"""
st.markdown(bench_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Design patterns
# ---------------------------------------------------------------------------
st.markdown("### Design Patterns")

patterns = [
    ("Strategy Pattern", "Base classes (`BaseRegimeDetector`, `BaseFactor`, `BaseStrategy`) with pluggable implementations. Enables fair backtesting of multiple approaches against the same data."),
    ("Pipeline Pattern", "`FactorPipeline` chains data loading → feature engineering → factor calculation → analytics. ML training uses similar pipeline with embargo cross-validation."),
    ("Event-Driven", "Feed handlers and backtesting engines use observer/callback patterns. Orders, fills, and market data propagate through event queues."),
    ("Factory Pattern", "`model_factory.py` constructs neural architectures from config. Strategy configs are loaded from YAML and instantiated at runtime."),
    ("Point-in-Time", "All backtests use point-in-time data snapshots to prevent look-ahead bias. Factor values computed with only historically available data."),
    ("Embargo Cross-Validation", "Time series CV with purging/embargo gaps prevents information leakage between train/test folds. Critical for ML strategy validation."),
]

for name, desc in patterns:
    st.markdown(
        f"""<div class="metric-card" style="text-align: left; margin-bottom: 0.5rem;">
        <div style="font-family: JetBrains Mono, monospace; font-size: 0.9rem; color: #f59e0b; margin-bottom: 0.25rem;">{name}</div>
        <div style="font-family: DM Sans, sans-serif; font-size: 0.85rem; color: #9ca3af; line-height: 1.5;">{desc}</div>
        </div>""",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Project distribution
# ---------------------------------------------------------------------------
st.markdown("### Project Distribution")

cards = get_all_cards()
cat_counts = {}
for c in cards:
    cat = c.get("category", "unknown")
    cat_counts[cat] = cat_counts.get(cat, 0) + 1

import plotly.graph_objects as go
from components.theme import plotly_layout

cats = [CATEGORY_DISPLAY_NAMES.get(c, c) for c in CATEGORY_ORDER if c in cat_counts]
counts = [cat_counts[c] for c in CATEGORY_ORDER if c in cat_counts]
colors = [CATEGORY_COLORS.get(c, "#f59e0b") for c in CATEGORY_ORDER if c in cat_counts]

fig = go.Figure(go.Bar(
    x=cats, y=counts,
    marker_color=colors,
    text=counts,
    textposition="outside",
    textfont=dict(family="JetBrains Mono, monospace", size=12, color="#e5e7eb"),
))
fig.update_layout(**plotly_layout("Projects by Category", height=350))
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# Testing & config
# ---------------------------------------------------------------------------
st.markdown("### Testing & Configuration")

st.markdown("""
- **Testing:** `pytest` with 25 test files across 18 projects. Run per-project: `pytest <project>/tests/ -v`
- **Configuration:** YAML-based per-project configs (strategy parameters, risk limits, data sources). Loaded via `PyYAML` with dataclass validation.
- **ML Ops:** MLflow + Weights & Biases for experiment tracking. Hydra for hierarchical config composition.
- **CI:** pytest + black + flake8 + mypy for code quality.
""")
