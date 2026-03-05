"""Architecture — System design and tech stack showcase."""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.theme import load_css, CATEGORY_COLORS, CATEGORY_DISPLAY, CATEGORY_ICONS

load_css()

st.markdown("#### 🏗️ System Architecture")
st.markdown("How the quantitative research platform is designed — from data ingestion to execution.")

# System topology
st.markdown("##### System Topology")
st.markdown("""
```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌─────────────┐
│  Market Data │────▶│ Feed Handler │────▶│ Signal Engine │────▶│  Strategy   │
│  (yfinance,  │     │ (UDP, async, │     │ (Features,    │     │  Layer      │
│   CCXT, FRED)│     │  ring buffer)│     │  ML models)   │     │             │
└─────────────┘     └──────────────┘     └───────────────┘     └──────┬──────┘
                                                                       │
                    ┌──────────────┐     ┌───────────────┐     ┌──────▼──────┐
                    │  Matching    │◀────│  Execution    │◀────│    Risk     │
                    │  Engine      │     │  Engine       │     │  Manager    │
                    │  (LOB, FIFO) │     │  (VWAP, IS)   │     │  (VaR, DD)  │
                    └──────────────┘     └───────────────┘     └─────────────┘
```
""")

# Tech stack
st.markdown("##### Technology Stack")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    **Core Languages**
    - Python 3.8+ — research, backtesting, ML
    - C++20 — latency-critical components
    - pybind11 — Python ↔ C++ bindings
    """)

with c2:
    st.markdown("""
    **ML / Deep Learning**
    - PyTorch — LSTM, Transformer, RL agents
    - scikit-learn — factor models, feature selection
    - stable-baselines3 — PPO, DQN, SAC agents
    - hmmlearn — Hidden Markov Models
    """)

with c3:
    st.markdown("""
    **Quantitative Finance**
    - cvxpy — convex optimization
    - QuantLib — options pricing
    - statsmodels — time series analysis
    - arch — GARCH models
    """)

st.divider()

# Design patterns
st.markdown("##### Design Patterns")

patterns = [
    ("Strategy Pattern", "Base classes (`BaseRegimeDetector`, `BaseFactor`, `BaseStrategy`) with pluggable implementations. Allows swapping models without changing pipeline code."),
    ("Pipeline Pattern", "`FactorPipeline` chains data loading → feature engineering → signal generation → execution. Each stage is independently testable."),
    ("Event-Driven", "Market events flow through a priority queue: `MarketEvent → SignalEvent → OrderEvent → FillEvent`. Enables realistic backtesting with latency modeling."),
    ("Factory Pattern", "`model_factory.py` creates neural architectures from config. Decouples model definition from training loop."),
    ("Point-in-Time Data", "`PointInTimeJoiner` prevents look-ahead bias by aligning fundamental data to its publication date, not the reporting period."),
    ("Embargo Cross-Validation", "Time series CV with purging (remove overlapping samples) and embargo gaps (buffer between train/test) to prevent information leakage."),
]

for name, desc in patterns:
    with st.expander(f"**{name}**"):
        st.markdown(desc)

st.divider()

# Category breakdown
st.markdown("##### Project Categories")

for cat in ["HFT_strategy_projects", "ai_ml_trading", "core_research_backtesting",
            "market_microstructure_engines", "market_microstructure_execution",
            "research_intraday_strategies", "risk_engineering"]:
    icon = CATEGORY_ICONS.get(cat, "📁")
    display = CATEGORY_DISPLAY.get(cat, cat)
    color = CATEGORY_COLORS.get(cat, "#00D4AA")
    st.markdown(f'<span style="color: {color}; font-weight: 600;">{icon} {display}</span>', unsafe_allow_html=True)

st.divider()

# Performance benchmarks
st.markdown("##### Performance Benchmarks (Simulated)")

b1, b2, b3, b4 = st.columns(4)
b1.markdown("""
<div class="metric-panel">
    <div class="metric-value" style="color: #00D4AA;">100K+</div>
    <div class="metric-label">ORDERS/SEC (LOB)</div>
</div>
""", unsafe_allow_html=True)
b2.markdown("""
<div class="metric-panel">
    <div class="metric-value" style="color: #00D4AA;">&lt;1μs</div>
    <div class="metric-label">DECODE LATENCY</div>
</div>
""", unsafe_allow_html=True)
b3.markdown("""
<div class="metric-panel">
    <div class="metric-value" style="color: #00D4AA;">&lt;20ns</div>
    <div class="metric-label">SPSC QUEUE P99</div>
</div>
""", unsafe_allow_html=True)
b4.markdown("""
<div class="metric-panel">
    <div class="metric-value" style="color: #00D4AA;">1M+</div>
    <div class="metric-label">MSG/SEC (FEED)</div>
</div>
""", unsafe_allow_html=True)

st.divider()

# Testing & config
st.markdown("##### Testing & Configuration")
st.markdown("""
- **Testing**: pytest with per-project test suites (`tests/test_*.py`), benchmark tests for C++ components
- **Configuration**: YAML configs + Python dataclasses; Hydra/OmegaConf for reproducible experiments
- **Reproducibility**: MLflow experiment tracking, DVC data versioning, fixed random seeds
- **CI**: pytest discovery across all projects, code quality via black/flake8/mypy
""")
