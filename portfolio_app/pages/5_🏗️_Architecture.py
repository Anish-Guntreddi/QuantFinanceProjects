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

from utils.data_loader import get_all_cards, load_card, CATEGORY_DISPLAY_NAMES, CATEGORY_ORDER
from components.theme import CATEGORY_COLORS

# ---------------------------------------------------------------------------
# Strategy context (set when navigating from Strategy page)
# ---------------------------------------------------------------------------
_arch_pid = st.session_state.get("arch_project")
_arch_card = load_card(_arch_pid) if _arch_pid else None


def _strategy_diagram(card: dict) -> tuple[str, list[tuple[str, str, str]]]:
    """Return (ascii_diagram, component_list) for a given strategy card."""
    cat = card.get("category", "")
    has_cpp = card.get("has_cpp", False)
    techniques = card.get("key_techniques", [])
    data_src = card.get("data_source", "yfinance")

    if cat == "HFT_strategy_projects":
        diagram = (
            "┌──────────────────┐   ┌──────────────────┐   ┌───────────────────┐\n"
            "│  Synthetic LOB   │──▶│  Feed Handler    │──▶│  Signal Engine    │\n"
            "│  (Hawkes/GBM)    │   │  (C++ / asyncio) │   │  (OBI / Imbalance)│\n"
            "└──────────────────┘   └──────────────────┘   └────────┬──────────┘\n"
            "                                                        │\n"
            "┌──────────────────┐   ┌──────────────────┐   ┌────────▼──────────┐\n"
            "│  Analytics &     │◀──│  Matching Engine │◀──│  Market Making    │\n"
            "│  Metrics         │   │  (C++ / pybind11)│   │  Engine + Risk    │\n"
            "└──────────────────┘   └──────────────────┘   └───────────────────┘"
        )
        components = [
            ("Synthetic LOB", "Hawkes process order arrivals, GBM mid-price, price-time priority book", "#ef4444"),
            ("Feed Handler (C++)", "Lock-free SPSC queue, binary protocol parsing, sub-microsecond latency", "#f59e0b"),
            ("Signal Engine", "OBI computation (SIMD/AVX2), trade flow imbalance, quote scoring", "#3b82f6"),
            ("Matching Engine (C++)", "pybind11-wrapped order matching, fill simulation, queue management", "#f59e0b"),
            ("Market Making Engine", "Avellaneda-Stoikov / RL agent for quote placement and inventory control", "#10b981"),
            ("Risk Manager", "Hard inventory limits, PnL stop-loss, spread-adjusted position caps", "#8b5cf6"),
        ]
    elif cat == "ai_ml_trading":
        diagram = (
            "┌──────────────────┐   ┌──────────────────┐   ┌───────────────────┐\n"
            f"│  Market Data     │──▶│  Feature Eng.    │──▶│  ML Model         │\n"
            f"│  ({data_src})    │   │  (50+ features)  │   │  (LSTM/Transform) │\n"
            "└──────────────────┘   └──────────────────┘   └────────┬──────────┘\n"
            "                                                        │\n"
            "┌──────────────────┐   ┌──────────────────┐   ┌────────▼──────────┐\n"
            "│  Backtest &      │◀──│  Position Sizing │◀──│  Signal + Regime  │\n"
            "│  Analytics       │   │  (vol targeting) │   │  Classification   │\n"
            "└──────────────────┘   └──────────────────┘   └───────────────────┘"
        )
        components = [
            ("Data Pipeline", f"yfinance OHLCV + VIX + credit spreads, point-in-time construction", "#3b82f6"),
            ("Feature Engineering", "Price-derived, technical, microstructure, cross-sectional (50+ features)", "#10b981"),
            ("ML Model", "LSTM (bidirectional + attention) and/or Transformer encoder, calibrated outputs", "#8b5cf6"),
            ("Regime Classifier", "HMM / Markov-switching / K-Means for market state identification", "#a78bfa"),
            ("Signal Layer", "Probability threshold filtering, embargoed CV validation, SHAP attribution", "#f59e0b"),
            ("Portfolio Construction", "Volatility-targeting position sizing, sector neutralization", "#06b6d4"),
        ]
    elif cat in ("market_microstructure_engines", "market_microstructure_execution"):
        diagram = (
            "┌──────────────────┐   ┌──────────────────┐   ┌───────────────────┐\n"
            "│  Market Data     │──▶│  Feed Handler    │──▶│  Order Book       │\n"
            "│  (UDP multicast) │   │  (ring buffer)   │   │  (C++ / pybind11) │\n"
            "└──────────────────┘   └──────────────────┘   └────────┬──────────┘\n"
            "                                                        │\n"
            "┌──────────────────┐   ┌──────────────────┐   ┌────────▼──────────┐\n"
            "│  TCA / Metrics   │◀──│  Execution Algos │◀──│  Smart Router     │\n"
            "│  (IS, VWAP slip) │   │  (VWAP/TWAP/IS)  │   │  (venue selection)│\n"
            "└──────────────────┘   └──────────────────┘   └───────────────────┘"
        )
        components = [
            ("Feed Handler", "UDP multicast reception, lock-free ring buffer, FIX protocol encoding", "#f59e0b"),
            ("Order Book (C++)", "Price-time priority, Hawkes arrival process, iceberg/stop/limit orders", "#ef4444"),
            ("Smart Order Router", "Multi-venue cost optimization, fill probability estimation, dark pool allocation", "#3b82f6"),
            ("Execution Algorithms", "VWAP, TWAP, POV, Almgren-Chriss IS — with dynamic schedule adjustment", "#10b981"),
            ("Risk Checks", "Pre-trade: fat-finger, position limits, rate limits, buying power (< 1µs)", "#8b5cf6"),
            ("TCA Framework", "IS, VWAP slippage, market impact coefficient, spread cost attribution", "#06b6d4"),
        ]
    elif any(k in techniques for k in ["cointegration", "pairs-trading", "statistical-arbitrage"]):
        diagram = (
            "┌──────────────────┐   ┌──────────────────┐   ┌───────────────────┐\n"
            f"│  Market Data     │──▶│  Cointegration   │──▶│  Spread Modeling  │\n"
            f"│  ({data_src})    │   │  (EG / Johansen) │   │  (OU process)     │\n"
            "└──────────────────┘   └──────────────────┘   └────────┬──────────┘\n"
            "                                                        │\n"
            "┌──────────────────┐   ┌──────────────────┐   ┌────────▼──────────┐\n"
            "│  Analytics &     │◀──│  Portfolio Mgmt  │◀──│  Entry/Exit       │\n"
            "│  Metrics         │   │  (hedge ratios)  │   │  (z-score signals)│\n"
            "└──────────────────┘   └──────────────────┘   └───────────────────┘"
        )
        components = [
            ("Data Pipeline", f"{data_src} OHLCV, point-in-time construction, sector universe filtering", "#3b82f6"),
            ("Cointegration Tests", "Engle-Granger ADF test, Johansen multivariate test, p-value screening", "#10b981"),
            ("Spread / OU Model", "Kalman filter hedge ratios, OU parameter estimation (κ, μ, σ) via MLE", "#8b5cf6"),
            ("Signal Generation", "Z-score entry/exit, half-life filtering, structural break detection", "#f59e0b"),
            ("Portfolio Manager", "Dollar-neutral long/short, dynamic hedge ratio updates, pair rotation", "#a78bfa"),
            ("Backtesting Engine", "Event-driven, walk-forward validation, realistic slippage and costs", "#06b6d4"),
        ]
    elif cat == "risk_engineering":
        diagram = (
            "┌──────────────────┐   ┌──────────────────┐   ┌───────────────────┐\n"
            "│  Market Data     │──▶│  Covariance Est. │──▶│  Portfolio        │\n"
            f"│  ({data_src})    │   │  (EWMA/LW/NCO)   │   │  Optimizer(CVXPY) │\n"
            "└──────────────────┘   └──────────────────┘   └────────┬──────────┘\n"
            "                                                        │\n"
            "┌──────────────────┐   ┌──────────────────┐   ┌────────▼──────────┐\n"
            "│  Risk Analytics  │◀──│  Constraint Mgmt │◀──│  Weight           │\n"
            "│  (VaR/CVaR/DD)   │   │  (box/sector)    │   │  Allocation       │\n"
            "└──────────────────┘   └──────────────────┘   └───────────────────┘"
        )
        components = [
            ("Data Pipeline", "yfinance daily OHLCV, return series construction, outlier handling", "#3b82f6"),
            ("Covariance Estimators", "Sample, EWMA (λ=0.94), Ledoit-Wolf shrinkage, NCO hierarchical clustering", "#10b981"),
            ("Portfolio Optimizer", "CVXPY quadratic program: MV, Risk Parity (Newton), Kelly, Black-Litterman", "#8b5cf6"),
            ("Constraint Manager", "Max weight, sector limits, turnover constraints, long-only / long-short", "#f59e0b"),
            ("Risk Analytics", "VaR/CVaR (historical/parametric), max drawdown, contribution decomposition", "#ef4444"),
            ("MLflow / DVC", "Experiment tracking, data versioning, reproducibility validation", "#a78bfa"),
        ]
    else:  # intraday / research / default
        diagram = (
            "┌──────────────────┐   ┌──────────────────┐   ┌───────────────────┐\n"
            f"│  Market Data     │──▶│  Technical Ind.  │──▶│  Signal Engine    │\n"
            f"│  ({data_src})    │   │  (EMA/RSI/MACD)  │   │  (entry/exit)     │\n"
            "└──────────────────┘   └──────────────────┘   └────────┬──────────┘\n"
            "                                                        │\n"
            "┌──────────────────┐   ┌──────────────────┐   ┌────────▼──────────┐\n"
            "│  Analytics &     │◀──│  Backtest Engine │◀──│  Position Sizing  │\n"
            "│  Metrics         │   │  (event-driven)  │   │  (vol targeting)  │\n"
            "└──────────────────┘   └──────────────────┘   └───────────────────┘"
        )
        components = [
            ("Data Pipeline", f"{data_src} OHLCV ingestion, point-in-time construction, universe filtering", "#3b82f6"),
            ("Technical Indicators", "EMA/SMA crossover, RSI, MACD, Bollinger Bands, ATR, ADX", "#10b981"),
            ("Signal Engine", "Multi-timeframe confluence, threshold filtering, signal aggregation", "#f59e0b"),
            ("Position Sizing", "Volatility targeting (σ-based), Kelly fraction, risk-parity weights", "#8b5cf6"),
            ("Backtest Engine", "Event-driven queue, walk-forward validation, realistic cost model", "#06b6d4"),
            ("Analytics", "Sharpe, Sortino, Max DD, CAGR, Win Rate, Profit Factor, monthly heatmap", "#a78bfa"),
        ]

    return diagram, components


# Page header
st.markdown(
    '<p class="hero-title" style="margin-bottom: 0.25rem;">System Architecture</p>'
    '<h1 style="margin-top: 0;">Infrastructure & Design</h1>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Strategy-specific section (shown when navigating from Strategy page)
# ---------------------------------------------------------------------------
if _arch_card:
    cat_color = CATEGORY_COLORS.get(_arch_card.get("category", ""), "#f59e0b")
    diagram, components = _strategy_diagram(_arch_card)

    st.markdown(
        f'<div style="background: {cat_color}10; border: 1px solid {cat_color}35; border-radius: 8px; '
        f'padding: 0.75rem 1.25rem; margin-bottom: 1.5rem;">'
        f'<span style="font-size: 0.75rem; color: {cat_color}; font-weight: 600; text-transform: uppercase; '
        f'letter-spacing: 0.05em;">Architecture for</span><br>'
        f'<span style="font-size: 1.05rem; color: #f3f4f6; font-weight: 600;">{_arch_card.get("title", "")}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("##### Component Diagram")
        st.code(diagram, language=None)
    with col_b:
        st.markdown("##### Components")
        for name, desc, color in components:
            st.markdown(
                f'<div style="display: flex; gap: 0.6rem; align-items: flex-start; margin-bottom: 0.5rem;">'
                f'<div style="min-width: 3px; height: auto; background: {color}; border-radius: 2px; align-self: stretch;"></div>'
                f'<div><div style="font-family: JetBrains Mono, monospace; font-size: 0.82rem; color: {color}; '
                f'font-weight: 600; margin-bottom: 0.1rem;">{name}</div>'
                f'<div style="font-size: 0.8rem; color: #9ca3af; line-height: 1.5;">{desc}</div></div></div>',
                unsafe_allow_html=True,
            )

    # Strategy tech stack
    langs = _arch_card.get("languages", ["Python"])
    has_cpp = _arch_card.get("has_cpp", False)
    techniques = _arch_card.get("key_techniques", [])
    freq = _arch_card.get("frequency", "")
    asset_cls = _arch_card.get("asset_class", "")

    tech_tags = []
    if "Python" in langs:
        tech_tags.append(("Python 3.11+", "#3b82f6"))
    if has_cpp or "C++" in langs:
        tech_tags.append(("C++ 20 (pybind11)", "#f59e0b"))
    if any(k in techniques for k in ["lstm", "transformer", "reinforcement-learning", "dqn", "ppo"]):
        tech_tags.append(("PyTorch / SB3", "#ef4444"))
    if any(k in techniques for k in ["hmm", "regime-detection", "clustering"]):
        tech_tags.append(("scikit-learn / hmmlearn", "#10b981"))
    if any(k in techniques for k in ["portfolio-optimization", "risk-parity", "black-litterman", "mean-variance"]):
        tech_tags.append(("cvxpy", "#8b5cf6"))
    if any(k in techniques for k in ["options", "greeks", "volatility-surface", "svi", "sabr"]):
        tech_tags.append(("QuantLib", "#06b6d4"))
    if any(k in techniques for k in ["cointegration", "markov-switching"]):
        tech_tags.append(("statsmodels / arch", "#a78bfa"))
    tech_tags.append(("NumPy / Pandas", "#ec4899"))
    tech_tags.append(("Plotly / Streamlit", "#f59e0b"))

    tags_html = " ".join(
        f'<span style="background: {c}15; border: 1px solid {c}40; color: {c}; '
        f'font-family: JetBrains Mono, monospace; font-size: 0.75rem; padding: 0.2rem 0.6rem; '
        f'border-radius: 4px; white-space: nowrap;">{t}</span>'
        for t, c in tech_tags
    )
    st.markdown(f'<div style="margin-top: 1rem;">##### Tech Stack for This Strategy</div>', unsafe_allow_html=True)
    st.markdown("##### Tech Stack for This Strategy")
    st.markdown(f'<div style="display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.25rem;">{tags_html}</div>',
                unsafe_allow_html=True)

    if st.button("✕ Clear strategy context", key="clear_arch_ctx"):
        st.session_state.pop("arch_project", None)
        st.rerun()

    st.markdown("---")
    st.markdown("### Full System Reference")

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
