"""Theme system — CSS injection, category colors, Plotly template."""

import os
import streamlit as st

CATEGORY_COLORS = {
    "HFT_strategy_projects": "#FF6B35",
    "ai_ml_trading": "#7B68EE",
    "core_research_backtesting": "#00D4AA",
    "market_microstructure_engines": "#FF4757",
    "market_microstructure_execution": "#FFA502",
    "research_intraday_strategies": "#1E90FF",
    "risk_engineering": "#A855F7",
}

CATEGORY_DISPLAY = {
    "HFT_strategy_projects": "HFT Strategies",
    "ai_ml_trading": "AI/ML Trading",
    "core_research_backtesting": "Core Research & Backtesting",
    "market_microstructure_engines": "Microstructure Engines",
    "market_microstructure_execution": "Microstructure Execution",
    "research_intraday_strategies": "Intraday Strategies",
    "risk_engineering": "Risk Engineering",
}

CATEGORY_ICONS = {
    "HFT_strategy_projects": "⚡",
    "ai_ml_trading": "🧠",
    "core_research_backtesting": "🔬",
    "market_microstructure_engines": "⚙️",
    "market_microstructure_execution": "📡",
    "research_intraday_strategies": "📊",
    "risk_engineering": "🛡️",
}

PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "#1A1F2E",
        "font": {"family": "JetBrains Mono, monospace", "color": "#E0E0E0", "size": 11},
        "title": {"font": {"family": "Space Grotesk, sans-serif", "size": 16, "color": "#E0E0E0"}},
        "xaxis": {"gridcolor": "#2D3748", "zerolinecolor": "#2D3748", "linecolor": "#2D3748"},
        "yaxis": {"gridcolor": "#2D3748", "zerolinecolor": "#2D3748", "linecolor": "#2D3748"},
        "colorway": ["#00D4AA", "#FF6B35", "#7B68EE", "#1E90FF", "#FFA502", "#FF4757", "#A855F7", "#F59E0B"],
        "margin": {"l": 60, "r": 20, "t": 50, "b": 40},
        "legend": {"bgcolor": "rgba(0,0,0,0)", "font": {"size": 10}},
    }
}

PROFIT_COLOR = "#00D4AA"
LOSS_COLOR = "#FF4757"


def load_css():
    """Inject custom CSS into the Streamlit app."""
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def get_plotly_layout(**overrides):
    """Get a Plotly layout dict with theme defaults."""
    layout = dict(PLOTLY_TEMPLATE["layout"])
    layout.update(overrides)
    return layout
