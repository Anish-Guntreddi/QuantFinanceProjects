"""Quantitative Research Lab — Portfolio App Entry Point."""

import streamlit as st
import os
import sys

# Add app dir to path for component imports
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

st.set_page_config(
    page_title="Quantitative Research Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load CSS
from components.theme import load_css
load_css()

# Sidebar
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem 0;">
    <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.2rem; font-weight: 700; color: #00D4AA;">
        QUANT RESEARCH LAB
    </div>
    <div style="font-family: 'IBM Plex Sans', sans-serif; font-size: 0.8rem; color: #9CA3AF; margin-top: 4px;">
        34 Projects · 7 Categories
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.divider()
st.sidebar.markdown("**Navigation**")
st.sidebar.page_link("app.py", label="Dashboard", icon="🏠")
st.sidebar.page_link("pages/2_Strategy_Detail.py", label="Strategy Detail", icon="📊")
st.sidebar.page_link("pages/3_Compare.py", label="Compare", icon="📈")
st.sidebar.page_link("pages/4_Research_Lab.py", label="Research Lab", icon="🔬")
st.sidebar.page_link("pages/5_Architecture.py", label="Architecture", icon="🏗️")

# Main Dashboard content (same as page 1)
from utils.data_loader import load_all_cards, get_projects_by_category
from components.cards import render_card_grid
from components.theme import CATEGORY_DISPLAY, CATEGORY_ICONS, CATEGORY_COLORS

cards = load_all_cards()
groups = get_projects_by_category(cards)

# Hero
st.markdown("""
<div style="padding: 2rem 0 1rem 0;">
    <div class="hero-title">QUANTITATIVE RESEARCH LAB</div>
    <div class="hero-subtitle">Systematic trading strategies, market microstructure engines, and risk systems</div>
</div>
""", unsafe_allow_html=True)

# Stats bar
avg_sharpe = 0
if cards:
    sharpes = [c.get("headline_metric", {}).get("value", 0) for c in cards if c.get("headline_metric", {}).get("value", 0) > 0]
    avg_sharpe = sum(sharpes) / len(sharpes) if sharpes else 0
n_cpp = sum(1 for c in cards if c.get("has_cpp"))

cols = st.columns(4)
with cols[0]:
    st.markdown(f'<div class="metric-panel"><div class="metric-value" style="color: #00D4AA;">{len(cards)}</div><div class="metric-label">PROJECTS</div></div>', unsafe_allow_html=True)
with cols[1]:
    st.markdown(f'<div class="metric-panel"><div class="metric-value" style="color: #00D4AA;">{len(groups)}</div><div class="metric-label">CATEGORIES</div></div>', unsafe_allow_html=True)
with cols[2]:
    st.markdown(f'<div class="metric-panel"><div class="metric-value" style="color: #00D4AA;">{avg_sharpe:.2f}</div><div class="metric-label">AVG SHARPE</div></div>', unsafe_allow_html=True)
with cols[3]:
    st.markdown(f'<div class="metric-panel"><div class="metric-value" style="color: #00D4AA;">{n_cpp}</div><div class="metric-label">C++ COMPONENTS</div></div>', unsafe_allow_html=True)

st.divider()

# Category sections
for cat in ["HFT_strategy_projects", "ai_ml_trading", "core_research_backtesting",
            "market_microstructure_engines", "market_microstructure_execution",
            "research_intraday_strategies", "risk_engineering"]:
    cat_cards = groups.get(cat, [])
    if not cat_cards:
        continue
    icon = CATEGORY_ICONS.get(cat, "📁")
    display = CATEGORY_DISPLAY.get(cat, cat)
    color = CATEGORY_COLORS.get(cat, "#00D4AA")

    with st.expander(f"{icon} {display} ({len(cat_cards)})", expanded=(cat == "HFT_strategy_projects")):
        render_card_grid(cat_cards, cols=3)
