"""
Quantitative Research Portfolio — Dashboard
Signal Lab theme · Built with Streamlit
"""

import streamlit as st
from pathlib import Path

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Quant Research Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Load CSS
# ---------------------------------------------------------------------------
_ASSETS = Path(__file__).parent / "assets"

def load_css():
    css_file = _ASSETS / "style.css"
    if css_file.exists():
        st.markdown(f"<style>{css_file.read_text()}</style>", unsafe_allow_html=True)

load_css()

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
from utils.data_loader import (
    get_projects_by_category, get_all_cards, load_manifest,
    CATEGORY_DISPLAY_NAMES, CATEGORY_ICONS, CATEGORY_ORDER,
)
from components.strategy_card import render_strategy_card, render_card_grid
from components.theme import CATEGORY_COLORS

manifest = load_manifest()
all_cards = get_all_cards()
by_category = get_projects_by_category()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<p style="font-family: JetBrains Mono, monospace; font-size: 0.75rem; '
        'color: #f59e0b; letter-spacing: 0.1em; text-transform: uppercase;">Signal Lab</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Category filter
    cat_filter = st.multiselect(
        "Filter by category",
        options=CATEGORY_ORDER,
        format_func=lambda c: f"{CATEGORY_ICONS.get(c, '')} {CATEGORY_DISPLAY_NAMES.get(c, c)}",
        default=[],
    )

    # Simulation tier filter
    tier_filter = st.multiselect(
        "Simulation tier",
        options=["live", "cached_sweep", "precomputed"],
        format_func=lambda t: {"live": "🟢 Live", "cached_sweep": "🟡 Cached", "precomputed": "⚪ Pre-computed"}[t],
        default=[],
    )

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
import streamlit.components.v1 as _components
_components.html(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300..700&family=JetBrains+Mono:wght@300..700&display=swap');
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: transparent; }
        .hero {
            padding: 1.5rem 0 1.25rem 0;
            border-bottom: 1px solid #1f2937;
            margin-bottom: 0;
        }
        .hero-title {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            color: #f59e0b;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            margin-bottom: 0.4rem;
        }
        .hero-name {
            font-family: 'DM Serif Display', Georgia, serif;
            font-size: 2.5rem;
            color: #f9fafb;
            line-height: 1.15;
            margin-bottom: 0.35rem;
        }
        .hero-role {
            font-family: 'DM Sans', sans-serif;
            font-size: 1.1rem;
            color: #9ca3af;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }
        #typed-role {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.05rem;
            color: #f59e0b;
            font-weight: 500;
            border-right: 2px solid #f59e0b;
            padding-right: 2px;
            white-space: nowrap;
        }
        @keyframes blink { 0%,100%{border-color:#f59e0b} 50%{border-color:transparent} }
    </style>

    <div class="hero">
        <div class="hero-title">Quant Finance Portfolio</div>
        <div class="hero-name">Anish Guntreddi</div>
        <div class="hero-role">
            <span>Role:&nbsp;</span>
            <span id="typed-role"></span>
        </div>
    </div>

    <script>
    (function() {
        const el           = document.getElementById("typed-role");
        const words        = ["Quant Dev", "Quant Researcher"];
        const TYPE_SPEED   = 80;
        const DELETE_SPEED = 45;
        const PAUSE_AFTER  = 1800;
        const PAUSE_BEFORE = 400;

        let wordIdx = 0, charIdx = 0, deleting = false;
        el.style.animation = "blink 0.75s step-end infinite";

        function tick() {
            const word = words[wordIdx];
            if (!deleting) {
                el.textContent = word.slice(0, ++charIdx);
                if (charIdx === word.length) {
                    deleting = true;
                    setTimeout(tick, PAUSE_AFTER);
                } else {
                    setTimeout(tick, TYPE_SPEED);
                }
            } else {
                el.textContent = word.slice(0, --charIdx);
                if (charIdx === 0) {
                    deleting = false;
                    wordIdx  = (wordIdx + 1) % words.length;
                    setTimeout(tick, PAUSE_BEFORE);
                } else {
                    setTimeout(tick, DELETE_SPEED);
                }
            }
        }
        setTimeout(tick, 400);
    })();
    </script>
    """,
    height=140,
)

# ---------------------------------------------------------------------------
# Stats bar
# ---------------------------------------------------------------------------
total = len(all_cards)
languages = set()
for c in all_cards:
    languages.update(c.get("languages", []))
n_categories = len(by_category)
avg_sharpe = sum(c.get("headline_metric", {}).get("value", 0) for c in all_cards) / max(total, 1)
n_live = sum(1 for c in all_cards if c.get("simulation_tier") == "live")

cols = st.columns(5)
stat_data = [
    (str(total), "Projects"),
    (", ".join(sorted(languages)), "Languages"),
    (str(n_categories), "Categories"),
    (f"{avg_sharpe:.2f}", "Avg Sharpe"),
    (str(n_live), "Live Sims"),
]
for col, (num, label) in zip(cols, stat_data):
    col.markdown(
        f'<div class="stat-item">'
        f'<div class="stat-number">{num}</div>'
        f'<div class="stat-label">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown('<div style="margin-top: 1rem;"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Project grid grouped by category
# ---------------------------------------------------------------------------
display_categories = cat_filter if cat_filter else CATEGORY_ORDER

for cat in display_categories:
    cards = by_category.get(cat, [])
    if not cards:
        continue

    # Apply tier filter
    if tier_filter:
        cards = [c for c in cards if c.get("simulation_tier") in tier_filter]
        if not cards:
            continue

    display_name = CATEGORY_DISPLAY_NAMES.get(cat, cat)
    icon = CATEGORY_ICONS.get(cat, "")
    color = CATEGORY_COLORS.get(cat, "#f59e0b")

    st.markdown(
        f'<div class="category-header" style="border-bottom-color: {color};">'
        f'{icon} {display_name} <span class="category-count">({len(cards)})</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    selected = render_card_grid(cards, cols=3)
    if selected:
        st.session_state["selected_project"] = selected
        st.switch_page("pages/2_📊_Strategy.py")
