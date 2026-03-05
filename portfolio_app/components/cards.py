"""Strategy card HTML component."""

import streamlit as st
from components.theme import CATEGORY_COLORS
from utils.formatting import sharpe_color


def render_strategy_card(card, key_prefix=""):
    """Render a single strategy card as clickable HTML."""
    cat = card.get("category", "")
    accent = CATEGORY_COLORS.get(cat, "#00D4AA")
    headline = card.get("headline_metric", {})
    metric_val = headline.get("value", 0)
    metric_name = headline.get("name", "Sharpe")
    color = sharpe_color(metric_val)
    langs = card.get("languages", ["Python"])
    badges = "".join(
        f'<span class="badge badge-{"cpp" if l == "C++" else "python"}">{l}</span>'
        for l in langs
    )

    html = f"""
    <div class="strategy-card" style="border-left-color: {accent};">
        <div class="card-title">{card.get("title", "Untitled")}</div>
        <div class="card-description">{card.get("short_description", "")}</div>
        <div class="card-metric" style="color: {color};">{metric_name} {metric_val:.2f}</div>
        <div class="card-badges">{badges}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_card_grid(cards, cols=3):
    """Render cards in a grid layout."""
    columns = st.columns(cols)
    for i, card in enumerate(cards):
        with columns[i % cols]:
            render_strategy_card(card, key_prefix=f"card_{i}")
