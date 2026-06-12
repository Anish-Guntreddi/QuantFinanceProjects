"""Strategy card components for the Dashboard grid view."""

from __future__ import annotations

from typing import Any

import streamlit as st

from .theme import CATEGORY_COLORS


def _language_badges(languages: list[str]) -> str:
    badges = []
    for lang in languages:
        cls = "badge-python" if lang == "Python" else "badge-cpp"
        badges.append(f'<span class="badge {cls}">{lang}</span>')
    return " ".join(badges)


def _tier_badge(tier: str) -> str:
    cls = {"live": "tier-live", "cached_sweep": "tier-cached", "precomputed": "tier-precomputed"}.get(tier, "tier-precomputed")
    label = {"live": "LIVE", "cached_sweep": "CACHED", "precomputed": "PRE-COMPUTED"}.get(tier, tier.upper())
    return f'<span class="badge {cls}">{label}</span>'


def render_strategy_card(card: dict[str, Any], index: int = 0) -> None:
    """Render a single strategy card as styled HTML."""
    color = CATEGORY_COLORS.get(card.get("category", ""), "#f59e0b")
    headline = card.get("headline_metric", {})
    metric_name = headline.get("name", "Sharpe")
    metric_val = headline.get("value", 0)
    metric_color = "#10b981" if metric_val > 0 else "#ef4444"

    html = f"""
    <div class="strategy-card" style="border-left-color: {color}; animation-delay: {index * 0.04}s;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5rem;">
            <div class="card-title">{card.get('title', '')}</div>
            {_tier_badge(card.get('simulation_tier', 'precomputed'))}
        </div>
        <div class="card-description">{card.get('short_description', '')}</div>
        <div style="display: flex; justify-content: space-between; align-items: flex-end;">
            <div>
                <div class="card-metric-label">{metric_name}</div>
                <div class="card-metric" style="color: {metric_color};">{metric_val:.2f}</div>
            </div>
            <div>{_language_badges(card.get('languages', ['Python']))}</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_card_grid(cards: list[dict[str, Any]], cols: int = 3) -> str | None:
    """Render cards in a column grid. Returns the project_id if a card is clicked."""
    selected = None
    columns = st.columns(cols)
    for i, card in enumerate(cards):
        with columns[i % cols]:
            render_strategy_card(card, index=i)
            if st.button("View Details", key=f"btn_{card['project_id']}", width="stretch"):
                selected = card["project_id"]
    return selected
