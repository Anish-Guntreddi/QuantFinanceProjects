"""Data loading utilities for the portfolio app.

Reads project metadata (cards), backtest results, and manifest from the
portfolio_app/data/ directory.  All public functions are cached via
Streamlit's @st.cache_data so JSON is parsed at most once per session.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import streamlit as st

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ---------------------------------------------------------------------------
# Core loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_manifest() -> dict[str, Any]:
    with open(DATA_DIR / "manifest.json") as f:
        return json.load(f)


@st.cache_data(ttl=3600)
def load_card(project_id: str) -> dict[str, Any]:
    manifest = load_manifest()
    for p in manifest["projects"]:
        if p["id"] == project_id:
            with open(DATA_DIR / p["card_path"]) as f:
                return json.load(f)
    raise KeyError(f"Project '{project_id}' not found in manifest")


@st.cache_data(ttl=3600)
def load_results(project_id: str) -> dict[str, Any]:
    manifest = load_manifest()
    for p in manifest["projects"]:
        if p["id"] == project_id:
            with open(DATA_DIR / p["results_path"]) as f:
                return json.load(f)
    raise KeyError(f"Project '{project_id}' not found in manifest")


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_all_cards() -> list[dict[str, Any]]:
    manifest = load_manifest()
    cards = []
    for p in manifest["projects"]:
        with open(DATA_DIR / p["card_path"]) as f:
            card = json.load(f)
        card["simulation_tier"] = p["simulation_tier"]
        cards.append(card)
    return cards


@st.cache_data(ttl=3600)
def get_projects_by_category() -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for card in get_all_cards():
        groups[card["category"]].append(card)
    return dict(groups)


def get_project_ids() -> list[str]:
    return [p["id"] for p in load_manifest()["projects"]]


CATEGORY_DISPLAY_NAMES = {
    "HFT_strategy_projects": "HFT Strategies",
    "ai_ml_trading": "AI/ML Trading",
    "core_research_backtesting": "Core Research & Backtesting",
    "market_microstructure_engines": "Microstructure Engines",
    "market_microstructure_execution": "Microstructure Execution",
    "research_intraday_strategies": "Intraday Strategies",
    "risk_engineering": "Risk Engineering",
}

CATEGORY_ICONS = {
    "HFT_strategy_projects": "\u26a1",
    "ai_ml_trading": "\U0001f9e0",
    "core_research_backtesting": "\U0001f52c",
    "market_microstructure_engines": "\u2699\ufe0f",
    "market_microstructure_execution": "\U0001f4e1",
    "research_intraday_strategies": "\U0001f4c8",
    "risk_engineering": "\U0001f6e1\ufe0f",
}

CATEGORY_ORDER = [
    "HFT_strategy_projects",
    "ai_ml_trading",
    "core_research_backtesting",
    "market_microstructure_engines",
    "market_microstructure_execution",
    "research_intraday_strategies",
    "risk_engineering",
]
