"""Data loader — manifest parsing, JSON loading, caching."""

import json
import os
import streamlit as st

APP_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")


@st.cache_data(ttl=3600)
def load_manifest():
    """Load the project manifest."""
    path = os.path.join(DATA_DIR, "manifest.json")
    if not os.path.exists(path):
        return {"categories": {}, "projects": []}
    with open(path) as f:
        return json.load(f)


@st.cache_data(ttl=3600)
def load_strategy_card(project_id):
    """Load a single strategy card."""
    path = os.path.join(DATA_DIR, "cards", f"{project_id}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data(ttl=3600)
def load_results(project_id):
    """Load results for a project."""
    path = os.path.join(DATA_DIR, "results", f"{project_id}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data(ttl=3600)
def load_all_cards():
    """Load all strategy cards."""
    cards_dir = os.path.join(DATA_DIR, "cards")
    cards = []
    if not os.path.exists(cards_dir):
        return cards
    for fn in sorted(os.listdir(cards_dir)):
        if fn.endswith(".json"):
            with open(os.path.join(cards_dir, fn)) as f:
                cards.append(json.load(f))
    return cards


def get_projects_by_category(cards):
    """Group cards by category."""
    groups = {}
    for card in cards:
        cat = card.get("category", "Other")
        if cat not in groups:
            groups[cat] = []
        groups[cat].append(card)
    return groups
