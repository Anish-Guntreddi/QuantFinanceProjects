"""
Data discovery and model loading for quant finance portfolio app.
Scans REPO_ROOT/portfolio_projects for subdirs containing model_card.yaml + optional results.yaml.
"""
import yaml
import streamlit as st
from pathlib import Path

# portfolio_app/ is at REPO_ROOT/portfolio_app/, so parent.parent = REPO_ROOT
REPO_ROOT = Path(__file__).parent.parent.parent

CATEGORY_COLORS = {
    "Backtesting Engine": {"css": "backtesting", "color": "#60a5fa", "bg": "#1e3a5f"},
    "ML Alpha": {"css": "mlalpha", "color": "#fbbf24", "bg": "#422006"},
    "Macro / Regimes": {"css": "macro", "color": "#34d399", "bg": "#064e3b"},
    "Options / Volatility": {"css": "options", "color": "#a78bfa", "bg": "#2e1065"},
    "Crypto / DeFi": {"css": "crypto", "color": "#fb7185", "bg": "#4c0519"},
}

PLOTLY_COLORS = ["#60a5fa", "#fbbf24", "#34d399", "#a78bfa", "#fb7185", "#22d3ee", "#818cf8", "#facc15"]


@st.cache_data
def discover_projects():
    """Scan portfolio_projects/ for project directories with model_card.yaml."""
    projects = []
    projects_root = REPO_ROOT / "portfolio_projects"
    if not projects_root.exists():
        return projects
    for d in sorted(projects_root.iterdir()):
        if not d.is_dir():
            continue
        card_path = d / 'model_card.yaml'
        if not card_path.exists():
            continue
        try:
            card = yaml.safe_load(card_path.read_text(encoding='utf-8'))
        except Exception:
            continue
        results = None
        results_path = d / 'results.yaml'
        if results_path.exists():
            try:
                results = yaml.safe_load(results_path.read_text(encoding='utf-8'))
            except Exception:
                pass
        projects.append({
            'dir': d.name,
            'path': str(d),
            'card': card,
            'results': results,
        })
    # Order by project_id (build order: engine first, consumers after)
    projects.sort(key=lambda p: str(p['card'].get('project_id', '99')))
    return projects


def get_project_by_id(project_id):
    """Get a single project by its ID (e.g. '01')."""
    for p in discover_projects():
        if p['card'].get('project_id') == project_id:
            return p
    return None


def get_category_style(category):
    """Get styling info for a category."""
    return CATEGORY_COLORS.get(category, {"css": "backtesting", "color": "#2563eb", "bg": "#dbeafe"})


def format_metric_value(value, metric_name):
    """Format a metric value for display — quant finance conventions."""
    if value is None:
        return "N/A"
    # Pre-formatted string values (e.g. "< 1e-6", "0.451 vs 0.260") pass through
    if isinstance(value, str):
        return value
    name_lower = metric_name.lower()

    # Percent-formatted metrics (show as % if <=1 or raw if already scaled)
    if any(k in name_lower for k in ['accuracy', 'hit_rate', 'drawdown', 'return', 'agreement']):
        if isinstance(value, (int, float)):
            if value <= 1.0:
                return f"{value * 100:.1f}%"
            return f"{value:.1f}%"
        return f"{value}%"

    # 2-3 decimal ratio metrics
    if any(k in name_lower for k in ['sharpe', 'sortino', 'icir', 'ic', 'cramers_v']):
        return f"{value:.2f}"

    # Log-loss style metrics — 3 decimals
    if any(k in name_lower for k in ['log_loss', 'qlike']):
        return f"{value:.3f}"

    # Count metrics — integer
    if any(k in name_lower for k in ['tests', 'trades', 'windows', 'count', 'n_']):
        return f"{int(value):,}"

    # Dollar/PnL values
    if 'pnl' in name_lower:
        return f"${value:,.2f}"

    # Generic float fallback
    if isinstance(value, float):
        if abs(value) < 0.01:
            return f"{value:.4f}"
        return f"{value:.3f}"

    return f"{value}"
