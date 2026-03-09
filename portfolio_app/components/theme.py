"""Centralized theme constants and Plotly layout defaults."""

from __future__ import annotations

CATEGORY_COLORS: dict[str, str] = {
    "HFT_strategy_projects": "#f59e0b",
    "ai_ml_trading": "#3b82f6",
    "core_research_backtesting": "#8b5cf6",
    "market_microstructure_engines": "#06b6d4",
    "market_microstructure_execution": "#14b8a6",
    "research_intraday_strategies": "#22c55e",
    "risk_engineering": "#f43f5e",
}

VIZ_COLORS = [
    "#f59e0b", "#3b82f6", "#10b981", "#f43f5e",
    "#8b5cf6", "#06b6d4", "#ec4899",
]

PROFIT_GREEN = "#10b981"
LOSS_RED = "#ef4444"

BG_PRIMARY = "#0a0e17"
BG_CARD = "#111827"
TEXT_PRIMARY = "#e5e7eb"
TEXT_SECONDARY = "#9ca3af"
TEXT_MUTED = "#6b7280"
BORDER = "#1f2937"
ACCENT = "#f59e0b"


def chart_colors(n: int = 7) -> list[str]:
    """Return first *n* visualization colors."""
    return (VIZ_COLORS * ((n // len(VIZ_COLORS)) + 1))[:n]


def plotly_layout(title: str = "", height: int = 400, **overrides) -> dict:
    """Standard Plotly layout matching the Signal Lab theme."""
    layout = dict(
        title=dict(
            text=title,
            font=dict(family="DM Serif Display, Georgia, serif", size=16, color=TEXT_PRIMARY),
            x=0, xanchor="left",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", size=12, color=TEXT_SECONDARY),
        height=height,
        margin=dict(l=50, r=20, t=50, b=40),
        legend=dict(
            font=dict(family="DM Sans, sans-serif", size=11, color=TEXT_SECONDARY),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
        xaxis=dict(
            gridcolor="rgba(31,41,55,0.5)",
            gridwidth=1,
            griddash="dot",
            zerolinecolor=BORDER,
            tickfont=dict(family="JetBrains Mono, monospace", size=10, color=TEXT_MUTED),
        ),
        yaxis=dict(
            gridcolor="rgba(31,41,55,0.5)",
            gridwidth=1,
            griddash="dot",
            zerolinecolor=BORDER,
            tickfont=dict(family="JetBrains Mono, monospace", size=10, color=TEXT_MUTED),
        ),
        hoverlabel=dict(
            bgcolor=BG_CARD,
            font=dict(family="JetBrains Mono, monospace", size=11, color=TEXT_PRIMARY),
            bordercolor=BORDER,
        ),
    )
    layout.update(overrides)
    return layout
