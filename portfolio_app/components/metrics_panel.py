"""Metrics panel — displays key performance metrics as large mono numbers."""

from __future__ import annotations

from typing import Any

import streamlit as st

# Metrics that should be displayed as percentages
_PCT_METRICS = {
    "total_return", "cagr", "annualized_vol", "max_drawdown",
    "win_rate", "avg_drawdown", "tracking_error",
}

# Display order and labels
_KEY_METRICS = [
    ("sharpe_ratio", "Sharpe"),
    ("sortino_ratio", "Sortino"),
    ("max_drawdown", "Max DD"),
    ("win_rate", "Win Rate"),
    ("cagr", "CAGR"),
    ("calmar_ratio", "Calmar"),
    ("profit_factor", "Profit Factor"),
    ("total_trades", "Trades"),
]


def _format_val(name: str, val: float) -> str:
    if name == "total_trades":
        return f"{int(val):,}"
    if name in _PCT_METRICS:
        return f"{val * 100:+.2f}%"
    return f"{val:.2f}"


def _color(name: str, val: float) -> str:
    if name in ("max_drawdown", "avg_drawdown"):
        return "metric-negative" if val < 0 else "metric-neutral"
    if name in ("sharpe_ratio", "sortino_ratio", "calmar_ratio", "cagr", "total_return"):
        return "metric-positive" if val > 0 else "metric-negative"
    if name == "win_rate":
        return "metric-positive" if val > 0.5 else "metric-negative"
    return "metric-neutral"


def render_metrics_panel(metrics: dict[str, Any], cols: int = 4) -> None:
    """Render a grid of key metrics as styled HTML."""
    items = []
    for key, label in _KEY_METRICS:
        if key in metrics:
            val = metrics[key]
            css = _color(key, val)
            formatted = _format_val(key, val)
            items.append(
                f'<div class="metric-card">'
                f'<div class="metric-label">{label}</div>'
                f'<div class="metric-value {css}">{formatted}</div>'
                f'</div>'
            )

    grid_css = f"display: grid; grid-template-columns: repeat({cols}, 1fr); gap: 0.75rem;"
    html = f'<div style="{grid_css}">{"".join(items)}</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_metrics_comparison(all_metrics: dict[str, dict[str, Any]], labels: list[str]) -> None:
    """Render a comparison table of metrics across multiple strategies."""
    header = "<tr><th style='text-align:left; padding:8px; border-bottom:1px solid #1f2937; font-family: DM Sans; color: #9ca3af;'>Metric</th>"
    for label in labels:
        header += f"<th style='text-align:right; padding:8px; border-bottom:1px solid #1f2937; font-family: DM Sans; color: #9ca3af;'>{label}</th>"
    header += "</tr>"

    rows = ""
    for key, display_label in _KEY_METRICS:
        row = f"<tr><td style='padding:8px; font-family: DM Sans; color: #e5e7eb; border-bottom:1px solid #111827;'>{display_label}</td>"
        for pid in all_metrics:
            m = all_metrics[pid]
            if key in m:
                val = m[key]
                css_color = "#10b981" if _color(key, val) == "metric-positive" else "#ef4444" if _color(key, val) == "metric-negative" else "#9ca3af"
                row += f"<td style='padding:8px; text-align:right; font-family: JetBrains Mono; color: {css_color}; border-bottom:1px solid #111827;'>{_format_val(key, val)}</td>"
            else:
                row += "<td style='padding:8px; text-align:right; color:#6b7280;'>—</td>"
        row += "</tr>"
        rows += row

    html = f"""
    <div style="overflow-x: auto;">
    <table style="width:100%; border-collapse:collapse; background:#111827; border-radius:6px;">
    {header}{rows}
    </table>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
