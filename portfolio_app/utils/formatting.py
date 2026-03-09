"""Display formatting helpers for metrics, percentages, and numbers."""

from __future__ import annotations


def format_pct(value: float, decimals: int = 2) -> str:
    return f"{value * 100:+.{decimals}f}%"


def format_number(value: float, decimals: int = 2) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.{decimals}f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.{decimals}f}K"
    return f"{value:.{decimals}f}"


def format_metric(name: str, value: float) -> str:
    """Format a metric value based on its name convention."""
    pct_metrics = {
        "total_return", "cagr", "annualized_vol", "max_drawdown",
        "win_rate", "avg_drawdown", "tracking_error",
        "implementation_shortfall", "vwap_slippage",
    }
    if name in pct_metrics:
        return format_pct(value)
    return f"{value:.4f}" if abs(value) < 10 else f"{value:.2f}"


def color_class(value: float) -> str:
    """Return CSS class name for profit/loss coloring."""
    if value > 0:
        return "metric-positive"
    if value < 0:
        return "metric-negative"
    return "metric-neutral"


def metric_html(label: str, value: float, name: str = "") -> str:
    """Render a single metric as styled HTML."""
    css = color_class(value)
    formatted = format_metric(name or label.lower().replace(" ", "_"), value)
    return (
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value {css}">{formatted}</div>'
        f'</div>'
    )
