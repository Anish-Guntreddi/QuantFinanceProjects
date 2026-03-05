"""Number formatting and color helpers."""


def fmt_pct(value, decimals=2):
    """Format as percentage."""
    if value is None:
        return "—"
    return f"{value * 100:,.{decimals}f}%"


def fmt_number(value, decimals=2):
    """Format a number."""
    if value is None:
        return "—"
    return f"{value:,.{decimals}f}"


def fmt_currency(value, decimals=0):
    """Format as currency."""
    if value is None:
        return "—"
    return f"${value:,.{decimals}f}"


def metric_color(value, positive_is_good=True):
    """Return CSS color class for a metric value."""
    if value is None:
        return ""
    if positive_is_good:
        return "metric-positive" if value >= 0 else "metric-negative"
    return "metric-negative" if value >= 0 else "metric-positive"


def sharpe_color(sharpe):
    """Return color for Sharpe ratio."""
    if sharpe is None:
        return "#6B7280"
    if sharpe >= 2.0:
        return "#00D4AA"
    elif sharpe >= 1.0:
        return "#1E90FF"
    elif sharpe >= 0.5:
        return "#FFA502"
    else:
        return "#FF4757"
