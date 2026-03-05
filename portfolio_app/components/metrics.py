"""KPI metric display components."""

import streamlit as st
from utils.formatting import fmt_pct, fmt_number


def render_metrics_row(metrics, columns=4):
    """Render a row of key metrics with large mono numbers."""
    if not metrics:
        return

    display_metrics = [
        ("Sharpe", metrics.get("sharpe_ratio"), False, ""),
        ("Sortino", metrics.get("sortino_ratio"), False, ""),
        ("Max DD", metrics.get("max_drawdown"), True, "pct"),
        ("Win Rate", metrics.get("win_rate"), False, "pct"),
        ("CAGR", metrics.get("cagr"), False, "pct"),
        ("Calmar", metrics.get("calmar_ratio"), False, ""),
        ("Total Return", metrics.get("total_return"), False, "pct"),
        ("Profit Factor", metrics.get("profit_factor"), False, ""),
    ]

    cols = st.columns(min(columns, len(display_metrics)))
    for i, (label, value, invert_color, fmt) in enumerate(display_metrics[:columns]):
        if value is None:
            continue
        with cols[i % columns]:
            if fmt == "pct":
                display_val = fmt_pct(value)
            else:
                display_val = fmt_number(value)

            is_positive = value >= 0 if not invert_color else value <= 0
            color = "#00D4AA" if is_positive else "#FF4757"

            html = f"""
            <div class="metric-panel">
                <div class="metric-value" style="color: {color};">{display_val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)


def render_metric_card(label, value, fmt="number", invert=False):
    """Render a single metric card."""
    if value is None:
        display_val = "—"
        color = "#6B7280"
    else:
        if fmt == "pct":
            display_val = fmt_pct(value)
        elif fmt == "currency":
            display_val = f"${value:,.0f}"
        else:
            display_val = fmt_number(value)
        is_positive = value >= 0 if not invert else value <= 0
        color = "#00D4AA" if is_positive else "#FF4757"

    html = f"""
    <div class="metric-panel">
        <div class="metric-value" style="color: {color};">{display_val}</div>
        <div class="metric-label">{label}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
