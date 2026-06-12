"""Equity curve + drawdown chart using Plotly."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from .theme import ACCENT, PROFIT_GREEN, LOSS_RED, TEXT_MUTED, BORDER, plotly_layout


def render_equity_curve(
    results: dict[str, Any],
    title: str = "Equity Curve",
    height: int = 480,
    show_benchmark: bool = True,
) -> None:
    """Render a dual-axis equity curve (top) + drawdown (bottom) chart."""
    ec = results.get("equity_curve", {})
    dates = ec.get("dates", [])
    values = ec.get("values", [])
    bench = ec.get("benchmark_values", [])

    if not dates or not values:
        st.info("No equity curve data available.")
        return

    vals = np.array(values, dtype=float)
    drawdown = vals / np.maximum.accumulate(vals) - 1

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.04,
    )

    # Strategy equity
    fig.add_trace(
        go.Scatter(
            x=dates, y=values,
            name="Strategy",
            line=dict(color=ACCENT, width=2),
            hovertemplate="%{x}<br>Value: %{y:.4f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # Benchmark
    if show_benchmark and bench and len(bench) == len(dates):
        fig.add_trace(
            go.Scatter(
                x=dates, y=bench,
                name="Benchmark (SPY)",
                line=dict(color=TEXT_MUTED, width=1, dash="dot"),
                hovertemplate="%{x}<br>Benchmark: %{y:.4f}<extra></extra>",
            ),
            row=1, col=1,
        )

    # Drawdown
    fig.add_trace(
        go.Bar(
            x=dates, y=drawdown.tolist(),
            name="Drawdown",
            marker_color=LOSS_RED,
            opacity=0.5,
            hovertemplate="%{x}<br>DD: %{y:.2%}<extra></extra>",
            showlegend=False,
        ),
        row=2, col=1,
    )

    layout = plotly_layout(title=title, height=height)
    fig.update_layout(
        **{k: v for k, v in layout.items() if k not in ("xaxis", "yaxis", "height", "title", "legend")},
        height=height,
        title=layout["title"],
        legend=dict(
            x=0.01, y=0.99, xanchor="left", yanchor="top",
            font=dict(family="DM Sans, sans-serif", size=11, color=TEXT_MUTED),
            bgcolor="rgba(0,0,0,0)",
        ),
    )

    # Axis styling
    axis_common = dict(
        gridcolor="rgba(31,41,55,0.5)", gridwidth=1, griddash="dot",
        zerolinecolor=BORDER,
        tickfont=dict(family="JetBrains Mono, monospace", size=10, color=TEXT_MUTED),
    )
    fig.update_xaxes(**axis_common)
    fig.update_yaxes(**axis_common)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)

    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
