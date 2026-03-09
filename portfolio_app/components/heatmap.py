"""Monthly returns heatmap using Plotly."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from .theme import plotly_layout, TEXT_MUTED


def render_monthly_heatmap(
    monthly_returns: dict[str, float],
    title: str = "Monthly Returns",
    height: int = 300,
) -> None:
    """Render a years x months heatmap of monthly returns."""
    if not monthly_returns:
        st.info("No monthly returns data available.")
        return

    # Parse into year/month grid
    years: set[int] = set()
    for key in monthly_returns:
        years.add(int(key[:4]))
    years_sorted = sorted(years)
    months = list(range(1, 13))
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    z = []
    for year in years_sorted:
        row = []
        for month in months:
            key = f"{year}-{month:02d}"
            row.append(monthly_returns.get(key, np.nan))
        z.append(row)

    # Custom text for hover
    text = []
    for i, year in enumerate(years_sorted):
        row = []
        for j, month in enumerate(months):
            val = z[i][j]
            if np.isnan(val):
                row.append("")
            else:
                row.append(f"{val * 100:+.2f}%")
        text.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=month_labels,
        y=[str(y) for y in years_sorted],
        text=text,
        texttemplate="%{text}",
        textfont=dict(family="JetBrains Mono, monospace", size=10),
        colorscale=[
            [0.0, "#ef4444"],
            [0.5, "#1f2937"],
            [1.0, "#10b981"],
        ],
        zmid=0,
        showscale=False,
        hovertemplate="%{y} %{x}: %{text}<extra></extra>",
    ))

    layout = plotly_layout(title=title, height=height)
    fig.update_layout(**layout)
    fig.update_yaxes(autorange="reversed")

    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
