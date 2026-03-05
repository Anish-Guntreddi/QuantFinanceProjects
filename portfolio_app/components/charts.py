"""Plotly chart builders with consistent dark theme."""

import plotly.graph_objects as go
import numpy as np
from components.theme import get_plotly_layout, PROFIT_COLOR, LOSS_COLOR


def equity_curve_chart(dates, equity, benchmark=None, title="Equity Curve"):
    """Dual-axis equity curve with drawdown."""
    equity_arr = np.array(equity)
    peak = np.maximum.accumulate(equity_arr)
    drawdown = (equity_arr / peak - 1) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=equity, name="Strategy", line=dict(color=PROFIT_COLOR, width=2)))
    if benchmark:
        fig.add_trace(go.Scatter(x=dates, y=benchmark, name="Benchmark", line=dict(color="#6B7280", width=1, dash="dot")))

    fig.update_layout(**get_plotly_layout(
        title=title, height=400, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    ))
    return fig


def drawdown_chart(dates, equity, title="Drawdown"):
    """Drawdown area chart."""
    equity_arr = np.array(equity)
    peak = np.maximum.accumulate(equity_arr)
    drawdown = (equity_arr / peak - 1) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=drawdown, fill="tozeroy",
                             fillcolor="rgba(255, 71, 87, 0.3)", line=dict(color=LOSS_COLOR, width=1), name="Drawdown"))
    fig.update_layout(**get_plotly_layout(title=title, height=200, yaxis_title="Drawdown %"))
    return fig


def monthly_heatmap(monthly_returns, title="Monthly Returns (%)"):
    """Monthly returns heatmap."""
    import pandas as pd
    if not monthly_returns:
        return go.Figure()

    data = []
    for k, v in monthly_returns.items():
        parts = k.split("-")
        data.append({"year": int(parts[0]), "month": int(parts[1]), "ret": v * 100})

    df = pd.DataFrame(data)
    pivot = df.pivot_table(values="ret", index="year", columns="month", aggfunc="first")
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=[months[m - 1] for m in pivot.columns], y=[str(y) for y in pivot.index],
        colorscale=[[0, LOSS_COLOR], [0.5, "#1A1F2E"], [1, PROFIT_COLOR]],
        zmid=0, zmin=-5, zmax=5,
        text=np.round(pivot.values, 1), texttemplate="%{text:.1f}",
        textfont={"size": 10, "family": "JetBrains Mono"},
        hovertemplate="Month: %{x}<br>Year: %{y}<br>Return: %{z:.2f}%<extra></extra>",
    ))
    fig.update_layout(**get_plotly_layout(title=title, height=max(200, len(pivot.index) * 40 + 100)))
    return fig


def sensitivity_chart(sensitivity_data, title="Parameter Sensitivity"):
    """Parameter sensitivity line charts."""
    if not sensitivity_data:
        return go.Figure()

    s = sensitivity_data[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s["values"], y=s["sharpe"], name="Sharpe Ratio",
                             line=dict(color=PROFIT_COLOR, width=2), mode="lines+markers"))
    fig.update_layout(**get_plotly_layout(
        title=f"{title}: {s['param']}", height=300,
        xaxis_title=s["param"], yaxis_title="Sharpe Ratio",
    ))
    return fig


def radar_chart(labels, values_dict, title="Strategy Comparison"):
    """Radar chart for comparing strategies."""
    fig = go.Figure()
    colors = ["#00D4AA", "#FF6B35", "#7B68EE", "#1E90FF", "#FFA502", "#FF4757"]
    for i, (name, vals) in enumerate(values_dict.items()):
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=labels + [labels[0]],
            fill="toself", fillcolor=f"rgba({','.join(str(int(colors[i % len(colors)][j:j+2], 16)) for j in (1,3,5))}, 0.1)",
            line=dict(color=colors[i % len(colors)], width=2), name=name,
        ))
    fig.update_layout(**get_plotly_layout(
        title=title, height=450,
        polar=dict(bgcolor="#1A1F2E", radialaxis=dict(gridcolor="#2D3748", linecolor="#2D3748"),
                   angularaxis=dict(gridcolor="#2D3748", linecolor="#2D3748")),
    ))
    return fig
