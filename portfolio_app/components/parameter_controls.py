"""Dynamic parameter controls built from strategy_card.json interactive_params."""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_controls(interactive_params: list[dict[str, Any]], key_prefix: str = "") -> dict[str, Any]:
    """Render Streamlit widgets for each interactive param and return current values."""
    params: dict[str, Any] = {}
    for p in interactive_params:
        name = p["name"]
        label = p.get("label", name)
        widget_key = f"{key_prefix}{name}"
        ptype = p.get("type", "slider")

        if ptype == "slider":
            params[name] = st.slider(
                label,
                min_value=float(p["min"]),
                max_value=float(p["max"]),
                value=float(p["default"]),
                step=float(p.get("step", (p["max"] - p["min"]) / 20)),
                key=widget_key,
            )
        elif ptype == "selectbox":
            options = p.get("options", [])
            default_idx = options.index(p["default"]) if p["default"] in options else 0
            params[name] = st.selectbox(label, options, index=default_idx, key=widget_key)
        elif ptype == "number_input":
            params[name] = st.number_input(
                label,
                min_value=float(p["min"]),
                max_value=float(p["max"]),
                value=float(p["default"]),
                step=float(p.get("step", 1)),
                key=widget_key,
            )
    return params


def render_sensitivity_chart(sensitivity: list[dict[str, Any]]) -> None:
    """Render parameter sensitivity charts from results.json data."""
    import plotly.graph_objects as go
    from .theme import plotly_layout, ACCENT, LOSS_RED

    if not sensitivity:
        return

    for s in sensitivity:
        param_name = s.get("param", "parameter")
        values = s.get("values", [])
        sharpes = s.get("sharpe", [])
        drawdowns = s.get("max_drawdowns", [])

        if not values or not sharpes:
            continue

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=values, y=sharpes,
            name="Sharpe",
            line=dict(color=ACCENT, width=2),
            mode="lines+markers",
            marker=dict(size=5),
        ))

        if drawdowns:
            fig.add_trace(go.Scatter(
                x=values, y=drawdowns,
                name="Max DD",
                line=dict(color=LOSS_RED, width=1.5, dash="dash"),
                mode="lines+markers",
                marker=dict(size=4),
                yaxis="y2",
            ))

        layout = plotly_layout(
            title=f"Sensitivity: {param_name}",
            height=300,
        )
        layout["xaxis"]["title"] = dict(text=param_name, font=dict(size=12, color="#9ca3af"))
        layout["yaxis"]["title"] = dict(text="Sharpe Ratio", font=dict(size=12, color="#9ca3af"))

        if drawdowns:
            layout["yaxis2"] = dict(
                title=dict(text="Max Drawdown", font=dict(size=12, color="#9ca3af")),
                overlaying="y", side="right",
                tickformat=".0%",
                gridcolor="rgba(0,0,0,0)",
                tickfont=dict(family="JetBrains Mono, monospace", size=10, color="#6b7280"),
            )

        fig.update_layout(**layout)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
