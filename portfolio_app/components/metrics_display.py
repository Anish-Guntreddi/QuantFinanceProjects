"""
Plotly chart components for quant finance metrics visualization.
Ported from ML app — identical layout helpers, quant-specific chart functions replacing
training curves / per-class heatmap.
"""
import plotly.graph_objects as go
import plotly.express as px
from components.model_loader import PLOTLY_COLORS


def plotly_layout(title="", height=400):
    """Standard Plotly layout matching portfolio theme."""
    return go.Layout(
        title=dict(text=title, font=dict(family="JetBrains Mono", size=14, color="#fafaf9")),
        font=dict(family="Source Sans 3", size=12, color="#a8a29e"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=50, r=20, t=40, b=40),
        showlegend=True,
        legend=dict(font=dict(size=11)),
    )


def metric_row(metrics_dict, format_fn=None):
    """
    Render a row of st.metric cards from a dict.
    format_fn(value, key) -> str; falls back to str() if not provided.
    """
    import streamlit as st
    if not metrics_dict:
        return
    cols = st.columns(min(len(metrics_dict), 4))
    for i, (key, val) in enumerate(metrics_dict.items()):
        with cols[i % len(cols)]:
            label = key.replace('_', ' ').title()
            value_str = format_fn(val, key) if format_fn else str(val)
            st.metric(label, value_str)


def headline_metric(name, value, format_fn=None):
    """Render a single large st.metric (headline card)."""
    import streamlit as st
    value_str = format_fn(value, name) if format_fn else str(value)
    st.metric(name.replace('_', ' ').title(), value_str)


# ── Quant chart helpers ────────────────────────────────────────────────────


def equity_curve_chart(series_dict, height=400):
    """
    Plot one or more named equity / cumulative-return series.
    series_dict: {name: [float, ...], ...}
    Uses portfolio palette + JetBrains Mono fonts, transparent bg.
    """
    if not series_dict:
        return None

    fig = go.Figure(layout=plotly_layout("Equity Curves", height=height))

    for i, (name, values) in enumerate(series_dict.items()):
        x = list(range(len(values)))
        color = PLOTLY_COLORS[i % len(PLOTLY_COLORS)]
        dash = 'dash' if i > 0 else 'solid'
        fig.add_trace(go.Scatter(
            x=x, y=values,
            name=name, mode='lines',
            line=dict(color=color, width=2, dash=dash),
        ))

    fig.update_xaxes(title_text="Bar", gridcolor="#292524", gridwidth=1)
    fig.update_yaxes(title_text="Cumulative Return", gridcolor="#292524", gridwidth=1)
    return fig


def comparison_table(table_dict):
    """
    Render a {title, columns: [...], rows: [[...]]} dict as a styled HTML table.
    Reuses ML app's model_comparison_table styling verbatim.
    """
    if not table_dict:
        return None

    columns = table_dict.get('columns', [])
    rows = table_dict.get('rows', [])

    if not columns:
        return None

    header = ''.join(
        f'<th style="padding:8px 12px;text-align:left;border-bottom:2px solid #2dd4bf;'
        f'font-family:JetBrains Mono;font-size:0.8rem;color:#fafaf9;">'
        f'{str(c).replace("_", " ").title()}</th>'
        for c in columns
    )

    rows_html = ''
    for row in rows:
        cells = ''
        for val in row:
            if isinstance(val, float):
                val = f'{val:.4f}' if abs(val) < 1 else f'{val:.2f}'
            cells += (
                f'<td style="padding:6px 12px;border-bottom:1px solid #292524;'
                f'font-family:Source Sans 3;font-size:0.85rem;color:#e7e5e4;">{val}</td>'
            )
        rows_html += f'<tr>{cells}</tr>'

    title = table_dict.get('title', '')
    title_html = (
        f'<div style="font-family:JetBrains Mono;font-size:0.9rem;color:#fafaf9;'
        f'margin-bottom:8px;">{title}</div>'
        if title else ''
    )

    return (
        f'{title_html}'
        f'<table style="width:100%;border-collapse:collapse;margin:8px 0;">'
        f'<thead><tr>{header}</tr></thead><tbody>{rows_html}</tbody></table>'
    )


def bar_comparison_chart(labels, values, title="Comparison", height=300):
    """
    Plotly bar chart using portfolio palette.
    labels: [str], values: [float]
    """
    if not labels or not values:
        return None

    colors = [PLOTLY_COLORS[i % len(PLOTLY_COLORS)] for i in range(len(labels))]

    fig = go.Figure(layout=plotly_layout(title, height=height))
    fig.add_trace(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"{v:.3f}" for v in values],
        textposition='outside',
        textfont=dict(family="JetBrains Mono", size=11),
    ))
    fig.update_xaxes(gridcolor="#292524")
    fig.update_yaxes(gridcolor="#292524")
    fig.update_layout(showlegend=False)
    return fig


# ── Cross-project comparison table (HTML, mirrors ML app) ─────────────────


def cross_project_comparison_table(projects_data):
    """
    Build an HTML comparison table from a list of dicts:
    [{project, category, tests, headline_name, headline_value}, ...]
    """
    if not projects_data:
        return None

    columns = list(projects_data[0].keys())
    header = ''.join(
        f'<th style="padding:8px 12px;text-align:left;font-family:JetBrains Mono;'
        f'font-size:0.8rem;border-bottom:2px solid #2dd4bf;color:#fafaf9;">{c.replace("_"," ").title()}</th>'
        for c in columns
    )

    rows_html = ''
    for row in projects_data:
        cells = ''
        for k, val in row.items():
            cells += (
                f'<td style="padding:8px 12px;font-family:Source Sans 3;font-size:0.85rem;'
                f'border-bottom:1px solid #292524;color:#e7e5e4;">{val}</td>'
            )
        rows_html += f'<tr>{cells}</tr>'

    return (
        f'<table style="width:100%;border-collapse:collapse;margin-top:16px;">'
        f'<thead><tr>{header}</tr></thead><tbody>{rows_html}</tbody></table>'
    )
