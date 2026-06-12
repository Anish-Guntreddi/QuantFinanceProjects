"""
Project Detail page — quant finance portfolio.
Tabs: Overview, Metrics & Results, Technical Details, Live Simulation, Compare.
"""
import streamlit as st
from pathlib import Path
from components.model_loader import (
    discover_projects, get_project_by_id, get_category_style,
    format_metric_value, PLOTLY_COLORS,
)
from components.metrics_display import (
    equity_curve_chart, comparison_table, bar_comparison_chart,
    cross_project_comparison_table,
)
from components.simulations import render_simulation


def render():
    projects = discover_projects()
    selected_id = st.session_state.get('selected_project', '01')

    project = get_project_by_id(selected_id)
    if not project:
        st.error(f"Project {selected_id} not found.")
        return

    card = project['card']
    results = project.get('results') or {}
    category = card.get('category', 'Other')
    style = get_category_style(category)

    # Back button
    if st.button("< Back to All Projects", key="back_btn"):
        st.session_state['page'] = 'home'
        st.session_state['selected_project'] = None
        st.rerun()

    # Header
    st.markdown(f"""
    <div style="margin-bottom: 24px;">
        <span class="badge badge-{style['css']}">{category}</span>
        <div class="hero-title" style="font-size: 2rem; margin-top: 8px;">{card.get('title', '')}</div>
        <div class="hero-subtitle">{card.get('short_description', '')}</div>
    </div>
    """, unsafe_allow_html=True)

    # Quick info row
    info_cols = st.columns(4)
    dataset = card.get('dataset', {})
    with info_cols[0]:
        st.markdown(f"**Dataset:** {dataset.get('name', 'N/A')}")
    with info_cols[1]:
        test_count = results.get('test_count', card.get('test_count', 'N/A'))
        st.markdown(f"**Tests:** {test_count}")
    with info_cols[2]:
        demo_type = card.get('demo_type', 'N/A')
        st.markdown(f"**Demo:** `{demo_type}`")
    with info_cols[3]:
        tags = card.get('tags', [])
        st.markdown(f"**Tags:** {', '.join(tags[:3]) if tags else 'N/A'}")

    st.markdown("---")

    # Tabbed layout
    tab_overview, tab_metrics, tab_tech, tab_sim, tab_compare = st.tabs([
        "Overview", "Metrics & Results", "Technical Details", "Live Simulation", "Compare"
    ])

    with tab_overview:
        _render_overview(results, card, project)

    with tab_metrics:
        _render_metrics_results(results, project)

    with tab_tech:
        _render_technical_details(card)

    with tab_sim:
        _render_simulation(card)

    with tab_compare:
        _render_compare(selected_id, projects)

    # Tags footer
    if tags:
        st.markdown("---")
        tag_html = " ".join(
            f'<span style="background:#1c1917;border:1px solid #292524;border-radius:2px;'
            f'padding:2px 8px;font-family:JetBrains Mono;font-size:0.75rem;color:#a8a29e;">{t}</span>'
            for t in tags
        )
        st.markdown(tag_html, unsafe_allow_html=True)


# ── Tab renderers ──────────────────────────────────────────────────────────

def _render_overview(results, card, project):
    """Overview: headline metrics, equity curve if present, summary + pipeline."""
    # Headline metrics row from results.yaml
    metrics = results.get('metrics', {})
    if metrics:
        st.markdown("#### Key Metrics")
        metric_cols = st.columns(min(len(metrics), 4))
        for i, (key, val) in enumerate(metrics.items()):
            with metric_cols[i % len(metric_cols)]:
                label = key.replace('_', ' ').title()
                st.metric(label, format_metric_value(val, key))

    # Equity curve from series
    series = results.get('series', {})
    if series:
        st.markdown("#### Equity / Return Series")
        fig = equity_curve_chart(series)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown("---")

    # Technical overview summary + pipeline
    overview = card.get('technical_overview', {})
    summary = overview.get('summary', '')
    if summary:
        st.markdown("#### Summary")
        st.markdown(summary)

    pipeline = overview.get('pipeline', [])
    if pipeline:
        st.markdown("**Pipeline**")
        for i, step in enumerate(pipeline, 1):
            if ':' in step:
                name, desc = step.split(':', 1)
                st.markdown(
                    f'<div style="margin:6px 0;padding:8px 12px;background:#1c1917;'
                    f'border-left:3px solid #2dd4bf;border-radius:2px;">'
                    f'<span style="font-family:JetBrains Mono;font-size:0.8rem;color:#2dd4bf;">{i}.</span> '
                    f'<span style="font-family:JetBrains Mono;font-size:0.8rem;color:#fafaf9;">{name.strip()}</span>'
                    f'<span style="font-family:Source Sans 3;font-size:0.85rem;color:#a8a29e;">:{desc}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div style="margin:6px 0;padding:8px 12px;background:#1c1917;'
                    f'border-left:3px solid #2dd4bf;border-radius:2px;">'
                    f'<span style="font-family:JetBrains Mono;font-size:0.8rem;color:#2dd4bf;">{i}.</span> '
                    f'<span style="font-family:Source Sans 3;font-size:0.85rem;color:#e7e5e4;">{step}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )


def _render_metrics_results(results, project):
    """Metrics & Results: tables from results.yaml + any figure paths."""
    if not results:
        st.info("No results available for this project.")
        return

    # All scalar metrics
    metrics = results.get('metrics', {})
    if metrics:
        st.markdown("#### All Metrics")
        metric_cols = st.columns(min(len(metrics), 4))
        for i, (key, val) in enumerate(metrics.items()):
            with metric_cols[i % len(metric_cols)]:
                st.metric(key.replace('_', ' ').title(), format_metric_value(val, key))

    # Tables
    tables = results.get('tables', {})
    if tables:
        for table_key, table_data in tables.items():
            st.markdown(f"#### {table_key.replace('_', ' ').title()}")
            html = comparison_table(table_data)
            if html:
                st.markdown(html, unsafe_allow_html=True)
            st.markdown("")

    # Figures: list of repo-relative PNG paths
    figures = results.get('figures', [])
    if figures:
        st.markdown("#### Figures")
        from components.model_loader import REPO_ROOT
        for fig_path in figures:
            full_path = REPO_ROOT / fig_path
            if full_path.exists():
                caption = Path(fig_path).stem.replace('_', ' ').title()
                st.image(str(full_path), caption=caption, use_container_width=True)
            else:
                st.caption(f"Figure not found: {fig_path}")


def _render_technical_details(card):
    """Technical Details: technical_overview, methodology/key_techniques,
    correctness_guarantees (checklist), dataset/dgp."""

    # Technical overview
    overview = card.get('technical_overview', {})
    if overview:
        st.markdown("#### Technical Overview")
        summary = overview.get('summary', '')
        if summary:
            st.markdown(summary)

        pipeline = overview.get('pipeline', [])
        if pipeline:
            st.markdown("")
            st.markdown("**Pipeline**")
            for i, step in enumerate(pipeline, 1):
                if ':' in step:
                    name, desc = step.split(':', 1)
                    st.markdown(
                        f'<div style="margin:6px 0;padding:8px 12px;background:#1c1917;'
                        f'border-left:3px solid #2dd4bf;border-radius:2px;">'
                        f'<span style="font-family:JetBrains Mono;font-size:0.8rem;color:#2dd4bf;">{i}.</span> '
                        f'<span style="font-family:JetBrains Mono;font-size:0.8rem;color:#fafaf9;">{name.strip()}</span>'
                        f'<span style="font-family:Source Sans 3;font-size:0.85rem;color:#a8a29e;">:{desc}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div style="margin:6px 0;padding:8px 12px;background:#1c1917;'
                        f'border-left:3px solid #2dd4bf;border-radius:2px;">'
                        f'<span style="font-family:JetBrains Mono;font-size:0.8rem;color:#2dd4bf;">{i}.</span> '
                        f'<span style="font-family:Source Sans 3;font-size:0.85rem;color:#e7e5e4;">{step}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
        st.markdown("---")

    # Methodology
    methodology = card.get('methodology', {})
    if methodology:
        st.markdown("#### Methodology")
        summary = methodology.get('summary', '')
        if summary:
            st.markdown(summary)
        techniques = methodology.get('key_techniques', [])
        if techniques:
            st.markdown("**Key Techniques:**")
            tech_html = " ".join(
                f'<span style="background:#042f2e;border:1px solid #0f766e;border-radius:2px;'
                f'padding:2px 8px;font-family:JetBrains Mono;font-size:0.75rem;color:#2dd4bf;'
                f'margin-right:4px;">{t}</span>'
                for t in techniques
            )
            st.markdown(tech_html, unsafe_allow_html=True)
        st.markdown("---")

    # Correctness guarantees — rendered as a checklist
    guarantees = card.get('correctness_guarantees', [])
    if guarantees:
        st.markdown("#### Correctness Guarantees")
        for g in guarantees:
            st.markdown(
                f'<div style="margin:4px 0;padding:6px 12px;background:#1c1917;'
                f'border-radius:2px;font-family:Source Sans 3;font-size:0.9rem;color:#e7e5e4;">'
                f'<span style="color:#34d399;font-weight:700;margin-right:8px;">&#10003;</span>{g}'
                f'</div>',
                unsafe_allow_html=True
            )
        st.markdown("---")

    # Dataset / DGP
    dataset = card.get('dataset', {})
    if dataset:
        st.markdown("#### Dataset / DGP")
        cols = st.columns(2)
        with cols[0]:
            st.markdown(f"**Name:** {dataset.get('name', 'N/A')}")
        with cols[1]:
            st.markdown(f"**Description:** {dataset.get('description', 'N/A')}")


def _render_simulation(card):
    """Live Simulation tab — dispatches to simulations.py by demo_type."""
    demo_type = card.get('demo_type', '')
    if not demo_type:
        st.info("No interactive simulation defined for this project (demo_type not set in model_card.yaml).")
        return
    render_simulation(demo_type)


def _render_compare(current_id, projects):
    """Compare tab — cross-project headline metric table."""
    st.markdown("#### Compare Projects")
    st.caption("Headline metric comparison across all projects in the portfolio.")

    rows = []
    for p in projects:
        pcard = p['card']
        presults = p.get('results') or {}
        pid = pcard.get('project_id', p['dir'][:2])
        headline = pcard.get('headline_metric', {})
        hl_name = headline.get('name', '-')
        hl_val = headline.get('value')
        hl_formatted = format_metric_value(hl_val, hl_name) if hl_val is not None else '-'
        test_count = presults.get('test_count', '-')
        rows.append({
            'project': pid,
            'title': pcard.get('title', p['dir']),
            'category': pcard.get('category', '-'),
            'tests': test_count,
            hl_name: hl_formatted,
        })

    if rows:
        html = cross_project_comparison_table(rows)
        if html:
            st.markdown(html, unsafe_allow_html=True)

    # Current project highlight
    project = get_project_by_id(current_id)
    if not project:
        return

    st.markdown("---")
    st.markdown("#### Current Project Metrics")
    card = project['card']
    results = project.get('results') or {}
    metrics = results.get('metrics', {})
    if metrics:
        metric_cols = st.columns(min(len(metrics), 4))
        for i, (key, val) in enumerate(metrics.items()):
            with metric_cols[i % len(metric_cols)]:
                st.metric(key.replace('_', ' ').title(), format_metric_value(val, key))
    else:
        headline = card.get('headline_metric', {})
        if headline:
            st.metric(
                headline.get('name', 'Metric').replace('_', ' ').title(),
                format_metric_value(headline.get('value'), headline.get('name', '')),
            )
