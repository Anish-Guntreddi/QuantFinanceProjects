"""
Reusable project card component with category accent.
Ported from ML app — identical HTML/CSS classes, quant metric formatting.
"""
import streamlit as st
from components.model_loader import get_category_style, format_metric_value


def render_project_card(project, index=0):
    """Render a single project card. Returns True if clicked."""
    card = project['card']
    results = project.get('results')
    category = card.get('category', 'Other')
    style = get_category_style(category)

    # Get headline metric from card schema
    headline = card.get('headline_metric', {})
    metric_value = ""
    metric_label = ""
    if headline:
        metric_value = format_metric_value(headline.get('value'), headline.get('name', ''))
        metric_label = headline.get('name', '').replace('_', ' ').title()
    elif results and 'metrics' in results:
        metrics = results['metrics']
        key, val = next(iter(metrics.items()))
        metric_value = format_metric_value(val, key)
        metric_label = key.replace('_', ' ').title()

    project_id = card.get('project_id', project['dir'][:2])

    st.markdown(f"""
    <div class="project-card {style['css']} fade-in fade-in-{index + 1}" onclick="void(0)">
        <div class="card-number">PROJECT {project_id}</div>
        <div class="card-title">{card.get('title', project['dir'])}</div>
        <div class="card-description">{card.get('short_description', '')}</div>
        <div style="display: flex; justify-content: space-between; align-items: flex-end;">
            <div>
                <div class="card-metric">{metric_value}</div>
                <div class="card-metric-label">{metric_label}</div>
            </div>
            <span class="badge badge-{style['css']}">{category}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    return st.button(f"View Project {project_id}", key=f"card_{project_id}", use_container_width=True)
