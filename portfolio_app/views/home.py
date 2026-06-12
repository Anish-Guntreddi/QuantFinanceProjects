"""
Home page — quant finance portfolio.
Hero with typing animation (identical structure to ML app), stat cards, category filter, project grid.
"""
import streamlit as st
import streamlit.components.v1 as components
from components.model_loader import discover_projects, get_category_style, format_metric_value


def render():
    # Hero with typing animation — identical structure to ML app, quant text
    components.html("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Source+Sans+3:wght@300;400;500;600;700&display=swap');
        body { margin: 0; background: transparent; }
        .hero-name {
            font-family: 'Source Sans 3', sans-serif;
            font-size: 2.8rem;
            font-weight: 700;
            color: #fafaf9;
            text-align: center;
            margin-bottom: 4px;
            letter-spacing: -0.02em;
        }
        .typing-container {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 2rem;
            margin-top: 4px;
        }
        .typing-text {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.15rem;
            color: #2dd4bf;
            font-weight: 500;
        }
        .typing-cursor {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.15rem;
            color: #2dd4bf;
            animation: blink 0.7s infinite;
            margin-left: 1px;
        }
        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0; }
        }
        .hero-sub {
            font-family: 'Source Sans 3', sans-serif;
            font-size: 1.05rem;
            color: #a8a29e;
            text-align: center;
            margin-top: 16px;
            max-width: 640px;
            margin-left: auto;
            margin-right: auto;
        }
    </style>
    <div>
        <div class="hero-name">Anish Guntreddi</div>
        <div class="typing-container">
            <span class="typing-text"></span><span class="typing-cursor">|</span>
        </div>
        <div class="hero-sub">5 research-grade quantitative finance systems spanning
        backtesting, ML alpha, macro regime detection, volatility surfaces, and crypto &mdash;
        397 tests &middot; leakage-audited.</div>
    </div>
    <script>
    const titles = [
        "Quantitative Research Engineering",
        "Event-Driven Backtesting",
        "Regime Detection",
        "Volatility Surfaces",
        "Systematic Alpha"
    ];
    let titleIndex = 0;
    let charIndex = 0;
    let isDeleting = false;
    function typeEffect() {
        const el = document.querySelector('.typing-text');
        if (!el) { setTimeout(typeEffect, 200); return; }
        const current = titles[titleIndex];
        if (!isDeleting) {
            el.textContent = current.substring(0, charIndex + 1);
            charIndex++;
            if (charIndex === current.length) {
                isDeleting = true;
                setTimeout(typeEffect, 1800);
                return;
            }
            setTimeout(typeEffect, 80);
        } else {
            el.textContent = current.substring(0, charIndex - 1);
            charIndex--;
            if (charIndex === 0) {
                isDeleting = false;
                titleIndex = (titleIndex + 1) % titles.length;
                setTimeout(typeEffect, 400);
                return;
            }
            setTimeout(typeEffect, 40);
        }
    }
    typeEffect();
    </script>
    """, height=180)

    st.markdown("")

    # Headline stat cards (4 cards)
    stat_cols = st.columns(4)
    stats = [
        ("397", "Tests passing"),
        ("< 1e-6", "IV round-trip error"),
        ("4", "Leakage oracle classes"),
        ("5", "Research systems"),
    ]
    for i, (val, label) in enumerate(stats):
        with stat_cols[i]:
            st.markdown(f"""
            <div class="project-card fade-in fade-in-{i+1}" style="text-align:center;padding:20px 16px;">
                <div class="card-metric" style="font-size:1.6rem;">{val}</div>
                <div class="card-metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    projects = discover_projects()

    if not projects:
        st.warning(
            "No projects found. Ensure model_card.yaml files exist in "
            "portfolio_projects/<project-dir>/model_card.yaml."
        )
        return

    # Category filter pills
    categories = sorted(set(p['card'].get('category', 'Other') for p in projects))
    all_categories = ["All"] + categories

    selected = st.pills("Filter by category", all_categories, default="All", key="cat_filter")
    if not selected:
        selected = "All"

    st.markdown("")

    # Filter projects
    filtered = projects if selected == "All" else [
        p for p in projects if p['card'].get('category') == selected
    ]

    # Render card grid (3 columns)
    cols = st.columns(3)
    for i, project in enumerate(filtered):
        card = project['card']
        results = project.get('results')
        category = card.get('category', 'Other')
        style = get_category_style(category)
        project_id = card.get('project_id', project['dir'][:2])

        # Headline metric: prefer card['headline_metric'], fall back to results.yaml
        headline = card.get('headline_metric', {})
        metric_value = ""
        metric_label = ""
        if headline:
            metric_value = format_metric_value(headline.get('value'), headline.get('name', ''))
            metric_label = headline.get('name', '').replace('_', ' ').title()
        elif results and 'metrics' in results:
            metrics = results['metrics']
            if metrics:
                key, val = next(iter(metrics.items()))
                metric_value = format_metric_value(val, key)
                metric_label = key.replace('_', ' ').title()

        with cols[i % 3]:
            st.markdown(f"""
            <div class="project-card {style['css']} fade-in fade-in-{i + 1}">
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

            if st.button("View Details", key=f"home_card_{project_id}", use_container_width=True):
                st.session_state['selected_project'] = project_id
                st.session_state['page'] = 'project'
                st.rerun()
