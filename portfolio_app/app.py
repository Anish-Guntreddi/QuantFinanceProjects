"""
Quant Finance Portfolio - Streamlit App
Entry point: page config, CSS loading, session-state routing.
"""
import streamlit as st
from pathlib import Path

# Page config must be first Streamlit command
st.set_page_config(
    page_title="Quant Finance Portfolio",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load CSS
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'
if 'selected_project' not in st.session_state:
    st.session_state['selected_project'] = None

# Route based on session state
if st.session_state['page'] == 'project' and st.session_state['selected_project']:
    from views import project
    project.render()
else:
    from views import home
    home.render()
