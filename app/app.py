"""Streamlit application shell and navigation (Deliverable D6.2)."""

import streamlit as st

st.set_page_config(
    page_title="Tendency Tax Explorer",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Navigation definition with st.Page and st.navigation
pages = [
    st.Page(
        "views/test_panel.py",
        title="Test Panel",
        icon="⚖️",
        default=True,
    ),
    st.Page(
        "views/situation_explorer.py",
        title="Situation Explorer",
        icon="🗺️",
    ),
    st.Page(
        "views/team_scout.py",
        title="Team Scout",
        icon="🎯",
    ),
    st.Page(
        "views/about_method.py",
        title="About the Method",
        icon="📖",
    ),
    st.Page(
        "views/style_guide.py",
        title="Design System & Style Guide",
        icon="🎨",
    ),
]

nav = st.navigation(pages)
nav.run()
