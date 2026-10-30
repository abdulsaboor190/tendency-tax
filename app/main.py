"""Streamlit entry point for the Tendency Tax interactive application."""

import streamlit as st

from app.ui.theme import inject_css
from app.views.style_guide import render_style_guide

st.set_page_config(
    page_title="Tendency Tax Explorer",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

# Multi-page / views routing
page = st.sidebar.radio(
    "Navigation",
    ["Style Guide"],
    index=0,
)

if page == "Style Guide":
    render_style_guide()
