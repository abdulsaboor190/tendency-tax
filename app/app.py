"""Streamlit application shell and navigation (Deliverable D6.2)."""

import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path and remove app/ directory to prevent
# Streamlit from resolving 'app.py' as the top-level 'app' module.
_repo_root = str(Path(__file__).resolve().parent.parent)
_app_dir = str(Path(__file__).resolve().parent)

sys.path = [p for p in sys.path if Path(p).resolve() != Path(_app_dir).resolve()]
if not any(Path(p).resolve() == Path(_repo_root).resolve() for p in sys.path):
    sys.path.insert(0, _repo_root)

# If 'app' was mistakenly imported as a standalone module from app.py, clear it
if "app" in sys.modules and not hasattr(sys.modules["app"], "__path__"):
    del sys.modules["app"]

import streamlit as st  # noqa: E402

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
]

# Style guide is only exposed during development mode (TENDENCY_DEV=1)
if os.environ.get("TENDENCY_DEV") == "1":
    pages.append(
        st.Page(
            "views/style_guide.py",
            title="Design System & Style Guide",
            icon="🎨",
        )
    )

nav = st.navigation(pages)
nav.run()
