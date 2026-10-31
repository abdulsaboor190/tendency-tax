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
    page_title="Tendency Tax",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar brand header and attribution
st.sidebar.markdown(
    """
    <div style="margin-bottom: 1.25rem;">
        <h2 style="margin: 0; font-size: 1.25rem; font-weight: 700;">Tendency Tax</h2>
        <p style="margin: 0.2rem 0 0 0; font-size: 0.82rem; "
           "color: var(--color-text-muted); line-height: 1.35;">
            Does being predictable cost an NFL offense? A pre-declared test on "
            "held-out 2024–2025 seasons.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Navigation definition with st.Page and st.navigation using Material icons
pages = [
    st.Page(
        "views/test_panel.py",
        title="Test Panel",
        icon=":material/gavel:",
        default=True,
    ),
    st.Page(
        "views/situation_explorer.py",
        title="Situation Explorer",
        icon=":material/grid_view:",
    ),
    st.Page(
        "views/team_scout.py",
        title="Team Scout",
        icon=":material/groups:",
    ),
    st.Page(
        "views/about_method.py",
        title="About the Method",
        icon=":material/menu_book:",
    ),
]

# Style guide is only exposed during development mode (TENDENCY_DEV=1)
if os.environ.get("TENDENCY_DEV") == "1":
    pages.append(
        st.Page(
            "views/style_guide.py",
            title="Design System & Style Guide",
            icon=":material/palette:",
        )
    )

nav = st.navigation(pages)
nav.run()

# Sidebar footer attribution
st.sidebar.markdown(
    """
    <div style="margin-top: 2rem; padding-top: 1rem; "
         "border-top: 1px solid var(--color-border); font-size: 0.75rem; "
         "color: var(--color-text-muted); line-height: 1.35;">
        Play-by-play data: <strong>nflverse</strong> (CC BY 4.0).<br>
        EPA, win probability &amp; xpass are nflfastR model outputs.
    </div>
    """,
    unsafe_allow_html=True,
)
