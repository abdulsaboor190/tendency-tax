"""Shared shell layouts (header, footer, fixture banner, error) for Tendency Tax app."""

import streamlit as st

from app.data import LoadResult
from app.ui.theme import inject_css

APP_TITLE = "Tendency Tax"
APP_SUBTITLE = (
    "Does being predictable cost an NFL offense? "
    "A pre-declared test on held-out 2024–2025 seasons."
)
FOOTER_ATTRIBUTION = (
    "Play-by-play data: nflverse (CC BY 4.0). EPA, win probability and xpass "
    "are nflfastR model outputs."
)


def render_page_header(load_result: LoadResult) -> None:
    """Render the standard design system CSS and fixture banner."""
    inject_css()

    if load_result.is_fixtures:
        st.warning(
            "⚠️ **Sample data, not real results** — currently viewing synthetic or "
            "fixture estimates.",
            icon=None,
        )

    if load_result.error_file:
        st.error(
            f"**Missing Required Result File:** `{load_result.error_file}`\n\n"
            f"{load_result.error_message}\n\n"
            f"To generate this file, run:\n```bash\n{load_result.make_command}\n```"
        )


def render_page_footer() -> None:
    """Render the standard footer attribution across every page."""
    st.divider()
    st.markdown(
        f"<div style='text-align: center; color: var(--color-text-muted); "
        f"font-size: 12px; padding: 12px 0;'>"
        f"{FOOTER_ATTRIBUTION}</div>",
        unsafe_allow_html=True,
    )
