"""About the Method view: Research plan, preregistration, and protocol (D6.2)."""

import streamlit as st

from app.data import get_results_bundle
from app.ui.layout import render_page_footer, render_page_header
from app.ui.theme import render_callout


def render() -> None:
    """Render methodological overview and protocol transparency."""
    load_res = get_results_bundle()
    render_page_header(load_res)

    st.header("About the Method")
    st.caption("Pre-registered observational protocol and statistical design.")

    st.write(
        "Tendency Tax studies whether going against situational tendencies is "
        "associated with higher offensive efficiency (EPA/play). To avoid snooping "
        "and p-hacking, the design followed strict pre-registration principles."
    )

    st.divider()

    st.subheader("Core Methodological Principles")

    st.markdown(
        render_callout(
            "none",
            "Holdout data (2024–2025) remained completely sealed throughout model "
            "selection, hyperparameter tuning, and protocol specification. "
            "Estimates were produced exactly once with zero post-hoc modifications.",
            title="1. Sealed Holdout Protection",
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        render_callout(
            "cant_tell",
            "All findings are strictly observational associations. We use language "
            "such as 'is associated with' and avoid causal assertions ('causes', "
            "'proves', 'optimal') because unobserved defensive counter-strategies "
            "and execution confound pure causal attribution.",
            title="2. Non-Causal / Observational Interpretation",
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        render_callout(
            "premium",
            "Inference relies on game-clustered bootstrap distributions (1,000 "
            "replicates), paired with non-parametric placebo permutations to rule out "
            "mechanical artifact and collinearity distortions.",
            title="3. Clustered Uncertainty & Placebo Nulls",
        ),
        unsafe_allow_html=True,
    )

    render_page_footer()


if __name__ == "__main__":
    render()
