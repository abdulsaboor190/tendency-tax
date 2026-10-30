"""Test Panel view: Pre-registered primary hypothesis test (Deliverable D6.2)."""

import streamlit as st

from app.data import format_ci, format_epa, format_int, get_results_bundle
from app.ui.layout import render_page_footer, render_page_header
from app.ui.theme import render_callout, render_verdict_chip


def render() -> None:
    """Render the primary hypothesis test panel."""
    load_res = get_results_bundle()
    render_page_header(load_res)

    st.header("Test Panel")
    st.caption("Pre-registered test of the primary Tendency Tax hypothesis.")

    st.write(
        "Evaluates whether offensive play-calling against situational "
        "expectations ($\\hat{p}$) is associated with higher EPA."
    )

    st.divider()

    # Verdict-First Layout Skeleton
    if load_res.bundle and load_res.bundle.primary:
        prim = load_res.bundle.primary
        st.subheader("Confirmatory Verdict (Holdout 2024–2025)")

        # Chip + Sentence
        verdict_state = prim.verdict.state
        chip_state = "premium" if verdict_state == "supported" else "cant_tell"
        st.markdown(render_verdict_chip(chip_state), unsafe_allow_html=True)
        st.write("")

        callout_html = render_callout(
            chip_state,
            prim.verdict.sentence,
            title="Pre-Registered Decision Rule",
        )
        st.markdown(callout_html, unsafe_allow_html=True)

        # Primary metrics
        c1, c2, c3, c4 = st.columns(4)
        spec_b = prim.holdout.spec_b
        with c1:
            st.metric("Primary Delta (EPA)", format_epa(spec_b.delta))
        with c2:
            st.metric("95% Clustered CI", format_ci(spec_b.ci_lo, spec_b.ci_hi))
        with c3:
            st.metric("Bootstrap SE", f"{spec_b.se:.3f}")
        with c4:
            st.metric("Sample Size", f"{format_int(spec_b.n_plays)} plays")

    else:
        st.info("Primary result artifact is loading or unavailable.")

    render_page_footer()


if __name__ == "__main__":
    render()
