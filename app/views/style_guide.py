"""Style guide page demonstrating design system tokens and UI components (D6.3)."""

import streamlit as st

from app.ui.components import (
    caveat_callout,
    empty_state,
    interval_plot,
    loading_skeleton,
    metric_card,
    sample_size_badge,
    state_legend,
    verdict_banner,
    verdict_chip,
)
from app.ui.theme import (
    FONT_FAMILY,
    SERIES_COLORS,
    VERDICT_TOKENS,
    inject_css,
)


def render_style_guide() -> None:
    """Render full interactive style guide displaying all tokens and components."""
    inject_css()

    st.title("Design System & UI Components")
    st.caption("Tendency Tax Design System & Reusable Components (Deliverable D6.3)")

    st.markdown(
        "Design principle: **Verdict first, evidence beneath**. "
        "Every state has an icon and text label. Never use color alone. "
        "Language is strictly observational and association-only."
    )

    st.divider()

    # 1. State Legend
    st.subheader("1. State Legend")
    state_legend()

    st.write("")

    # 2. Verdict Banner
    st.subheader("2. Verdict Banner (Top of Test Panel)")
    verdict_banner(
        sentence=(
            "Plays against the situational expectation are associated with "
            "higher EPA (consistent with a predictability cost). "
            "Confirmatory holdout delta = +0.041 [+0.012, +0.070], N = 48,469 plays."
        ),
        state="premium",
    )

    # 3. Verdict Chips (Regular and Small)
    st.subheader("3. Verdict Chips")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(verdict_chip("premium"), unsafe_allow_html=True)
        st.write("")
        st.markdown(verdict_chip("premium", small=True), unsafe_allow_html=True)
    with c2:
        st.markdown(verdict_chip("reverse"), unsafe_allow_html=True)
        st.write("")
        st.markdown(verdict_chip("reverse", small=True), unsafe_allow_html=True)
    with c3:
        st.markdown(verdict_chip("none"), unsafe_allow_html=True)
        st.write("")
        st.markdown(verdict_chip("none", small=True), unsafe_allow_html=True)
    with c4:
        st.markdown(verdict_chip("cant_tell"), unsafe_allow_html=True)
        st.write("")
        st.markdown(verdict_chip("cant_tell", small=True), unsafe_allow_html=True)

    st.divider()

    # 4. Metric Cards & Sample Badges
    st.subheader("4. Metric Cards & Sample Size Badges")
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        metric_card(
            label="Spec B Primary Delta",
            value="+0.041",
            interval=(0.012, 0.070),
            n=48469,
            help_text="EPA difference: pass vs run demeaned within buckets",
        )
    with mc2:
        metric_card(
            label="Situation Grid Delta",
            value="+0.142",
            interval=(-0.010, 0.294),
            n=1200,
            help_text="Spec A interaction estimate within cell 1_long_midfield",
        )
    with mc3:
        metric_card(
            label="Team Scout Shrunk Delta",
            value="+0.018",
            interval=(-0.045, 0.081),
            n=45,  # low n demo
            help_text="DerSimonian-Laird empirical-Bayes posterior estimate",
        )

    st.write("")
    st.write("Sample size badges (normal vs warning threshold):")
    st.markdown(
        f"{sample_size_badge(142946)} &nbsp;&nbsp; {sample_size_badge(45, min_n=60)}",
        unsafe_allow_html=True,
    )

    st.divider()

    # 5. Point & Interval Chart
    st.subheader("5. Point and Interval Chart (`interval_plot`)")
    sample_rows = [
        {
            "label": "Early (2016–2021)",
            "delta": 0.041,
            "ci_lo": 0.012,
            "ci_hi": 0.070,
            "n_plays": 142946,
            "verdict": "premium",
        },
        {
            "label": "Tune (2022–2023)",
            "delta": 0.052,
            "ci_lo": -0.010,
            "ci_hi": 0.114,
            "n_plays": 50122,
            "verdict": "cant_tell",
        },
        {
            "label": "Holdout (2024–2025)",
            "delta": 0.048,
            "ci_lo": 0.018,
            "ci_hi": 0.078,
            "n_plays": 48469,
            "verdict": "premium",
        },
        {
            "label": "Null Placebo Reference",
            "delta": 0.001,
            "ci_lo": -0.015,
            "ci_hi": 0.017,
            "n_plays": 500,
            "verdict": "none",
        },
    ]
    fig = interval_plot(
        sample_rows,
        title="Cross-Split Estimate Comparison",
        xaxis_title="Surprise Delta (EPA / play)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # 6. Caveats, Skeletons, and Empty States
    st.subheader("6. Caveat Callouts, Empty States & Skeletons")
    caveat_callout(
        "All estimates represent observational correlations across situations. "
        "They do not establish direct cause and effect.",
        kind="info",
    )
    caveat_callout(
        "Sample size for this subset is low (N < 60). "
        "Posterior estimates are heavily shrunk.",
        kind="warning",
    )

    sc1, sc2 = st.columns(2)
    with sc1:
        st.write("Empty State:")
        empty_state("No qualifying plays found for selected down and distance filter.")
    with sc2:
        st.write("Loading Skeleton:")
        loading_skeleton(height=100)

    st.divider()

    # 7. Raw Palette & Typography Reference
    st.subheader("7. Color Tokens & Type Reference")
    p_cols = st.columns(4)
    for col, (state, token) in zip(p_cols, VERDICT_TOKENS.items(), strict=True):
        with col:
            st.markdown(
                f"<div style='background-color: {token.color}; height: 28px; "
                f"border-radius: 4px; margin-bottom: 4px;'></div>"
                f"<b>{token.name}</b>: <code>{token.color}</code>",
                unsafe_allow_html=True,
            )

    s_cols = st.columns(3)
    split_items = [
        ("Early", SERIES_COLORS["early"]),
        ("Tune", SERIES_COLORS["tune"]),
        ("Holdout", SERIES_COLORS["holdout"]),
    ]
    for col, (split, col_hex) in zip(s_cols, split_items, strict=True):
        with col:
            st.markdown(
                f"<div style='background-color: {col_hex}; height: 28px; "
                f"border-radius: 4px; margin-bottom: 4px;'></div>"
                f"<b>{split}</b>: <code>{col_hex}</code>",
                unsafe_allow_html=True,
            )

    st.markdown(
        f"<div style='font-family: {FONT_FAMILY}; margin-top: 16px;'>"
        "<h4>Typography: Inter (WCAG AA compliant contrast)</h4>"
        "<p>All text uses relative sizing and responds fluidly to viewport widths.</p>"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    render_style_guide()
