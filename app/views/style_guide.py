"""Style guide page demonstrating design system tokens and components (D6.1)."""

import plotly.graph_objects as go
import streamlit as st

from app.ui.theme import (
    FONT_FAMILY,
    SERIES_COLORS,
    VERDICT_TOKENS,
    apply_template,
    inject_css,
    render_callout,
    render_verdict_chip,
)


def render_style_guide() -> None:
    """Render full interactive style guide."""
    inject_css()

    st.title("Design System & Style Guide")
    st.caption("Tendency Tax Design System (Deliverable D6.1)")

    st.markdown(
        "Design principle: **Verdict first, evidence beneath**. "
        "Every state has an icon and text label. Never use color alone. "
        "Language is strictly association-only."
    )

    st.divider()

    # 1. Verdict States & Chips
    st.subheader("1. Verdict States & Chips (Okabe-Ito)")
    st.write(
        "Four mutually exclusive states. Note that *Can't tell* is a "
        "first-class neutral state with a hatched pattern."
    )

    cols = st.columns(4)
    for col, (state, token) in zip(cols, VERDICT_TOKENS.items(), strict=True):
        with col:
            st.markdown(render_verdict_chip(state), unsafe_allow_html=True)
            caption_html = (
                "<div style='margin-top: 8px; font-size: 12px; "
                "color: var(--color-text-muted);'>"
                f"<code>{token.color}</code><br/>Icon: <b>{token.icon}</b>"
                "</div>"
            )
            st.markdown(caption_html, unsafe_allow_html=True)

    st.write("")

    # 2. Callouts
    st.subheader("2. Uncertainty & Evidence Callouts")
    st.markdown(
        render_callout(
            "premium",
            "Plays against the situational expectation are associated with "
            "higher EPA (consistent with a predictability cost). "
            "95% CI: [+0.027, +0.298], N = 21,843.",
            title="Surprise Premium Detected",
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        render_callout(
            "cant_tell",
            "The confidence interval includes zero and is too wide to rule out "
            "a meaningful difference. We find no reliable evidence of an "
            "association.",
            title="Inconclusive Evidence (Can't Tell)",
        ),
        unsafe_allow_html=True,
    )

    st.divider()

    # 3. Series & Surface Colors
    st.subheader("3. Split Series & Surface Palette")
    s_cols = st.columns(4)
    for col, (series, col_hex) in zip(s_cols, SERIES_COLORS.items(), strict=True):
        with col:
            st.markdown(
                f"<div style='background-color: {col_hex}; height: 36px; "
                f"border-radius: 6px; margin-bottom: 6px;'></div>"
                f"<b>{series.capitalize()}</b>: <code>{col_hex}</code>",
                unsafe_allow_html=True,
            )

    st.divider()

    # 4. Type Scale & Number Displays
    st.subheader("4. Type Scale & Numbers")
    type_demo_html = (
        f"<div style='font-family: {FONT_FAMILY};'>"
        "<h1>Heading 1 (Inter 32px / Bold)</h1>"
        "<h2>Heading 2 (Inter 24px / Semi-bold)</h2>"
        "<h3>Heading 3 (Inter 18px / Medium)</h3>"
        "<p>Body regular: Association-only language describing calls.</p>"
        "<div class='metric-display'>+0.041 <span style='font-size: 16px; "
        "font-weight: normal; color: var(--color-text-muted);'>"
        "EPA / play</span></div>"
        "<div class='metric-caption'>"
        "95% bootstrap CI [+0.012, +0.070], N = 142,946</div>"
        "</div>"
    )
    st.markdown(type_demo_html, unsafe_allow_html=True)

    st.divider()

    # 5. Plotly Template Chart
    st.subheader("5. Accessible Plotly Chart ('tendency' template)")
    st.write("Clean gridlines, transparent surface, readable on phone widths.")

    # Build sample forest plot / confidence interval comparison
    categories = ["Early (2016-2021)", "Tune (2022-2023)", "Holdout (2024-2025)"]
    deltas = [0.041, 0.052, 0.048]
    ci_los = [0.012, -0.010, 0.018]
    ci_his = [0.070, 0.114, 0.078]
    errors_plus = [hi - d for d, hi in zip(deltas, ci_his, strict=True)]
    errors_minus = [d - lo for d, lo in zip(deltas, ci_los, strict=True)]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=deltas,
            y=categories,
            mode="markers",
            marker=dict(
                size=12,
                color=[
                    SERIES_COLORS["early"],
                    SERIES_COLORS["tune"],
                    SERIES_COLORS["holdout"],
                ],
            ),
            error_x=dict(
                type="data",
                symmetric=False,
                array=errors_plus,
                arrayminus=errors_minus,
                thickness=2,
                width=6,
            ),
            hovertemplate=(
                "<b>%{y}</b><br/>"
                "Delta: %{x:.3f} EPA/play<br/>"
                "<extra></extra>"
            ),
        )
    )

    # Reference zero line
    fig.add_vline(x=0.0, line_dash="dash", line_color="rgba(138, 143, 152, 0.6)")

    fig.update_layout(
        title="Sample Spec B Delta by Data Split",
        xaxis_title="Delta (EPA / play)",
        yaxis_title="Data Split",
        height=320,
    )

    apply_template(fig)
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    render_style_guide()
