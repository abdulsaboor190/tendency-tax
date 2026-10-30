"""Situation Explorer: 27-cell grid of situational tendencies (D6.2)."""

import streamlit as st

from app.data import format_ci, format_epa, format_int, get_results_bundle
from app.ui.layout import render_page_footer, render_page_header
from app.ui.theme import render_verdict_chip


def render() -> None:
    """Render situational grid explorer."""
    load_res = get_results_bundle()
    render_page_header(load_res)

    st.header("Situation Explorer")
    st.caption("27-cell grid across down, distance bucket, and field zone.")

    st.write(
        "Interactive explorer evaluating situational tendency premiums across "
        "distinct game contexts with Benjamini-Hochberg FDR adjustments."
    )

    st.divider()

    # Verdict-First Layout Skeleton
    if load_res.bundle and load_res.bundle.cells is not None:
        cells_df = load_res.bundle.cells
        st.subheader("Situational Cells Overview")

        # Top summary chips
        n_premium = int((cells_df["verdict"] == "premium").sum())
        n_reverse = int((cells_df["verdict"] == "reverse").sum())
        n_cant = int((cells_df["verdict"] == "cant_tell").sum())

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                render_verdict_chip("premium", f"Premium cells: {n_premium}"),
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                render_verdict_chip("reverse", f"Reverse cells: {n_reverse}"),
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                render_verdict_chip("cant_tell", f"Can't tell: {n_cant}"),
                unsafe_allow_html=True,
            )

        st.write("")

        # Data preview table
        cols_to_show = [
            "split", "cell_id", "down", "dist_bucket", "zone",
            "n_plays", "delta", "ci_lo", "ci_hi", "q_bh", "verdict"
        ]
        available_cols = [c for c in cols_to_show if c in cells_df.columns]
        display_df = cells_df[available_cols].copy()
        if "delta" in display_df:
            display_df["delta"] = display_df["delta"].apply(format_epa)
        if "ci_lo" in display_df and "ci_hi" in display_df:
            display_df["95% CI"] = [
                format_ci(lo, hi)
                for lo, hi in zip(cells_df["ci_lo"], cells_df["ci_hi"], strict=False)
            ]
        if "n_plays" in display_df:
            display_df["n_plays"] = display_df["n_plays"].apply(format_int)

        st.dataframe(display_df.head(15), use_container_width=True)

    else:
        st.info("Cell-level grid results artifact is loading or unavailable.")

    render_page_footer()


if __name__ == "__main__":
    render()
