"""Team Scout view: Empirical-Bayes shrunk tendency tendencies (D6.2)."""

import streamlit as st

from app.data import format_ci, format_epa, format_int, format_pct, get_results_bundle
from app.ui.layout import render_page_footer, render_page_header


def render() -> None:
    """Render team scout exploration."""
    load_res = get_results_bundle()
    render_page_header(load_res)

    st.header("Team Scout")
    st.caption("Team-season situational tendency rates and Empirical-Bayes shrinkage.")

    st.write(
        "Evaluates offensive play-calling tendencies when in situation leans "
        "($\\hat{p} \\ge 0.65$ or $\\le 0.35$). Estimates are shrunk toward the league "
        "mean via DerSimonian-Laird random effects."
    )

    st.divider()

    # Verdict-First Layout Skeleton
    if load_res.bundle and load_res.bundle.teams is not None:
        teams_df = load_res.bundle.teams
        st.subheader("Team Season Summary")

        # Metrics overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Team-Seasons", format_int(len(teams_df)))
        with col2:
            avg_lean = (
                teams_df["lean_share"].mean() if "lean_share" in teams_df else None
            )
            st.metric("Mean Lean Share", format_pct(avg_lean))
        with col3:
            avg_follow = (
                teams_df["follow_rate"].mean() if "follow_rate" in teams_df else None
            )
            st.metric("Mean Follow Rate", format_pct(avg_follow))

        st.write("")

        # Data preview table
        cols_to_show = [
            "split", "season", "team", "n_plays", "n_lean", "lean_share",
            "follow_rate", "d_raw", "d_shrunk", "shrunk_lo", "shrunk_hi", "warn_low_n"
        ]
        available_cols = [c for c in cols_to_show if c in teams_df.columns]
        display_df = teams_df[available_cols].copy()
        if "lean_share" in display_df:
            display_df["lean_share"] = display_df["lean_share"].apply(format_pct)
        if "follow_rate" in display_df:
            display_df["follow_rate"] = display_df["follow_rate"].apply(format_pct)
        if "d_raw" in display_df:
            display_df["d_raw"] = display_df["d_raw"].apply(format_epa)
        if "d_shrunk" in display_df:
            display_df["d_shrunk"] = display_df["d_shrunk"].apply(format_epa)
        if "shrunk_lo" in display_df and "shrunk_hi" in display_df:
            display_df["95% Shrunk CI"] = [
                format_ci(lo, hi)
                for lo, hi in zip(
                    teams_df["shrunk_lo"], teams_df["shrunk_hi"], strict=False
                )
            ]

        st.dataframe(display_df.head(15), use_container_width=True)

    else:
        st.info("Team scout results artifact is loading or unavailable.")

    render_page_footer()


if __name__ == "__main__":
    render()
