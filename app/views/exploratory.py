"""Exploratory robustness and subgroup checks (Deliverable D7.5).

Provides:
  - Robustness variant interval plot for selectable split (primary vs shotgun,
    no penalties, xpass)
  - By-down interval plot across splits with data-driven sign agreement helper
  - By-season compact interval plot with noisy flag constant
  - Success rate variant on 0/1 scale (isolated from EPA axis)
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from app.data import format_ci, format_int
from app.ui.components import caveat_callout, empty_state, interval_plot

# Ratio threshold for labeling season estimates as noisy:
# If 95% CI width > NOISY_INTERVAL_RATIO_THRESHOLD * |estimate|, row is labeled noisy.
NOISY_INTERVAL_RATIO_THRESHOLD: float = 1.0

_PRIMARY_VARIANTS = [
    "primary",
    "controls_shotgun_no_huddle",
    "no_penalties",
    "alt_expectation_xpass",
]

_DOWN_VARIANTS = [
    "down_1",
    "down_2",
    "down_3",
]


def compute_down_sign_agreement(down_df: pd.DataFrame) -> dict[str, Any]:
    """Assess whether estimates agree in sign across splits and if CIs include zero.

    Parameters
    ----------
    down_df : pd.DataFrame
        Subset of robustness data for a down across splits ('early', 'tune', 'holdout').

    Returns
    -------
    dict[str, Any]
        Dictionary with:
          - sign_agrees: bool
          - sign_direction: 'positive' | 'negative' | 'mixed'
          - all_include_zero: bool
          - n_include_zero: int
          - n_splits: int
          - line: str (data-generated descriptive line)
    """
    if down_df.empty:
        return {
            "sign_agrees": False,
            "sign_direction": "none",
            "all_include_zero": False,
            "n_include_zero": 0,
            "n_splits": 0,
            "line": "No data available for this down.",
        }

    deltas = down_df["delta"].tolist()
    ci_los = down_df["ci_lo"].tolist()
    ci_his = down_df["ci_hi"].tolist()
    n_splits = len(deltas)

    all_pos = all(d > 0 for d in deltas)
    all_neg = all(d < 0 for d in deltas)
    sign_agrees = all_pos or all_neg
    sign_direction = "positive" if all_pos else ("negative" if all_neg else "mixed")

    includes_zero_list = [lo < 0 < hi for lo, hi in zip(ci_los, ci_his, strict=False)]
    all_include_zero = all(includes_zero_list)
    none_include_zero = not any(includes_zero_list)
    n_include_zero = sum(includes_zero_list)

    if sign_agrees:
        sign_clause = f"Sign agrees across all three splits ({sign_direction})"
    else:
        sign_clause = "Sign does not agree across splits (mixed directions)"

    if all_include_zero:
        zero_clause = "every 95% confidence interval includes zero"
    elif none_include_zero:
        zero_clause = "no 95% confidence interval includes zero"
    else:
        zero_clause = (
            f"{n_include_zero} of {n_splits} confidence intervals include zero"
        )

    line = f"{sign_clause}; {zero_clause}."

    return {
        "sign_agrees": sign_agrees,
        "sign_direction": sign_direction,
        "all_include_zero": all_include_zero,
        "n_include_zero": n_include_zero,
        "n_splits": n_splits,
        "line": line,
    }


def filter_robustness_variants(
    df: pd.DataFrame,
    split: str = "holdout",
) -> list[dict[str, Any]]:
    """Prepare rows for the primary and variants interval plot."""
    sub = df[(df["split"] == split) & (df["outcome"] == "epa")].copy()
    # Filter to primary and key variants
    sub = sub[sub["variant_id"].isin(_PRIMARY_VARIANTS)].copy()

    # Maintain consistent presentation order
    variant_order = {vid: i for i, vid in enumerate(_PRIMARY_VARIANTS)}
    sub["sort_order"] = sub["variant_id"].map(variant_order).fillna(99)
    sub = sub.sort_values("sort_order")

    rows: list[dict[str, Any]] = []
    for r in sub.itertuples():
        label_text = f"{r.variant_label} [exploratory]"
        rows.append({
            "display_label": label_text,
            "delta": r.delta,
            "ci_lo": r.ci_lo,
            "ci_hi": r.ci_hi,
            "n_plays": r.n_plays,
            "verdict": "cant_tell",
        })
    return rows


def filter_by_down_rows(
    df: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Prepare rows for by-down interval plot and compute sign agreement lines."""
    sub = df[df["variant_id"].isin(_DOWN_VARIANTS) & (df["outcome"] == "epa")].copy()

    down_labels = {
        "down_1": "1st Down",
        "down_2": "2nd Down",
        "down_3": "3rd Down",
    }
    split_order = {"early": 1, "tune": 2, "holdout": 3}
    down_order = {"down_1": 1, "down_2": 2, "down_3": 3}

    sub["split_order"] = sub["split"].map(split_order).fillna(99)
    sub["down_order"] = sub["variant_id"].map(down_order).fillna(99)
    sub = sub.sort_values(["down_order", "split_order"])

    rows: list[dict[str, Any]] = []
    for r in sub.itertuples():
        d_name = down_labels.get(r.variant_id, r.variant_id)
        label_text = f"{d_name} — {r.split.capitalize()} [exploratory]"
        rows.append({
            "display_label": label_text,
            "delta": r.delta,
            "ci_lo": r.ci_lo,
            "ci_hi": r.ci_hi,
            "n_plays": r.n_plays,
            "verdict": r.split,  # map to series color token if available
        })

    # Compute agreement lines per down
    agreement_lines: dict[str, str] = {}
    for d_id in _DOWN_VARIANTS:
        d_sub = df[
            (df["variant_id"] == d_id) & (df["outcome"] == "epa")
        ].sort_values("split")
        res = compute_down_sign_agreement(d_sub)
        d_display = down_labels.get(d_id, d_id)
        agreement_lines[d_id] = f"**{d_display}:** {res['line']}"

    return rows, agreement_lines


def filter_by_season_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Prepare rows for the compact by-season interval plot."""
    is_season = df["variant_id"].str.startswith("season_")
    sub = df[is_season & (df["outcome"] == "epa")].copy()
    sub = sub.sort_values("variant_id")

    rows: list[dict[str, Any]] = []
    for r in sub.itertuples():
        season_num = r.variant_id.replace("season_", "")
        width = r.ci_hi - r.ci_lo
        abs_est = abs(r.delta)
        is_noisy = width > (NOISY_INTERVAL_RATIO_THRESHOLD * abs_est)
        noisy_tag = " (noisy)" if is_noisy else ""
        label_text = f"Season {season_num}{noisy_tag} [exploratory]"

        rows.append({
            "display_label": label_text,
            "delta": r.delta,
            "ci_lo": r.ci_lo,
            "ci_hi": r.ci_hi,
            "n_plays": r.n_plays,
            "verdict": r.split,
        })
    return rows


def render_exploratory_checks(robustness_df: pd.DataFrame | None) -> None:
    """Render the exploratory robustness and subgroup checks section."""
    st.subheader("Exploratory Checks")

    # Always-visible disclaimer callout
    caveat_callout(
        "These checks were not part of the pre-declared test. "
        "Treat them as questions for future work, not findings.",
        kind="info",
    )

    if robustness_df is None or robustness_df.empty:
        empty_state("Robustness results unavailable.")
        return

    # 1. Primary & Variants by selectable split
    st.markdown("#### 1. Specification Robustness Variants")
    selected_split = st.selectbox(
        "Select split for specification variants:",
        options=["holdout", "early", "tune"],
        index=0,
        format_func=lambda s: {
            "holdout": "Holdout (2024–2025)",
            "early": "Early (2016–2021)",
            "tune": "Tune (2022–2023)",
        }.get(s, s),
    )

    variant_rows = filter_robustness_variants(robustness_df, split=selected_split)
    if variant_rows:
        fig_var = interval_plot(
            variant_rows,
            label_col="display_label",
            title=f"Specification Variants — {selected_split.capitalize()}",
            xaxis_title="Delta (EPA / play)",
            height=280,
        )
        st.plotly_chart(fig_var, use_container_width=True)
    else:
        empty_state(f"No variant data for split '{selected_split}'.")

    st.divider()

    # 2. By Down (9 rows across early, tune, holdout)
    st.markdown("#### 2. Estimates by Down")
    down_rows, agreement_lines = filter_by_down_rows(robustness_df)
    if down_rows:
        fig_down = interval_plot(
            down_rows,
            label_col="display_label",
            title="Estimates by Down Across Splits (EPA / play)",
            xaxis_title="Delta (EPA / play)",
            height=440,
        )
        st.plotly_chart(fig_down, use_container_width=True)

        st.markdown("**Down Consistency Summary (Data-Generated):**")
        for line in agreement_lines.values():
            st.markdown(f"- {line}")
    else:
        empty_state("Down-level robustness data unavailable.")

    st.divider()

    # 3. By Season
    st.markdown("#### 3. Single-Season Estimates")
    season_rows = filter_by_season_rows(robustness_df)
    if season_rows:
        fig_season = interval_plot(
            season_rows,
            label_col="display_label",
            title="Single-Season Estimates (2016–2025)",
            xaxis_title="Delta (EPA / play)",
            height=460,
        )
        st.plotly_chart(fig_season, use_container_width=True)
        st.caption(
            f"Seasons where the 95% confidence interval width exceeds "
            f"{NOISY_INTERVAL_RATIO_THRESHOLD:.1f}× the point estimate magnitude "
            "are labeled *(noisy)*."
        )
    else:
        empty_state("Single-season robustness data unavailable.")

    st.divider()

    # 4. Success Rate Variant (isolated from EPA axis)
    st.markdown("#### 4. Alternative Outcome: Play Success Rate")
    st.caption(
        "Success rate is an indicator (0/1) for whether a play gained positive EPA. "
        "Units are in percentage points / probability difference, not EPA."
    )

    success_df = robustness_df[robustness_df["outcome"] == "success"].copy()
    if not success_df.empty:
        s_rows: list[dict[str, Any]] = []
        for r in success_df.sort_values("split").itertuples():
            s_rows.append({
                "display_label": f"Success Rate — {r.split.capitalize()} [exploratory]",
                "delta": r.delta,
                "ci_lo": r.ci_lo,
                "ci_hi": r.ci_hi,
                "n_plays": r.n_plays,
                "verdict": r.split,
            })

        fig_success = interval_plot(
            s_rows,
            label_col="display_label",
            title="Success Rate Delta by Split (0/1 Scale)",
            xaxis_title="Delta (Difference in Success Rate)",
            height=260,
        )
        st.plotly_chart(fig_success, use_container_width=True)

        cols = st.columns(len(s_rows))
        for col, row in zip(cols, s_rows, strict=False):
            with col:
                ci_str = format_ci(row["ci_lo"], row["ci_hi"])
                n_str = format_int(row["n_plays"])
                st.metric(
                    label=row["display_label"],
                    value=f"{row['delta']:+.3f}",
                    help=f"95% CI {ci_str} | N = {n_str}",
                )
    else:
        empty_state("Success-rate variant data unavailable.")
