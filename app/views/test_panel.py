"""Test Panel view: Pre-registered primary hypothesis test (Deliverable D7.1).

Displays the stored verdict from primary.json verdict-first, with:
  - Verdict banner (mapped from stored verdict.state + verdict.direction)
  - Plain-language sentence from the numbers (association-only wording)
  - Three-condition checklist (met / not met; icon + text, never color alone)
  - Metric cards: holdout (pre-declared), early (context), tune (context)
  - Power callout from power.json (not hardcoded)
  - Pre-declaration commit link

Design-system state mapping (binding, from D7.1 spec):
  - "supported" + "positive"  -> "premium"
  - "supported" + "negative"  -> "reverse"
  - anything else             -> "cant_tell"
"""

import streamlit as st

from app.data import format_ci, format_epa, format_int, get_results_bundle
from app.ui.charts import build_decile_gap_chart, build_placebo_histogram
from app.ui.components import (
    caveat_callout,
    empty_state,
    metric_card,
    verdict_banner,
)
from app.ui.layout import render_page_footer, render_page_header
from app.ui.theme import inject_css
from app.views.about_method import render_method_content
from app.views.exploratory import render_exploratory_checks

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONDITION_LABELS: dict[str, str] = {
    "c1_ci_excludes_zero": "C1: 95% confidence interval excludes zero",
    "c2_sign_matches_early": "C2: Sign of delta matches the early split",
    "c3_outside_placebo": "C3: Observed delta lies outside placebo null range",
}

_CONDITION_DESCRIPTIONS: dict[str, str] = {
    "c1_ci_excludes_zero": (
        "The 95% bootstrap CI must not include zero under game-cluster resampling."
    ),
    "c2_sign_matches_early": (
        "The holdout delta must point in the same direction as the 2016–2021 estimate."
    ),
    "c3_outside_placebo": (
        "The observed delta must fall outside the 2.5–97.5 percentile range "
        "of the permutation null."
    ),
}

_REPO_BASE = "https://github.com/abdulsaboor190/tendency-tax/commit"


# ---------------------------------------------------------------------------
# Verdict state mapping
# ---------------------------------------------------------------------------

def map_verdict_state(verdict_state: str, direction: str) -> str:
    """Map stored verdict fields to a design-system chip/banner state.

    Parameters
    ----------
    verdict_state : str
        The ``verdict.state`` value from primary.json, one of
        ``"supported"`` or ``"not supported / inconclusive"``.
    direction : str
        The ``verdict.direction`` value, one of
        ``"positive"``, ``"negative"``, or ``"zero"``.

    Returns
    -------
    str
        One of ``"premium"``, ``"reverse"``, or ``"cant_tell"``.

    Mapping rules (binding, D7.1 spec):
      - "supported"  +  "positive" -> "premium"
      - "supported"  +  "negative" -> "reverse"
      - anything else              -> "cant_tell"
    """
    if verdict_state == "supported":
        if direction == "positive":
            return "premium"
        if direction == "negative":
            return "reverse"
    return "cant_tell"


# ---------------------------------------------------------------------------
# Page renderer
# ---------------------------------------------------------------------------

def render() -> None:
    """Render the pre-registered primary hypothesis test panel (D7.1)."""
    inject_css()
    load_res = get_results_bundle()
    render_page_header(load_res)

    st.header("Test Panel")
    st.caption(
        "Pre-registered confirmatory test of the primary Tendency Tax hypothesis."
    )

    # Guard: no bundle or no primary result
    if not load_res.bundle or not load_res.bundle.primary:
        empty_state(
            "Primary result artifact is unavailable. "
            "Run `make confirm` to generate it, then reload."
        )
        render_page_footer()
        return

    prim = load_res.bundle.primary
    v = prim.verdict
    holdout = prim.holdout
    sb = holdout.spec_b

    # ------------------------------------------------------------------
    # 1. Verdict banner (uses STORED sentence — never recomputed here)
    # ------------------------------------------------------------------
    ds_state = map_verdict_state(v.state, v.direction)
    verdict_banner(v.sentence, ds_state)

    if ds_state == "cant_tell":
        caveat_callout(
            "An inconclusive result is not evidence of no effect. It means the "
            "held-out evidence did not satisfy all three pre-declared decision "
            "rules.",
            kind="info",
        )

    # ------------------------------------------------------------------
    # 2. Plain-language line from the numbers (association-only)
    # ------------------------------------------------------------------
    ci_lo, ci_hi = sb.delta_ci_95[0], sb.delta_ci_95[1]
    ci_includes_zero = ci_lo < 0 < ci_hi
    ci_clause = (
        "the interval includes zero"
        if ci_includes_zero
        else "the interval excludes zero"
    )
    st.markdown(
        f"On the held-out 2024–2025 seasons, plays against the situational expectation "
        f"were associated with **{format_epa(sb.delta)} EPA per play** "
        f"(95% CI {format_ci(ci_lo, ci_hi)}); {ci_clause}."
    )

    st.divider()

    # ------------------------------------------------------------------
    # 3. Three-condition checklist
    # ------------------------------------------------------------------
    st.subheader("Pre-Declared Conditions (all three must hold)")
    conditions_dict: dict[str, bool] = {
        "c1_ci_excludes_zero": v.conditions.c1_ci_excludes_zero,
        "c2_sign_matches_early": v.conditions.c2_sign_matches_early,
        "c3_outside_placebo": v.conditions.c3_outside_placebo,
    }

    all_met = all(conditions_dict.values())
    summary = (
        "All three pre-declared conditions are met."
        if all_met
        else f"One or more conditions are not met: {', '.join(v.failed)}."
    )
    st.caption(summary)

    for cond_key, met in conditions_dict.items():
        icon = "✅" if met else "✗"
        status_text = "Met" if met else "Not met"
        label = _CONDITION_LABELS.get(cond_key, cond_key)
        desc = _CONDITION_DESCRIPTIONS.get(cond_key, "")
        st.markdown(f"**{icon} {label}** — *{status_text}*")
        st.caption(desc)

    st.divider()

    # ------------------------------------------------------------------
    # 4. Metric cards: holdout (pre-declared) + early + tune (context)
    # ------------------------------------------------------------------
    st.subheader("Split Estimates")
    col1, col2, col3 = st.columns(3)

    with col1:
        metric_card(
            label="Holdout Δ (pre-declared test, 2024–2025)",
            value=format_epa(sb.delta),
            interval=(ci_lo, ci_hi),
            n=holdout.n_plays,
            help_text=f"N games: {format_int(holdout.n_games)}",
        )

    with col2:
        early_sb = prim.early.spec_b
        metric_card(
            label="Early Δ (context, 2016–2021)",
            value=format_epa(early_sb.delta),
            interval=(early_sb.delta_ci_95[0], early_sb.delta_ci_95[1]),
            n=prim.early.n_plays,
        )

    with col3:
        tune_sb = prim.tune.spec_b
        metric_card(
            label="Tune Δ (context, 2022–2023)",
            value=format_epa(tune_sb.delta),
            interval=(tune_sb.delta_ci_95[0], tune_sb.delta_ci_95[1]),
            n=prim.tune.n_plays,
        )

    st.divider()

    # ------------------------------------------------------------------
    # 5. Central Decile Chart (Deliverable D7.2)
    # ------------------------------------------------------------------
    st.subheader("Pass − Run EPA Gap by Predictability Decile")

    # Split dataframes from bundle
    early_df = None
    tune_df = None
    if load_res.bundle.decile_gap_dev is not None:
        dev_df = load_res.bundle.decile_gap_dev
        early_df = dev_df[dev_df["split"] == "early"].copy()
        tune_df = dev_df[dev_df["split"] == "tune"].copy()
    holdout_df = load_res.bundle.decile_gap_holdout

    include_tune = st.toggle(
        "Show Tune split (2022–2023)",
        value=False,
        help="Display the tune split series alongside early and holdout.",
    )

    if early_df is not None and not early_df.empty:
        fig, dropped = build_decile_gap_chart(
            early_df=early_df,
            holdout_df=holdout_df,
            tune_df=tune_df,
            include_tune=include_tune,
            min_plays=200,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Dropped bin note
        if dropped:
            dropped_notes = [
                f"Decile {d['decile']} in {d['split']} ({d['reason']})"
                for d in dropped
            ]
            st.caption(
                "**Dropped bins (< 200 plays of either call):** "
                f"{'; '.join(dropped_notes)}."
            )

        # Descriptive caption
        st.caption(
            "This chart shows the gap in average EPA between passes and runs at each "
            "level of how predictable a pass was (decile of pre-snap p̂). "
            "Decile edges were computed from the early split. "
            "This visualization is purely descriptive and is not the "
            "confirmatory test statistic."
        )
    else:
        empty_state("Decile gap data unavailable.")

    st.divider()

    # ------------------------------------------------------------------
    # 6. Placebo Histogram (Deliverable D7.3)
    # ------------------------------------------------------------------
    st.subheader("Placebo Check: Permutation Null Distribution")

    placebo_holdout_df = load_res.bundle.placebo_holdout
    if placebo_holdout_df is not None and not placebo_holdout_df.empty:
        fig_placebo, shift_stats = build_placebo_histogram(
            placebo_df=placebo_holdout_df,
            observed_delta=sb.delta,
            split_name="Holdout (2024–2025)",
        )
        st.plotly_chart(fig_placebo, use_container_width=True)

        # One-sentence "how to read this" caption
        st.caption(
            "The histogram shows what the estimate looks like when the link "
            "between the situation and the play call is scrambled."
        )

        # Data-driven warning note if null median differs from 0 by > 0.5 * SD
        if shift_stats["is_shifted"]:
            caveat_callout(
                f"The placebo null distribution is not centered near zero "
                f"(null median = {format_epa(shift_stats['median'])}, "
                f"SD = {shift_stats['sd']:.4f}). "
                "Because the median differs from zero by more than half its "
                "standard deviation, comparing the observed value to this range "
                "should be read with care.",
                kind="warning",
            )

        # Context on conditions: placebo is C3, but verdict failed on another condition
        failed_list = v.failed
        if "c3_outside_placebo" not in failed_list:
            if failed_list:
                failed_str = ", ".join(
                    _CONDITION_LABELS.get(k, k) for k in failed_list
                )
                st.info(
                    f"This placebo check satisfies Pre-Declared Condition 3 "
                    f"(the observed estimate lies outside the null 2.5–97.5% range). "
                    f"The overall confirmatory verdict is inconclusive because a "
                    f"different condition was not met: {failed_str}."
                )
            else:
                st.info(
                    "This placebo check satisfies Pre-Declared Condition 3 "
                    "(the observed estimate lies outside the null 2.5–97.5% range)."
                )
        else:
            st.info(
                "The observed estimate falls within the placebo null range, "
                "failing Pre-Declared Condition 3."
            )
    else:
        empty_state("Holdout placebo permutation data is unavailable.")

    st.divider()

    # ------------------------------------------------------------------
    # 7. Exploratory Checks (Deliverable D7.5)
    # ------------------------------------------------------------------
    render_exploratory_checks(load_res.bundle.robustness)

    st.divider()

    # ------------------------------------------------------------------
    # 8. Power callout (from power.json — not hardcoded)
    # ------------------------------------------------------------------
    power = load_res.bundle.power
    smallest_delta = power.smallest_delta_80_power_formula
    holdout_se = power.holdout_expected_se

    caveat_callout(
        f"The holdout test was sized to detect a true tendency-tax premium of at least "
        f"**{format_epa(smallest_delta)} EPA/play** with 80% power "
        f"(expected SE ≈ {holdout_se:.3f}). "
        f"An inconclusive result is not evidence of no effect; "
        f"the test was underpowered for smaller effects.",
        kind="info",
    )

    # ------------------------------------------------------------------
    # 9. Method Drawer (Deliverable D7.4)
    # ------------------------------------------------------------------
    with st.expander("How this was computed (Method details)", expanded=False):
        render_method_content(prim, in_expander=True)

    # ------------------------------------------------------------------
    # 10. Pre-declaration link (plan_commit from primary.json)
    # ------------------------------------------------------------------
    plan_commit = prim.plan_commit
    repo_url = f"{_REPO_BASE}/{plan_commit}"
    st.markdown(
        f"[View pre-registered plan (PLAN.md, commit `{plan_commit[:8]}…`)]({repo_url})"
    )

    render_page_footer()


if __name__ == "__main__":
    render()
