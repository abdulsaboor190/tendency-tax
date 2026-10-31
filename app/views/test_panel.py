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

from app.data import format_epa, format_int, get_results_bundle
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
    power = load_res.bundle.power

    ci_lo, ci_hi = sb.delta_ci_95[0], sb.delta_ci_95[1]
    ds_state = map_verdict_state(v.state, v.direction)

    tab_result, tab_evidence, tab_exploratory, tab_method = st.tabs(
        ["Result", "Evidence", "Exploratory", "How it works"]
    )

    # ==================================================================
    # Tab 1: Result
    # ==================================================================
    with tab_result:
        # 1. Verdict banner
        headline_label = (
            "Inconclusive"
            if ds_state == "cant_tell"
            else ("Surprise Premium" if ds_state == "premium" else "Surprise Reverse")
        )
        banner_headline = (
            f"{headline_label} · {format_epa(sb.delta)} EPA per play "
            f"[{format_epa(ci_lo)}, {format_epa(ci_hi)}]"
        )
        banner_note = (
            "Inconclusive is not evidence of no effect. It means the held-out "
            "evidence did not satisfy all three pre-declared decision rules."
            if ds_state == "cant_tell"
            else None
        )

        verdict_banner(
            v.sentence,
            ds_state,
            headline=banner_headline,
            note=banner_note,
        )

        # 2. Row of three key numbers
        smallest_delta = (
            power.smallest_delta_80_power_formula if power else 0.038
        )
        holdout_se = power.holdout_expected_se if power else 0.038

        k1, k2, k3 = st.columns(3)
        with k1:
            metric_card(
                label="Holdout Δ (pre-declared test, 2024–2025)",
                value=format_epa(sb.delta),
                interval=(ci_lo, ci_hi),
                n=holdout.n_plays,
                help_text=f"95% bootstrap CI on {format_int(holdout.n_games)} games",
            )
        with k2:
            metric_card(
                label="Smallest detectable effect (80% power)",
                value=f"≥ {format_epa(smallest_delta)}",
                help_text=(
                    f"Design power threshold with expected SE ≈ {holdout_se:.3f}. "
                    "Test was underpowered for effects smaller than this."
                ),
            )
        with k3:
            metric_card(
                label="Held-Out Sample Evaluated",
                value=f"{format_int(holdout.n_plays)} plays",
                help_text=f"Across {format_int(holdout.n_games)} regular-season games",
            )

        st.divider()

        # 3. Three-condition checklist
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

        # 4. Context split estimates (early & tune)
        st.subheader("Contextual Splits (Exploratory / Tuning)")
        col_early, col_tune = st.columns(2)
        with col_early:
            early_sb = prim.early.spec_b
            metric_card(
                label="Early Δ (baseline context, 2016–2021)",
                value=format_epa(early_sb.delta),
                interval=(early_sb.delta_ci_95[0], early_sb.delta_ci_95[1]),
                n=prim.early.n_plays,
                help_text="Exploratory baseline split",
            )
        with col_tune:
            tune_sb = prim.tune.spec_b
            metric_card(
                label="Tune Δ (tuning context, 2022–2023)",
                value=format_epa(tune_sb.delta),
                interval=(tune_sb.delta_ci_95[0], tune_sb.delta_ci_95[1]),
                n=prim.tune.n_plays,
                help_text="Pre-registration validation split",
            )

        # Pre-declaration link
        plan_commit = prim.plan_commit
        if plan_commit:
            repo_url = f"{_REPO_BASE}/{plan_commit}"
            st.caption(
                f"Protocol pre-registered at commit "
                f"[`{plan_commit[:8]}…`]({repo_url}) in `PLAN.md` before examining "
                f"held-out data."
            )

    # ==================================================================
    # Tab 2: Evidence
    # ==================================================================
    with tab_evidence:
        st.subheader("Pass − Run EPA Gap by Predictability Decile")

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
            fig_decile, dropped = build_decile_gap_chart(
                early_df=early_df,
                holdout_df=holdout_df,
                tune_df=tune_df,
                include_tune=include_tune,
                min_plays=200,
            )
            st.plotly_chart(fig_decile, use_container_width=True)

            if dropped:
                dropped_notes = [
                    f"Decile {d['decile']} in {d['split']} ({d['reason']})"
                    for d in dropped
                ]
                st.caption(
                    "**Dropped bins (< 200 plays of either call):** "
                    f"{'; '.join(dropped_notes)}."
                )

            st.caption(
                "This chart shows the gap in average EPA between passes and "
                "runs at each level of how predictable a pass was "
                "(decile of pre-snap p̂). "
                "Decile edges were computed from the early split. "
                "This visualization is purely descriptive and is not the "
                "confirmatory test statistic."
            )
        else:
            empty_state("Decile gap data unavailable.")

        st.divider()

        st.subheader("Placebo Check: Permutation Null Distribution")

        placebo_holdout_df = load_res.bundle.placebo_holdout
        if placebo_holdout_df is not None and not placebo_holdout_df.empty:
            fig_placebo, shift_stats = build_placebo_histogram(
                placebo_df=placebo_holdout_df,
                observed_delta=sb.delta,
                split_name="Holdout (2024–2025)",
            )
            st.plotly_chart(fig_placebo, use_container_width=True)

            st.caption(
                "The histogram shows what the estimate looks like when the link "
                "between the situation and the play call is scrambled."
            )

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

            failed_list = v.failed
            if "c3_outside_placebo" not in failed_list:
                if failed_list:
                    failed_str = ", ".join(
                        _CONDITION_LABELS.get(k, k) for k in failed_list
                    )
                    st.info(
                        "This placebo check satisfies Pre-Declared Condition 3 "
                        "(the observed estimate lies outside the null 2.5–97.5% "
                        f"range). The overall confirmatory verdict is inconclusive "
                        f"because a different condition was not met: {failed_str}."
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

    # ==================================================================
    # Tab 3: Exploratory
    # ==================================================================
    with tab_exploratory:
        render_exploratory_checks(load_res.bundle.robustness)

    # ==================================================================
    # Tab 4: How it works
    # ==================================================================
    with tab_method:
        render_method_content(prim, in_expander=False)

    render_page_footer()


if __name__ == "__main__":
    render()
