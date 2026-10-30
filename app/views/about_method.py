"""About the Method view: Research plan, preregistration, and protocol (D7.4).

Provides a comprehensive methodological guide written for a non-technical reader:
  1. Research question & what "predictable" means (p̂).
  2. Three core definitions: p̂, surprise, and delta.
  3. Sealed-holdout design: splits, PLAN.md commit, execution timestamp, GitHub link.
  4. Verdict rule: 3 pre-declared conditions.
  5. Methodological limitations: association only, selection effects,
     no personnel/formation features, modeled EPA, regular season scope,
     holdout sample size.
  6. Data attribution and license (nflverse, CC BY 4.0).

All numbers are pulled dynamically from results artifacts (no hardcoding).
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from app.data import format_ci, format_epa, format_int, get_results_bundle
from app.ui.components import empty_state
from app.ui.layout import render_page_footer, render_page_header
from app.ui.theme import inject_css

_REPO_BASE = "https://github.com/abdulsaboor190/tendency-tax/commit"


def render_method_content(bundle_primary: Any, in_expander: bool = False) -> None:
    """Render the method explanation sections.

    Parameters
    ----------
    bundle_primary : PrimaryResult
        Primary results model from the results bundle.
    in_expander : bool
        Whether this is rendered inside an expander/drawer or as a standalone page.
    """
    prim = bundle_primary
    holdout = prim.holdout
    sb = holdout.spec_b
    delta_val = sb.delta
    ci_lo, ci_hi = sb.delta_ci_95[0], sb.delta_ci_95[1]
    plan_commit = prim.plan_commit
    timestamp = prim.timestamp

    header_lvl = "###" if in_expander else "##"

    # 1. The Research Question
    st.markdown(f"{header_lvl} 1. The Question: Does Predictability Pay?")
    st.write(
        "Does an NFL offense experience an efficiency penalty or benefit when its "
        "play call (pass vs. run) is predictable from the pre-snap situation? "
        "Here, **predictable** means having a high situational pass probability (p̂) "
        "estimated purely from pre-snap context (down, distance, time, field position, "
        "score differential, timeouts, and spread)—using a model that never sees "
        "EPA or play outcomes."
    )

    # 2. Key Definitions
    st.markdown(f"{header_lvl} 2. Key Definitions")
    st.markdown(
        "- **p̂ (Predicted Pass Probability)**: The pre-snap probability that an "
        "offense will pass rather than run, conditioned exclusively on situational "
        "context before the ball is snapped.\n"
        "- **Surprise**: How far an actual play call deviates from situational "
        "expectation. Passing when a run is expected, or running when a pass is "
        "expected, represents a counter-tendency call.\n"
        "- **Δ (Delta)**: The average EPA gap associated with counter-tendency "
        "play-calling. In Expected Points Added (EPA) per play, **positive Δ** "
        "indicates that plays called against situational expectations were associated "
        "with higher efficiency."
    )

    # 3. Sealed-Holdout Design
    st.markdown(f"{header_lvl} 3. Sealed-Holdout Protocol")
    st.write(
        "To protect against p-hacking and researcher degrees of freedom, the data "
        "were partitioned into three strictly segregated time periods:"
    )
    st.markdown(
        f"- **Early Split (2016–2021)**: {format_int(prim.early.n_plays)} plays used "
        "to build, explore, and establish the statistical specification.\n"
        f"- **Tune Split (2022–2023)**: {format_int(prim.tune.n_plays)} plays used "
        "to validate calibration, verify thresholds, and pre-register decision rules.\n"
        f"- **Holdout Split (2024–2025)**: {format_int(holdout.n_plays)} plays "
        f"({format_int(holdout.n_games)} games) sealed until the analysis protocol "
        "was committed to version control, evaluated exactly once."
    )

    repo_url = f"{_REPO_BASE}/{plan_commit}"
    st.markdown(
        f"**Pre-Registration Plan Commit**: [`{plan_commit[:8]}…`]({repo_url})  \n"
        f"**Confirmatory Run Timestamp**: `{timestamp}`"
    )

    # 4. Verdict Rule
    st.markdown(f"{header_lvl} 4. Confirmatory Verdict Rule")
    st.write(
        "The hypothesis of a meaningful tendency tax requires satisfying all three "
        "pre-declared conditions on the held-out 2024–2025 data:"
    )
    c1_met = prim.verdict.conditions.c1_ci_excludes_zero
    c2_met = prim.verdict.conditions.c2_sign_matches_early
    c3_met = prim.verdict.conditions.c3_outside_placebo

    c1_status = "✅ Met" if c1_met else "✗ Not met"
    c2_status = "✅ Met" if c2_met else "✗ Not met"
    c3_status = "✅ Met" if c3_met else "✗ Not met"

    early_delta = prim.early.spec_b.delta
    st.markdown(
        f"1. **95% Confidence Interval Excludes Zero** ({c1_status}): "
        f"Observed holdout Δ = {format_epa(delta_val)} with 95% CI "
        f"{format_ci(ci_lo, ci_hi)}.\n"
        f"2. **Directional Consistency** ({c2_status}): "
        f"Holdout estimate sign matches early split estimate "
        f"(early Δ = {format_epa(early_delta)}).\n"
        f"3. **Placebo Separation** ({c3_status}): "
        f"Observed Δ falls outside the 2.5–97.5% empirical range of 500 "
        f"situation-scrambled permutations."
    )

    # 5. Methodological Limitations
    st.markdown(f"{header_lvl} 5. Methodological Limitations")
    st.markdown(
        "- **Observational Association Only**: These estimates reflect empirical "
        "associations, not experimental interventions. Unobserved defensive "
        "anticipation and in-game adaptations cannot be completely separated.\n"
        "- **Coaching Selection Effects**: Coaches break tendency for specific "
        "tactical reasons (e.g., noticing a specific defensive alignment). Observed "
        "differences may reflect the quality of that situational read rather than "
        "an inherent return to surprise.\n"
        "- **No Personnel or Formation Features**: To avoid post-snap leakage and "
        "maintain pre-snap objectivity, pre-snap personnel packages (11 vs. 12 "
        "personnel) and formations (shotgun, under center) were excluded from "
        "the primary model.\n"
        "- **EPA is a Modeled Outcome**: Expected Points Added is itself generated "
        "from an underlying nflverse historical regression model, carrying its "
        "own modeling assumptions.\n"
        "- **Regular Season Scope**: Playoff games and overtime endgames are "
        "excluded to prevent distortions from win-or-go-home play-calling or "
        "extreme tactical incentives.\n"
        f"- **Holdout Uncertainty**: Evaluating two NFL seasons "
        f"({format_int(holdout.n_plays)} plays) yields relatively wide confidence "
        f"intervals, meaning smaller tendency premiums cannot be ruled out."
    )

    # 6. Data Attribution and License
    st.markdown(f"{header_lvl} 6. Data Attribution & License")
    st.markdown(
        "Play-by-play data provided by the **nflverse** project under the "
        "[Creative Commons Attribution 4.0 International (CC BY 4.0)]"
        "(https://creativecommons.org/licenses/by/4.0/) license. "
        "EPA, win probability, and baseline xpass are outputs of the "
        "nflfastR modeling suite."
    )


def render() -> None:
    """Render the full 'About the method' view."""
    inject_css()
    load_res = get_results_bundle()
    render_page_header(load_res)

    st.header("About the Method")
    st.caption(
        "Pre-registered observational protocol, statistical definitions, "
        "and design transparency."
    )

    if not load_res.bundle or not load_res.bundle.primary:
        empty_state(
            "Results bundle unavailable. Run `make confirm` to produce artifacts."
        )
        render_page_footer()
        return

    render_method_content(load_res.bundle.primary, in_expander=False)
    st.divider()
    render_page_footer()


if __name__ == "__main__":
    render()
