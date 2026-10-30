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
from app.ui.components import (
    caveat_callout,
    empty_state,
    metric_card,
    verdict_banner,
)
from app.ui.layout import render_page_footer, render_page_header
from app.ui.theme import inject_css

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
    # 5. Power callout (from power.json — not hardcoded)
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
    # 6. Pre-declaration link (plan_commit from primary.json)
    # ------------------------------------------------------------------
    plan_commit = prim.plan_commit
    repo_url = f"{_REPO_BASE}/{plan_commit}"
    st.markdown(
        f"[View pre-registered plan (PLAN.md, commit `{plan_commit[:8]}…`)]({repo_url})"
    )

    render_page_footer()


if __name__ == "__main__":
    render()
