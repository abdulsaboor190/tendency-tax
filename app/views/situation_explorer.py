"""Situation Explorer view: 27-cell grid of situational tendencies (Deliverable D8.1).

Controls:
  - Down (1, 2, 3)
  - Distance bucket (short 1–3, medium 4–7, long 8+)
  - Field zone (red zone <=20, midfield 21–79, backed up >=80)
  - Split selector (early, tune, holdout, all)

Syncs state bidirectionally with Streamlit query parameters.
Default split is 'tune' as pre-registered in PLAN.md (Decision 3).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from app.data import (
    format_ci,
    format_epa,
    format_int,
    get_query_param,
    get_results_bundle,
    set_query_param,
)
from app.ui.components import caveat_callout, empty_state
from app.ui.layout import render_page_footer, render_page_header
from app.ui.theme import inject_css

# ---------------------------------------------------------------------------
# Control Options and Defaults
# ---------------------------------------------------------------------------

DEFAULT_SPLIT: str = "tune"  # Decided in PLAN.md (Decision 3)
DEFAULT_DOWN: int = 1
DEFAULT_DIST: str = "short"
DEFAULT_ZONE: str = "midfield"

VALID_DOWNS: tuple[int, ...] = (1, 2, 3)
VALID_DISTS: tuple[str, ...] = ("short", "medium", "long")
VALID_ZONES: tuple[str, ...] = ("red_zone", "midfield", "backed_up")
VALID_SPLITS: tuple[str, ...] = ("early", "tune", "holdout", "all")

DIST_LABELS: dict[str, str] = {
    "short": "Short (1–3 yds)",
    "medium": "Medium (4–7 yds)",
    "long": "Long (8+ yds)",
}

# Zone ranges from config.ZONE_CUT_POINTS: (20, 80)
# red_zone: 1–20, midfield: 21–79, backed_up: 80–99
ZONE_LABELS: dict[str, str] = {
    "red_zone": "Red Zone (1–20)",
    "midfield": "Midfield (21–79)",
    "backed_up": "Backed Up (80–99)",
}

SPLIT_LABELS: dict[str, str] = {
    "early": "Early (2016–2021)",
    "tune": "Tune (2022–2023) [Default]",
    "holdout": "Holdout (2024–2025)",
    "all": "All (2016–2025)",
}


# ---------------------------------------------------------------------------
# State & Query Params Helpers
# ---------------------------------------------------------------------------

def parse_explorer_state() -> dict[str, Any]:
    """Parse down, dist, zone, and split from query params with fallback defaults.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys 'down' (int), 'dist' (str), 'zone' (str), 'split' (str).
    """
    # Down
    raw_down = get_query_param("down")
    try:
        parsed_down = int(raw_down) if raw_down is not None else DEFAULT_DOWN
        down_val = parsed_down if parsed_down in VALID_DOWNS else DEFAULT_DOWN
    except (ValueError, TypeError):
        down_val = DEFAULT_DOWN

    # Distance bucket
    raw_dist = str(get_query_param("dist", "")).lower()
    dist_val = raw_dist if raw_dist in VALID_DISTS else DEFAULT_DIST

    # Field zone
    raw_zone = str(get_query_param("zone", "")).lower()
    zone_val = raw_zone if raw_zone in VALID_ZONES else DEFAULT_ZONE

    # Split
    raw_split = str(get_query_param("split", "")).lower()
    split_val = raw_split if raw_split in VALID_SPLITS else DEFAULT_SPLIT

    return {
        "down": down_val,
        "dist": dist_val,
        "zone": zone_val,
        "split": split_val,
    }


def sync_explorer_query_params(
    down: int, dist: str, zone: str, split: str
) -> None:
    """Sync active selections to Streamlit query params."""
    set_query_param("down", down)
    set_query_param("dist", dist)
    set_query_param("zone", zone)
    set_query_param("split", split)


def get_cell_id(down: int, dist: str, zone: str) -> str:
    """Format standard 27-cell identifier: '{down}_{dist}_{zone}'."""
    return f"{down}_{dist}_{zone}"


def lookup_cell_record(
    cells_df: pd.DataFrame,
    split: str,
    down: int,
    dist: str,
    zone: str,
) -> dict[str, Any] | None:
    """Lookup cell row in cells_df for the specified parameters."""
    cell_id = get_cell_id(down, dist, zone)
    sub = cells_df[
        (cells_df["split"] == split) & (cells_df["cell_id"] == cell_id)
    ]
    if sub.empty:
        return None
    return sub.iloc[0].to_dict()


# ---------------------------------------------------------------------------
# View Renderer
# ---------------------------------------------------------------------------

def render() -> None:
    """Render the Situation Explorer with shareable controls and cell detail."""
    inject_css()
    load_res = get_results_bundle()
    render_page_header(load_res)

    st.header("Situation Explorer")
    st.caption("27-cell grid across down, distance bucket, and field zone.")

    st.write(
        "Interactive explorer evaluating situational tendency estimates across "
        "distinct offensive contexts."
    )

    st.divider()

    if not load_res.bundle or load_res.bundle.cells is None:
        empty_state(
            "Cell-level grid results artifact is unavailable. "
            "Run `make secondary` to generate it, then reload."
        )
        render_page_footer()
        return

    cells_df = load_res.bundle.cells

    # 1. Parse current state from query parameters (with robust fallback)
    state = parse_explorer_state()

    # Reset button handling
    if st.button(
        "↺ Reset Filters", help="Reset all selections to pre-registered defaults"
    ):
        sync_explorer_query_params(
            DEFAULT_DOWN, DEFAULT_DIST, DEFAULT_ZONE, DEFAULT_SPLIT
        )
        state = {
            "down": DEFAULT_DOWN,
            "dist": DEFAULT_DIST,
            "zone": DEFAULT_ZONE,
            "split": DEFAULT_SPLIT,
        }
        st.rerun()

    # 2. Controls layout
    st.subheader("Situational Controls")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Down**")
        selected_down = st.pills(
            "Down selection",
            options=list(VALID_DOWNS),
            default=state["down"],
            label_visibility="collapsed",
            key="explorer_down",
        )
        if selected_down is None:
            selected_down = state["down"]

    with c2:
        st.markdown("**Distance Bucket**")
        selected_dist = st.pills(
            "Distance bucket selection",
            options=list(VALID_DISTS),
            format_func=lambda d: DIST_LABELS.get(d, d),
            default=state["dist"],
            label_visibility="collapsed",
            key="explorer_dist",
        )
        if selected_dist is None:
            selected_dist = state["dist"]

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Field Zone** (yardline_100)")
        selected_zone = st.pills(
            "Field zone selection",
            options=list(VALID_ZONES),
            format_func=lambda z: ZONE_LABELS.get(z, z),
            default=state["zone"],
            label_visibility="collapsed",
            key="explorer_zone",
        )
        if selected_zone is None:
            selected_zone = state["zone"]

    with c4:
        st.markdown("**Split Selection**")
        selected_split = st.pills(
            "Split selection",
            options=list(VALID_SPLITS),
            format_func=lambda s: SPLIT_LABELS.get(s, s),
            default=state["split"],
            label_visibility="collapsed",
            key="explorer_split",
        )
        if selected_split is None:
            selected_split = state["split"]

    # Sync selection changes to query parameters
    if (
        selected_down != state["down"]
        or selected_dist != state["dist"]
        or selected_zone != state["zone"]
        or selected_split != state["split"]
    ):
        sync_explorer_query_params(
            selected_down, selected_dist, selected_zone, selected_split
        )

    # 3. Persistent callout when 'all' split is selected
    if selected_split == "all":
        caveat_callout(
            "This view pools all ten seasons (2016–2025), including the "
            "held-out seasons, and is strictly exploratory.",
            kind="warning",
        )

    st.divider()

    # 4. Cell Detail Card
    cell_id = get_cell_id(selected_down, selected_dist, selected_zone)
    record = lookup_cell_record(
        cells_df,
        split=selected_split,
        down=selected_down,
        dist=selected_dist,
        zone=selected_zone,
    )

    st.subheader(f"Cell Details: `{cell_id}`")
    st.caption(
        f"Context: Down {selected_down} • {DIST_LABELS[selected_dist]} • "
        f"{ZONE_LABELS[selected_zone]} • Split: {selected_split.capitalize()}"
    )

    if record is not None:
        delta_val = record.get("delta")
        ci_lo = record.get("ci_lo")
        ci_hi = record.get("ci_hi")
        n_plays = record.get("n_plays", 0)
        n_pass = record.get("n_pass", 0)
        n_run = record.get("n_run", 0)
        phat_med = record.get("phat_median")
        verdict = record.get("verdict", "cant_tell")
        testable = record.get("testable", True)
        reason = record.get("reason", "")

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Tendency Delta (EPA)", format_epa(delta_val))
        with col_m2:
            st.metric("95% Clustered CI", format_ci(ci_lo, ci_hi))
        with col_m3:
            st.metric("Plays Count", format_int(n_plays))
        with col_m4:
            st.metric("Verdict State", str(verdict).replace("_", " ").title())

        # Sub-metrics: pass/run counts, phat median
        st.write("")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.caption(f"Pass Plays: **{format_int(n_pass)}**")
        with col_s2:
            st.caption(f"Run Plays: **{format_int(n_run)}**")
        with col_s3:
            phat_ok = phat_med is not None and phat_med == phat_med
            phat_str = f"{phat_med:.3f}" if phat_ok else "—"
            st.caption(f"Median p̂: **{phat_str}**")

        if not testable:
            st.warning(f"Note: This cell is marked untestable ({reason}).")

    else:
        msg = f"No cell record found for '{cell_id}' in split '{selected_split}'."
        empty_state(msg)

    render_page_footer()


if __name__ == "__main__":
    render()
