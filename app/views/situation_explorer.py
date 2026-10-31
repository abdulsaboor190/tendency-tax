"""Situation Explorer view: 27-cell grid of situational tendencies (Phase 8).

Deliverables:
  - D8.1: Situational controls with query parameter sync
  - D8.2: Cell result panel with stored verdict, uncertainty intervals,
    and cross-split comparison
  - D8.3: 27-cell overview grid with hatched fill for can't tell, state legend,
    accessible fallback table, and interactive cell selection
  - D8.4: Scenario slider with interactive score differential & time remaining,
    scikit-learn version verification, and feature order validation

Controls:
  - Down (1, 2, 3)
  - Distance bucket (short 1–3, medium 4–7, long 8+)
  - Field zone (red zone <=20, midfield 21–79, backed up >=80)
  - Split selector (early, tune, holdout, all)

Default split is 'tune' as pre-registered in PLAN.md (Decision 3).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn
import streamlit as st

from app.data import (
    format_ci,
    format_epa,
    format_int,
    get_query_param,
    get_results_bundle,
    set_query_param,
)
from app.ui.charts import cell_grid_chart, compute_cells_summary
from app.ui.components import (
    caveat_callout,
    empty_state,
    interval_plot,
    metric_card,
    state_legend,
    verdict_chip,
)
from app.ui.layout import render_page_footer, render_page_header
from app.ui.theme import VERDICT_TOKENS, inject_css

# ---------------------------------------------------------------------------
# Control Options and Defaults
# ---------------------------------------------------------------------------

DEFAULT_SPLIT: str = "tune"  # Decided in PLAN.md (Decision 3)
DEFAULT_DOWN: int = 1
DEFAULT_DIST: str = "long"
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

# Representative defaults per situational cell
DIST_REPRESENTATIVE_YDSTOGO: dict[str, int] = {
    "short": 2,    # Midpoint of 1-3
    "medium": 6,   # Midpoint of 4-7
    "long": 10,    # Standard long yardage (8+)
}

ZONE_REPRESENTATIVE_YARDLINE: dict[str, int] = {
    "red_zone": 10,   # Inside red zone (1-20)
    "midfield": 50,   # Center midfield (21-79)
    "backed_up": 85,  # Backed up territory (80-99)
}

FIXED_SCENARIO_DEFAULTS: dict[str, Any] = {
    "qtr": 2,
    "posteam_timeouts_remaining": 3,
    "defteam_timeouts_remaining": 3,
    "posteam_spread": 0.0,
}

MODEL_PATH = Path("results/models/phat_model_2016_2023.joblib")
METADATA_PATH = Path("results/models/phat_model_2016_2023.json")


# ---------------------------------------------------------------------------
# State & Query Params Helpers
# ---------------------------------------------------------------------------

def get_readable_cell_title(down: int, dist: str, zone: str, split: str) -> str:
    """Format readable title, e.g. '1st & long (8+ yds) · midfield · Tune'."""
    ord_down = {1: "1st", 2: "2nd", 3: "3rd"}.get(down, f"{down}th")
    dist_text = DIST_LABELS.get(dist, dist).lower()
    zone_text = ZONE_LABELS.get(zone, zone).lower().split("(")[0].strip()
    split_text = SPLIT_LABELS.get(split, split.capitalize()).split("[")[0].strip()
    return f"{ord_down} & {dist_text} · {zone_text} · {split_text}"


def parse_explorer_state() -> dict[str, Any]:
    """Parse down, dist, zone, and split from query params with fallback defaults.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys 'down' (int), 'dist' (str), 'zone' (str), 'split' (str).
    """
    raw_down = get_query_param("down")
    try:
        parsed_down = int(raw_down) if raw_down is not None else DEFAULT_DOWN
        down_val = parsed_down if parsed_down in VALID_DOWNS else DEFAULT_DOWN
    except (ValueError, TypeError):
        down_val = DEFAULT_DOWN

    raw_dist = str(get_query_param("dist", "")).lower()
    dist_val = raw_dist if raw_dist in VALID_DISTS else DEFAULT_DIST

    raw_zone = str(get_query_param("zone", "")).lower()
    zone_val = raw_zone if raw_zone in VALID_ZONES else DEFAULT_ZONE

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


def build_cell_split_rows(
    cells_df: pd.DataFrame,
    cell_id: str,
) -> list[dict[str, Any]]:
    """Build cross-split comparison rows for a cell in standard order."""
    cell_rows = cells_df[cells_df["cell_id"] == cell_id]
    split_order = ["early", "tune", "holdout", "all"]
    rows = []
    for s in split_order:
        sub = cell_rows[cell_rows["split"] == s]
        if sub.empty:
            rows.append(
                {
                    "split": s,
                    "label": SPLIT_LABELS.get(s, s.capitalize()),
                    "delta": None,
                    "ci_lo": None,
                    "ci_hi": None,
                    "n_plays": 0,
                    "verdict": "cant_tell",
                    "custom_text": "not testable (n=0)",
                }
            )
            continue
        rec = sub.iloc[0].to_dict()
        testable = rec.get("testable", True)
        below_min = rec.get("below_min_plays", False)
        n_plays = rec.get("n_plays", 0)

        if not testable or below_min:
            rows.append(
                {
                    "split": s,
                    "label": SPLIT_LABELS.get(s, s.capitalize()),
                    "delta": None,
                    "ci_lo": None,
                    "ci_hi": None,
                    "n_plays": n_plays,
                    "verdict": "cant_tell",
                    "custom_text": f"not testable (n={n_plays})",
                }
            )
            continue

        delta = rec.get("delta")
        ci_lo = rec.get("ci_lo")
        ci_hi = rec.get("ci_hi")
        d_val = None if (delta is None or math.isnan(delta)) else float(delta)
        lo_val = None if (ci_lo is None or math.isnan(ci_lo)) else float(ci_lo)
        hi_val = None if (ci_hi is None or math.isnan(ci_hi)) else float(ci_hi)

        verdict = rec.get("verdict", "cant_tell")
        if verdict not in VERDICT_TOKENS:
            verdict = "cant_tell"

        rows.append(
            {
                "split": s,
                "label": SPLIT_LABELS.get(s, s.capitalize()),
                "delta": d_val,
                "ci_lo": lo_val,
                "ci_hi": hi_val,
                "n_plays": n_plays,
                "verdict": verdict,
            }
        )
    return rows


def build_fallback_table(
    cells_df: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    """Build a 27-row text-only table fallback for screen readers and accessibility."""
    sub_df = cells_df[cells_df["split"] == split]
    rows: list[dict[str, Any]] = []

    for d in VALID_DOWNS:
        for dist in VALID_DISTS:
            for z in VALID_ZONES:
                c_id = get_cell_id(d, dist, z)
                cell_row = sub_df[sub_df["cell_id"] == c_id]
                if not cell_row.empty:
                    rec = cell_row.iloc[0].to_dict()
                    testable = rec.get("testable", True)
                    verdict = rec.get("verdict", "cant_tell")
                    delta = rec.get("delta")
                    ci_lo = rec.get("ci_lo")
                    ci_hi = rec.get("ci_hi")
                    q_bh = rec.get("q_bh")
                    n_plays = rec.get("n_plays", 0)
                    reason = rec.get("reason", "")
                else:
                    testable = False
                    verdict = "cant_tell"
                    delta = None
                    ci_lo = None
                    ci_hi = None
                    q_bh = None
                    n_plays = 0
                    reason = "No data recorded"

                token = VERDICT_TOKENS.get(verdict, VERDICT_TOKENS["cant_tell"])
                state_str = f"{token.icon} {token.label}"

                if not testable:
                    d_str = "Untestable"
                    ci_str = "—"
                    q_str = "—"
                elif verdict == "cant_tell":
                    d_str = "—"
                    ci_str = format_ci(ci_lo, ci_hi)
                    q_str = (
                        f"{q_bh:.4f}"
                        if (q_bh is not None and not math.isnan(q_bh))
                        else "—"
                    )
                else:
                    d_str = format_epa(delta)
                    ci_str = format_ci(ci_lo, ci_hi)
                    q_str = (
                        f"{q_bh:.4f}"
                        if (q_bh is not None and not math.isnan(q_bh))
                        else "—"
                    )

                rows.append(
                    {
                        "Cell ID": c_id,
                        "Down": d,
                        "Distance": DIST_LABELS[dist],
                        "Field Zone": ZONE_LABELS[z],
                        "State": state_str,
                        "Delta (EPA)": d_str,
                        "95% CI": ci_str,
                        "FDR q": q_str,
                        "Plays": format_int(n_plays),
                        "Reason": reason,
                    }
                )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Scenario Slider Model Helpers (Deliverable D8.4)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_scenario_model() -> tuple[Any | None, dict[str, Any] | None, str | None]:
    """Load saved model and metadata with version check.

    Returns
    -------
    tuple[Any | None, dict[str, Any] | None, str | None]
        (model, metadata, error_or_mismatch_message)
    """
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        return (
            None,
            None,
            f"Saved model artifact not found at {MODEL_PATH}.",
        )

    try:
        with open(METADATA_PATH, encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        return None, None, f"Failed to load model metadata: {e}"

    expected_ver = meta.get("sklearn_version", "")
    installed_ver = sklearn.__version__
    if expected_ver and expected_ver != installed_ver:
        return (
            None,
            meta,
            (
                f"scikit-learn version mismatch: model was serialized with "
                f"version {expected_ver}, but installed version is {installed_ver}. "
                "Scenario slider is disabled to ensure numerical reliability."
            ),
        )

    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        return None, meta, f"Failed to load serialized model: {e}"

    return model, meta, None


def predict_scenario_phat(
    model: Any,
    features_list: list[str],
    down: int,
    dist: str,
    zone: str,
    score_diff: int,
    half_sec_remaining: int,
) -> float:
    """Predict pass probability p̂ for a specific scenario using fixed defaults."""
    ydstogo = DIST_REPRESENTATIVE_YDSTOGO.get(dist, 10)
    yardline_100 = ZONE_REPRESENTATIVE_YARDLINE.get(zone, 50)
    qtr = 2
    game_sec_remaining = half_sec_remaining

    input_dict = {
        "down": int(down),
        "ydstogo": int(ydstogo),
        "yardline_100": int(yardline_100),
        "qtr": int(qtr),
        "game_seconds_remaining": int(game_sec_remaining),
        "half_seconds_remaining": int(half_sec_remaining),
        "score_differential": int(score_diff),
        "posteam_timeouts_remaining": int(
            FIXED_SCENARIO_DEFAULTS["posteam_timeouts_remaining"]
        ),
        "defteam_timeouts_remaining": int(
            FIXED_SCENARIO_DEFAULTS["defteam_timeouts_remaining"]
        ),
        "posteam_spread": float(FIXED_SCENARIO_DEFAULTS["posteam_spread"]),
    }

    # Verify input DataFrame columns match metadata feature order exactly
    df_input = pd.DataFrame([input_dict])[features_list]
    proba = model.predict_proba(df_input)
    # Target class 1 is qb_dropback (pass)
    return float(proba[0, 1])


# ---------------------------------------------------------------------------
# View Renderer
# ---------------------------------------------------------------------------

def render() -> None:
    """Render the Situation Explorer with overview grid, cell panel, and slider."""
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

    # Reset & Jump to Most-Played Cell buttons
    btn_col1, btn_col2, _ = st.columns([1, 1.4, 2])
    with btn_col1:
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

    with btn_col2:
        if st.button(
            "⚡ Jump to Most-Played Cell",
            help="Navigate directly to the most populated testable cell",
        ):
            # Most played cell across splits: 1st & long midfield
            sync_explorer_query_params(1, "long", "midfield", state["split"])
            state["down"] = 1
            state["dist"] = "long"
            state["zone"] = "midfield"
            st.rerun()

    # 2. Controls layout grouped in one compact card
    with st.container():
        st.markdown('<div class="tendency-card">', unsafe_allow_html=True)
        st.markdown(
            "<div style='font-weight: 700; margin-bottom: 8px;'>"
            "Situational Controls</div>",
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.caption("Down")
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
            st.caption("Distance Bucket")
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
            st.caption("Field Zone (yardline_100)")
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
            st.caption("Split Selection")
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

        st.markdown("</div>", unsafe_allow_html=True)

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

    # 4. 27-Cell Overview Grid (Deliverable D8.3)
    st.subheader("27-Cell Situational Overview")
    split_cells_df = cells_df[cells_df["split"] == selected_split]

    # Data-generated summary line
    summary_info = compute_cells_summary(split_cells_df)
    st.markdown(f"**{summary_info['summary_line']}**")

    # State legend
    state_legend()

    # Interactive Grid
    current_cell_id = get_cell_id(selected_down, selected_dist, selected_zone)
    grid_fig = cell_grid_chart(split_cells_df, selected_cell_id=current_cell_id)

    # Streamlit chart selection or selectbox fallback
    selection = st.plotly_chart(
        grid_fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode=["points"],
        key="cell_grid_chart",
    )

    # Check if a cell was selected via click in the chart
    if selection and hasattr(selection, "get"):
        points = selection.get("points", [])
        if points and len(points) > 0:
            customdata = points[0].get("customdata", [])
            if customdata and len(customdata) >= 4:
                sel_d = int(customdata[0])
                sel_dist = str(customdata[1])
                sel_zone = str(customdata[2])
                if (
                    sel_d != selected_down
                    or sel_dist != selected_dist
                    or sel_zone != selected_zone
                ):
                    sync_explorer_query_params(
                        sel_d, sel_dist, sel_zone, selected_split
                    )
                    st.rerun()

    # Selectbox fallback to navigate directly to any of the 27 cells
    all_27_cell_ids = [
        get_cell_id(d, dist, z)
        for d in VALID_DOWNS
        for dist in VALID_DISTS
        for z in VALID_ZONES
    ]
    cur_idx = (
        all_27_cell_ids.index(current_cell_id)
        if current_cell_id in all_27_cell_ids
        else 0
    )

    def _format_cell_option(cid: str) -> str:
        parts = cid.split("_", 2)
        d_val = parts[0]
        dist_lbl = DIST_LABELS.get(parts[1], parts[1])
        zone_lbl = ZONE_LABELS.get(parts[2], parts[2])
        return f"{cid} (Down {d_val} • {dist_lbl} • {zone_lbl})"

    chosen_cell = st.selectbox(
        "Select cell to inspect (alternative to chart click):",
        options=all_27_cell_ids,
        index=cur_idx,
        format_func=_format_cell_option,
        key="cell_select_box",
    )
    if chosen_cell != current_cell_id:
        c_parts = chosen_cell.split("_", 2)
        sync_explorer_query_params(
            int(c_parts[0]), c_parts[1], c_parts[2], selected_split
        )
        st.rerun()

    # Accessible text-only table fallback
    with st.expander("Accessible Table: All 27 Cells (Screen Reader & Plain Text)"):
        table_df = build_fallback_table(cells_df, selected_split)
        st.dataframe(table_df, use_container_width=True, hide_index=True)

    st.divider()

    # 5. Cell Result Panel (Deliverable D8.2)
    record = lookup_cell_record(
        cells_df,
        split=selected_split,
        down=selected_down,
        dist=selected_dist,
        zone=selected_zone,
    )

    readable_title = get_readable_cell_title(
        selected_down, selected_dist, selected_zone, selected_split
    )
    st.subheader(readable_title)
    st.caption(f"Cell ID: `{current_cell_id}`")

    if record is not None:
        delta_val = record.get("delta")
        ci_lo = record.get("ci_lo")
        ci_hi = record.get("ci_hi")
        q_bh = record.get("q_bh")
        n_plays = record.get("n_plays", 0)
        n_games = record.get("n_games", 0)
        n_pass = record.get("n_pass", 0)
        n_run = record.get("n_run", 0)
        phat_med = record.get("phat_median")
        phat_p25 = record.get("phat_p25")
        phat_p75 = record.get("phat_p75")
        verdict = record.get("verdict", "cant_tell")
        testable = record.get("testable", True)
        below_min = record.get("below_min_plays", False)

        chip_state = verdict if verdict in VERDICT_TOKENS else "cant_tell"

        # Verdict chip and plain wording
        st.markdown(
            f"{verdict_chip(chip_state)} &nbsp;&nbsp; "
            f"<span style='color: var(--color-text-muted); font-size: 0.95rem;'>"
            f"Verdict: <strong>{VERDICT_TOKENS[chip_state].label}</strong></span>",
            unsafe_allow_html=True,
        )
        st.write("")

        # Untestable or below minimum plays: plain language, no point estimate
        if not testable or below_min:
            split_label = SPLIT_LABELS.get(selected_split, selected_split.capitalize())
            caveat_callout(
                f"Only {format_int(n_plays)} plays in this cell in the {split_label} "
                f"seasons ({format_int(n_pass)} passes, {format_int(n_run)} runs), "
                "too few to estimate. Try another split or cell.",
                kind="warning",
            )
            card_val = "Not Testable"
            card_interval = None
            q_str = "—"
        elif verdict == "cant_tell":
            # For cant_tell: never show a bare point estimate; always the interval
            card_val = format_ci(ci_lo, ci_hi)
            card_interval = (
                (ci_lo, ci_hi)
                if (ci_lo is not None and ci_hi is not None)
                else None
            )
            q_str = (
                f"{q_bh:.4f}"
                if (q_bh is not None and not math.isnan(q_bh))
                else "—"
            )
        else:
            # Positive or negative verdict (e.g. premium)
            card_val = format_epa(delta_val)
            card_interval = (
                (ci_lo, ci_hi)
                if (ci_lo is not None and ci_hi is not None)
                else None
            )
            q_str = (
                f"{q_bh:.4f}"
                if (q_bh is not None and not math.isnan(q_bh))
                else "—"
            )

        # Layout: one main card on the left, four small cards on the right
        col_main, col_sub = st.columns([1.2, 1.8])
        with col_main:
            delta_label = (
                "Tendency Delta 95% Interval (EPA / play)"
                if verdict == "cant_tell"
                else "Tendency Delta (EPA / play)"
            )
            metric_card(
                label=delta_label,
                value=card_val,
                interval=card_interval,
                n=n_plays if (testable and not below_min) else None,
                help_text=(
                    f"Adjusted p-value (q): {q_str}. "
                    "Controls False Discovery Rate via Benjamini-Hochberg."
                ),
            )

        with col_sub:
            s1, s2 = st.columns(2)
            with s1:
                metric_card(
                    label="Adjusted p-value (q)",
                    value=q_str,
                    help_text=(
                        "Benjamini-Hochberg False Discovery Rate "
                        "adjusted p-value across 27 cells"
                    ),
                )
                metric_card(
                    label="Total Plays Evaluated",
                    value=format_int(n_plays),
                )
            with s2:
                metric_card(
                    label="Regular Season Games",
                    value=format_int(n_games),
                )
                metric_card(
                    label="Pass / Run Balance",
                    value=f"{format_int(n_pass)} / {format_int(n_run)}",
                )

        # Situational pass probability summary (median + IQR) in plain language
        has_phat = phat_med is not None and not math.isnan(phat_med)
        if has_phat:
            med_pct = round(phat_med * 100)
            has_p25 = phat_p25 is not None and not math.isnan(phat_p25)
            has_p75 = phat_p75 is not None and not math.isnan(phat_p75)
            p25_pct = round(phat_p25 * 100) if has_p25 else 0
            p75_pct = round(phat_p75 * 100) if has_p75 else 0
            st.caption(
                f"Typical predicted pass chance: **{med_pct}%** "
                f"(middle half of plays: {p25_pct}%–{p75_pct}%) • "
                f"Sample: **{format_int(n_pass)}** pass plays, "
                f"**{format_int(n_run)}** run plays across "
                f"**{format_int(n_games)}** games."
            )
        else:
            st.caption(
                "Typical predicted pass chance: Not available • "
                f"Sample: **{format_int(n_pass)}** pass plays, "
                f"**{format_int(n_run)}** run plays."
            )

        # "All" split: show this cell's verdict in each single split
        if selected_split == "all":
            split_verdicts = []
            for s in ["early", "tune", "holdout"]:
                rec_s = lookup_cell_record(
                    cells_df,
                    split=s,
                    down=selected_down,
                    dist=selected_dist,
                    zone=selected_zone,
                )
                v_s = (
                    rec_s.get("verdict", "cant_tell")
                    if rec_s
                    else "cant_tell"
                )
                label_s = s.capitalize()
                split_verdicts.append(
                    f"**{label_s}**: {verdict_chip(v_s, small=True)}"
                )
            chips_joined = " | ".join(split_verdicts)
            st.markdown(
                f"Single-split verdicts for this cell: {chips_joined}",
                unsafe_allow_html=True,
            )

        # Cross-split interval plot for this cell
        st.write("")
        st.markdown("##### Cross-Split Agreement for this Cell")
        split_rows = build_cell_split_rows(cells_df, current_cell_id)
        # Check if all four splits are non-testable
        all_untestable = all(r.get("delta") is None for r in split_rows)
        if all_untestable:
            empty_state(
                "All splits for this cell have too few plays to compute an estimate."
            )
        elif split_rows:
            fig = interval_plot(
                rows=split_rows,
                estimate_col="delta",
                ci_lo_col="ci_lo",
                ci_hi_col="ci_hi",
                label_col="label",
                n_col="n_plays",
                state_col="verdict",
                title=f"Tendency delta estimates across splits ({current_cell_id})",
                xaxis_title="Tendency delta (EPA per play)",
                zero_line=True,
                height=260,
            )
            st.plotly_chart(fig, use_container_width=True)

        # -------------------------------------------------------------------
        # 6. Scenario Slider (Deliverable D8.4)
        # -------------------------------------------------------------------
        st.write("")
        st.markdown("##### Interactive Scenario Slider")
        st.caption(
            "Simulate situational pass expectation under hypothetical game state "
            "using the primary pre-snap expectation model (trained on 2016–2023)."
        )

        model, metadata, err_note = load_scenario_model()

        if err_note or model is None or metadata is None:
            # On version mismatch or missing artifact, show note and hide slider
            caveat_callout(
                err_note or "Scenario model unavailable.",
                kind="info",
            )
        else:
            features = metadata.get("features", [])

            col_sl1, col_sl2 = st.columns(2)
            with col_sl1:
                score_diff = st.slider(
                    "Score Differential (Possession team − Defense)",
                    min_value=-28,
                    max_value=28,
                    value=0,
                    step=1,
                    help="Negative means trailing; positive means leading.",
                    key="scenario_score_diff",
                )
            with col_sl2:
                time_remaining = st.slider(
                    "Time Remaining in Half (seconds)",
                    min_value=0,
                    max_value=1800,
                    value=900,
                    step=30,
                    help="1800s = half start; 900s = 15 min; 120s = 2 min warning.",
                    key="scenario_time_remaining",
                )

            # Predict scenario phat
            scenario_phat = predict_scenario_phat(
                model=model,
                features_list=features,
                down=selected_down,
                dist=selected_dist,
                zone=selected_zone,
                score_diff=score_diff,
                half_sec_remaining=time_remaining,
            )

            # Documented fixed defaults list
            rep_ydstogo = DIST_REPRESENTATIVE_YDSTOGO.get(selected_dist, 10)
            rep_yardline = ZONE_REPRESENTATIVE_YARDLINE.get(selected_zone, 50)
            mins = time_remaining // 60
            secs = time_remaining % 60
            dist_desc = DIST_LABELS[selected_dist]
            zone_desc = ZONE_LABELS[selected_zone]

            st.metric(
                "Model-Estimated Pass Probability p̂",
                f"{scenario_phat * 100:.1f}%",
                help="Pre-snap probability estimate from the model.",
            )

            st.caption(
                f"ℹ️ Note: This is a **model estimate** of pre-snap pass probability, "
                "not an observed play-calling rate. Fixed scenario parameters: "
                f"Down {selected_down}, Yards to Go = {rep_ydstogo} ({dist_desc}), "
                f"Yardline = {rep_yardline} ({zone_desc}), "
                f"Quarter = 2, Time = {mins}:{secs:02d}, "
                "Timeouts = 3 (both teams), Point Spread = 0.0."
            )

    else:
        msg = (
            f"No cell record found for '{current_cell_id}' in split '{selected_split}'."
        )
        empty_state(msg)

    render_page_footer()


if __name__ == "__main__":
    render()
