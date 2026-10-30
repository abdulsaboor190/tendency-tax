"""Team Scout view: Empirical-Bayes shrunk tendency tendencies (Phase 9).

Deliverables:
  - D9.1: Controls (split, season, team) with query parameter sync, lean statistics,
    metric cards, and plain-language interpretation.

Controls:
  - Split selector (early, tune, holdout; default 'tune' from PLAN.md)
  - Season selector (seasons present in that split)
  - Searchable team picker (32 NFL franchises)

All controls sync to Streamlit query parameters via app/data.py helpers.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import streamlit as st

from app.data import (
    format_epa,
    format_int,
    format_pct,
    get_query_param,
    get_results_bundle,
    set_query_param,
)
from app.ui.components import caveat_callout, empty_state, metric_card
from app.ui.layout import render_page_footer, render_page_header
from app.ui.theme import inject_css

# ---------------------------------------------------------------------------
# Options & Defaults
# ---------------------------------------------------------------------------

DEFAULT_SPLIT: str = "tune"  # Decided in PLAN.md (Decision 3)
VALID_SPLITS: tuple[str, ...] = ("early", "tune", "holdout")

SPLIT_LABELS: dict[str, str] = {
    "early": "Early (2016–2021)",
    "tune": "Tune (2022–2023) [Default]",
    "holdout": "Holdout (2024–2025)",
}

NFL_TEAMS: tuple[str, ...] = (
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
)


# ---------------------------------------------------------------------------
# State & Query Params Helpers
# ---------------------------------------------------------------------------

def parse_team_scout_state(
    teams_df: pd.DataFrame,
) -> dict[str, Any]:
    """Parse split, season, and team from query params with robust fallback.

    Parameters
    ----------
    teams_df : pd.DataFrame
        DataFrame of teams.parquet.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys 'split' (str), 'season' (int), 'team' (str).
    """
    # 1. Split
    raw_split = str(get_query_param("split", "")).lower()
    available_splits = [s for s in VALID_SPLITS if s in teams_df["split"].unique()]
    if not available_splits:
        available_splits = list(VALID_SPLITS)

    if raw_split in available_splits:
        split_val = raw_split
    elif DEFAULT_SPLIT in available_splits:
        split_val = DEFAULT_SPLIT
    else:
        split_val = available_splits[0]

    # Filter available seasons for this split
    split_sub = teams_df[teams_df["split"] == split_val]
    avail_seasons = sorted(split_sub["season"].unique()) if not split_sub.empty else []

    # 2. Season
    raw_season = get_query_param("season")
    try:
        parsed_season = int(raw_season) if raw_season is not None else None
        if parsed_season in avail_seasons:
            season_val = parsed_season
        elif avail_seasons:
            season_val = avail_seasons[0]
        else:
            season_val = 2022
    except (ValueError, TypeError):
        season_val = avail_seasons[0] if avail_seasons else 2022

    # 3. Team
    avail_teams = (
        sorted(split_sub["team"].unique())
        if not split_sub.empty
        else list(NFL_TEAMS)
    )
    raw_team = str(get_query_param("team", "")).upper()
    if raw_team in avail_teams:
        team_val = raw_team
    elif avail_teams:
        team_val = avail_teams[0]
    else:
        team_val = "ARI"

    return {
        "split": split_val,
        "season": season_val,
        "team": team_val,
    }


def sync_team_scout_query_params(
    split: str, season: int, team: str
) -> None:
    """Sync active team scout selections to Streamlit query params."""
    set_query_param("split", split)
    set_query_param("season", season)
    set_query_param("team", team)


def lookup_team_season_record(
    teams_df: pd.DataFrame,
    split: str,
    season: int,
    team: str,
) -> dict[str, Any] | None:
    """Lookup a single team-season record in teams_df."""
    sub = teams_df[
        (teams_df["split"] == split)
        & (teams_df["season"] == season)
        & (teams_df["team"] == team)
    ]
    if sub.empty:
        return None
    return sub.iloc[0].to_dict()


def build_team_seasons_table(
    teams_df: pd.DataFrame,
    split: str,
    team: str,
) -> pd.DataFrame:
    """Build a summary table of the selected team's seasons for the chosen split."""
    sub = teams_df[
        (teams_df["split"] == split) & (teams_df["team"] == team)
    ].copy()
    if sub.empty:
        return pd.DataFrame()

    sub = sub.sort_values("season")
    table_rows = []
    for _, r in sub.iterrows():
        d_val = r.get("d_raw")
        se_val = r.get("se_raw")
        d_str = (
            format_epa(d_val)
            if (d_val is not None and not math.isnan(d_val))
            else "—"
        )
        se_str = (
            f"±{se_val:.3f}"
            if (se_val is not None and not math.isnan(se_val))
            else "—"
        )

        table_rows.append(
            {
                "Season": int(r["season"]),
                "Total Plays": format_int(r.get("n_plays")),
                "Lean Plays": format_int(r.get("n_lean")),
                "Lean Share": format_pct(r.get("lean_share")),
                "Follow Plays": format_int(r.get("n_follow")),
                "Break Plays": format_int(r.get("n_break")),
                "Follow Rate": format_pct(r.get("follow_rate")),
                "EPA / Follow": format_epa(r.get("epa_follow")),
                "EPA / Break": format_epa(r.get("epa_break")),
                "Raw Difference (EPA)": d_str,
                "Std Error": se_str,
            }
        )

    return pd.DataFrame(table_rows)


# ---------------------------------------------------------------------------
# View Renderer
# ---------------------------------------------------------------------------

def render() -> None:
    """Render the Team Scout view with controls, lean statistics, and history table."""
    inject_css()
    load_res = get_results_bundle()
    render_page_header(load_res)

    st.header("Team Scout")
    st.caption("Team-season situational tendency rates and tendency adherence.")

    # 1. Plain-language definitions
    st.markdown(
        "A **lean play** is one where the situation model gave a pass probability "
        "of at least **65%** (strong pass lean) or at most **35%** (strong run lean); "
        "**followed** means the team made the situational expected call, while "
        "**broke** means the offense called the alternative play type."
    )

    st.divider()

    if not load_res.bundle or load_res.bundle.teams is None:
        empty_state(
            "Team scout results artifact is unavailable. "
            "Run `make secondary` to generate it, then reload."
        )
        render_page_footer()
        return

    teams_df = load_res.bundle.teams

    # 2. Parse state from query parameters
    state = parse_team_scout_state(teams_df)

    # Controls Layout
    st.subheader("Team & Season Controls")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Split Selection**")
        available_splits = [
            s for s in VALID_SPLITS if s in teams_df["split"].unique()
        ]
        if not available_splits:
            available_splits = list(VALID_SPLITS)

        cur_split_idx = (
            available_splits.index(state["split"])
            if state["split"] in available_splits
            else 0
        )
        selected_split = st.segmented_control(
            "Split selection",
            options=available_splits,
            format_func=lambda s: SPLIT_LABELS.get(s, s),
            default=available_splits[cur_split_idx],
            label_visibility="collapsed",
            key="team_scout_split",
        )
        if selected_split is None:
            selected_split = state["split"]

    # Available seasons for the currently active split
    split_sub = teams_df[teams_df["split"] == selected_split]
    avail_seasons = (
        sorted(split_sub["season"].unique())
        if not split_sub.empty
        else [2022]
    )
    cur_season_idx = (
        avail_seasons.index(state["season"])
        if state["season"] in avail_seasons
        else 0
    )

    with c2:
        st.markdown("**Season**")
        selected_season = st.selectbox(
            "Season selection",
            options=avail_seasons,
            index=cur_season_idx,
            label_visibility="collapsed",
            key="team_scout_season",
        )
        if selected_season is None:
            selected_season = state["season"]

    avail_teams = (
        sorted(split_sub["team"].unique())
        if not split_sub.empty
        else list(NFL_TEAMS)
    )
    cur_team_idx = (
        avail_teams.index(state["team"])
        if state["team"] in avail_teams
        else 0
    )

    with c3:
        st.markdown("**Team (Searchable)**")
        selected_team = st.selectbox(
            "Team selection",
            options=avail_teams,
            index=cur_team_idx,
            label_visibility="collapsed",
            key="team_scout_team",
        )
        if selected_team is None:
            selected_team = state["team"]

    # Sync state to query parameters if changed
    if (
        selected_split != state["split"]
        or selected_season != state["season"]
        or selected_team != state["team"]
    ):
        sync_team_scout_query_params(
            selected_split, selected_season, selected_team
        )

    st.divider()

    # 3. Team-Season Lean Statistics (Deliverable D9.1)
    record = lookup_team_season_record(
        teams_df,
        split=selected_split,
        season=selected_season,
        team=selected_team,
    )

    st.subheader(f"Lean Tendency Statistics: {selected_team} ({selected_season})")
    st.caption(
        f"Split: {SPLIT_LABELS.get(selected_split, selected_split)} • "
        f"Franchise: {selected_team} • Season: {selected_season}"
    )

    if record is not None:
        n_plays = record.get("n_plays", 0)
        n_lean = record.get("n_lean", 0)
        lean_share = record.get("lean_share", 0.0)
        n_follow = record.get("n_follow", 0)
        n_break = record.get("n_break", 0)
        follow_rate = record.get("follow_rate", 0.0)
        epa_follow = record.get("epa_follow")
        epa_break = record.get("epa_break")
        d_raw = record.get("d_raw")
        se_raw = record.get("se_raw")
        warn_low_n = record.get("warn_low_n", False)

        has_few_plays = (
            n_break < 2
            or n_follow < 2
            or d_raw is None
            or (isinstance(d_raw, float) and math.isnan(d_raw))
        )

        if has_few_plays:
            empty_state(
                f"Too few lean plays for {selected_team} in {selected_season} "
                f"(break plays N = {n_break}, follow plays N = {n_follow}). "
                "At least 2 plays in each group are required to evaluate a difference."
            )
        else:
            # Play volume & Lean share cards
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                metric_card(
                    label="Total Offensive Plays",
                    value=format_int(n_plays),
                    help_text=(
                        f"Total eligible regular-season snaps for "
                        f"{selected_team} in {selected_season}."
                    ),
                )
            with col_m2:
                metric_card(
                    label="Lean Plays & Share",
                    value=format_pct(lean_share),
                    n=n_lean,
                    help_text=(
                        f"{format_int(n_lean)} plays with situational expectation "
                        "p̂ ≥ 0.65 or p̂ ≤ 0.35."
                    ),
                )
            with col_m3:
                break_rate = 1.0 - follow_rate if follow_rate is not None else 0.0
                metric_card(
                    label="Follow Rate (Tendency Adherence)",
                    value=format_pct(follow_rate),
                    help_text=(
                        f"Followed situational lean on {format_int(n_follow)} plays; "
                        f"broke lean on {format_int(n_break)} plays "
                        f"({format_pct(break_rate)})."
                    ),
                )

            st.write("")

            # Outcome EPA when Following vs Breaking & Raw Difference
            col_e1, col_e2, col_e3 = st.columns(3)
            with col_e1:
                metric_card(
                    label="EPA / Play when Following Lean",
                    value=format_epa(epa_follow),
                    n=n_follow,
                    help_text="Average EPA per play on calls matching the lean.",
                )
            with col_e2:
                metric_card(
                    label="EPA / Play when Breaking Lean",
                    value=format_epa(epa_break),
                    n=n_break,
                    help_text="Average EPA per play on calls against situational lean.",
                )
            with col_e3:
                has_se = se_raw is not None and not math.isnan(se_raw)
                se_str = f"SE = ±{se_raw:.3f}" if has_se else ""
                metric_card(
                    label="Raw Difference: Break − Follow (EPA)",
                    value=format_epa(d_raw),
                    help_text=f"Raw difference in EPA per play. {se_str}",
                )

            # Plain-language interpretation
            st.write("")
            sign_text = "higher" if d_raw > 0 else "lower"
            d_str_val = format_epa(d_raw)
            se_str_val = f"±{se_raw:.3f}"
            st.caption(
                "ℹ️ **Interpretation**: A positive raw difference indicates "
                "that breaking the situational expectation was associated with "
                f"higher EPA per play. For {selected_team} in {selected_season},"
                f" breaking the lean was associated with **{d_str_val}** EPA"
                f" per play ({sign_text} than following) with a standard error"
                f" of **{se_str_val}**"
                f" (N = {n_break} break plays, N = {n_follow} follow plays)."
            )

            if warn_low_n:
                caveat_callout(
                    f"{selected_team} had fewer than 60 plays breaking the lean"
                    f" in {selected_season} (N = {n_break}). Single-season "
                    "unadjusted estimates with small sample sizes carry high "
                    "sampling variability.",
                    kind="warning",
                )

    else:
        msg = (
            f"No team-season record found for {selected_team} ({selected_season}) "
            f"in split '{selected_split}'."
        )
        empty_state(msg)

    st.divider()

    # 4. History Table for Selected Team Across Split
    split_lbl = SPLIT_LABELS.get(selected_split, selected_split)
    st.subheader(f"Season History: {selected_team} ({split_lbl})")
    st.caption("All observed seasons for the selected team in this data split.")

    history_table = build_team_seasons_table(teams_df, selected_split, selected_team)
    if not history_table.empty:
        st.dataframe(history_table, use_container_width=True, hide_index=True)
    else:
        no_rec_msg = (
            f"No records available for {selected_team}"
            f" in split '{selected_split}'."
        )
        st.info(no_rec_msg)

    render_page_footer()


if __name__ == "__main__":
    render()
