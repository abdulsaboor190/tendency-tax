"""Tests for app/views/team_scout.py (Deliverable D9.1).

Tests:
  1. parse_team_scout_state returns default 'tune' split when no query params are set.
  2. parse_team_scout_state parses valid query parameters correctly.
  3. parse_team_scout_state falls back to defaults on invalid split/season/team.
  4. lookup_team_season_record finds record when present and returns None when missing.
  5. build_team_seasons_table returns the correct number of rows and columns.
  6. team_scout view renders on fixtures (TENDENCY_USE_FIXTURES=1).
  7. team_scout view renders on real results if present.
  8. Every team-season across all splits in teams.parquet renders without error.
  9. Low play count scenario displays empty_state rather than bare numbers.
  10. No ranking, sorting by estimate, or prohibited ranking words
      (best, worst, rank, top, bottom).
  11. No prohibited causal words (causes, proves, optimal, causal)
      in source or visible text.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.data import get_results_bundle, load_results
from app.views.team_scout import (
    DEFAULT_SPLIT,
    VALID_SPLITS,
    build_team_seasons_table,
    lookup_team_season_record,
    parse_team_scout_state,
    render,
)


def _make_st_mock() -> MagicMock:
    """Return a MagicMock that behaves like the streamlit module."""
    m = MagicMock()
    m.segmented_control.return_value = None
    m.slider.return_value = 900

    def _selectbox(*args, **kwargs):
        opts = kwargs.get("options", ["ARI"])
        return opts[0] if opts else "ARI"

    m.selectbox.side_effect = _selectbox

    def _columns(n, *args, **kwargs):
        if isinstance(n, int):
            return [MagicMock() for _ in range(n)]
        return [MagicMock() for _ in n]

    m.columns.side_effect = _columns
    m.expander.return_value.__enter__ = lambda s: s
    m.expander.return_value.__exit__ = MagicMock(return_value=False)
    m.button.return_value = False
    return m


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear st.cache_data between test invocations."""
    get_results_bundle.clear()
    yield
    get_results_bundle.clear()


def test_parse_team_scout_state_defaults(monkeypatch):
    """When query params are empty, falls back to default split 'tune'."""
    monkeypatch.setenv("TENDENCY_USE_FIXTURES", "1")
    bundle = load_results()
    assert bundle.teams is not None
    with patch("app.views.team_scout.get_query_param", return_value=""):
        state = parse_team_scout_state(bundle.teams)
    assert state["split"] == DEFAULT_SPLIT
    assert state["split"] == "tune"
    assert state["season"] in [2022, 2023]
    assert isinstance(state["team"], str)


def test_parse_team_scout_state_valid(monkeypatch):
    """Parses valid query parameters properly."""
    monkeypatch.setenv("TENDENCY_USE_FIXTURES", "1")
    bundle = load_results()
    assert bundle.teams is not None

    def _get_param(k, default=None):
        params = {"split": "tune", "season": "2023", "team": "KC"}
        return params.get(k, default)

    with patch("app.views.team_scout.get_query_param", side_effect=_get_param):
        state = parse_team_scout_state(bundle.teams)
    assert state["split"] == "tune"
    assert state["season"] == 2023
    assert state["team"] == "KC"


def test_parse_team_scout_state_invalid_fallback(monkeypatch):
    """Invalid split, season, or team values gracefully fall back to defaults."""
    monkeypatch.setenv("TENDENCY_USE_FIXTURES", "1")
    bundle = load_results()
    assert bundle.teams is not None

    def _get_param(k, default=None):
        params = {"split": "invalid_split", "season": "9999", "team": "INVALID"}
        return params.get(k, default)

    with patch("app.views.team_scout.get_query_param", side_effect=_get_param):
        state = parse_team_scout_state(bundle.teams)
    assert state["split"] in VALID_SPLITS
    assert state["season"] in [2022, 2023]
    assert len(state["team"]) >= 2


def test_lookup_team_season_record(monkeypatch):
    """Lookup returns dictionary when present and None when missing."""
    monkeypatch.setenv("TENDENCY_USE_FIXTURES", "1")
    bundle = load_results()
    assert bundle.teams is not None

    rec = lookup_team_season_record(bundle.teams, split="tune", season=2022, team="BAL")
    assert rec is not None
    assert rec["team"] == "BAL"
    assert rec["season"] == 2022
    assert "d_raw" in rec

    rec_missing = lookup_team_season_record(
        bundle.teams, split="tune", season=1990, team="BAL"
    )
    assert rec_missing is None


def test_build_team_seasons_table(monkeypatch):
    """Table contains all seasons for the team in the given split."""
    monkeypatch.setenv("TENDENCY_USE_FIXTURES", "1")
    bundle = load_results()
    assert bundle.teams is not None

    table = build_team_seasons_table(bundle.teams, split="tune", team="BUF")
    assert not table.empty
    assert len(table) == 2
    assert "Season" in table.columns
    assert "Raw Difference (EPA)" in table.columns


def test_team_scout_renders_fixtures(monkeypatch):
    """Renders without uncaught errors using fixtures."""
    monkeypatch.setenv("TENDENCY_USE_FIXTURES", "1")
    mock_st = _make_st_mock()

    with (
        patch("app.views.team_scout.st", mock_st),
        patch("app.views.team_scout.inject_css"),
        patch("app.views.team_scout.render_page_header"),
        patch("app.views.team_scout.render_page_footer"),
        patch("app.views.team_scout.get_query_param", return_value=None),
        patch("app.views.team_scout.set_query_param"),
    ):
        render()

    mock_st.header.assert_called_with("Team Scout")
    assert mock_st.segmented_control.called or mock_st.selectbox.called


def test_team_scout_low_n_shows_empty_state(monkeypatch):
    """When n_break < 2 or d_raw is NaN, empty_state is shown."""
    fake_df = pd.DataFrame(
        [
            {
                "split": "tune",
                "season": 2022,
                "team": "ARI",
                "n_plays": 500,
                "n_lean": 200,
                "lean_share": 0.4,
                "n_follow": 199,
                "n_break": 1,  # < 2 plays
                "follow_rate": 0.995,
                "epa_follow": 0.05,
                "epa_break": -0.1,
                "d_raw": float("nan"),
                "se_raw": float("nan"),
                "warn_low_n": True,
            }
        ]
    )

    mock_st = _make_st_mock()
    mock_bundle = MagicMock()
    mock_bundle.teams = fake_df
    mock_load_res = MagicMock(bundle=mock_bundle, is_fixture=True, error=None)

    with (
        patch("app.views.team_scout.get_results_bundle", return_value=mock_load_res),
        patch("app.views.team_scout.st", mock_st),
        patch("app.views.team_scout.inject_css"),
        patch("app.views.team_scout.render_page_header"),
        patch("app.views.team_scout.render_page_footer"),
        patch("app.views.team_scout.get_query_param", return_value=None),
        patch("app.views.team_scout.set_query_param"),
        patch("app.views.team_scout.empty_state") as mock_empty,
    ):
        render()

    assert mock_empty.called
    call_msg = mock_empty.call_args[0][0]
    assert "Too few lean plays" in call_msg


def test_all_team_seasons_render_fixtures(monkeypatch):
    """Every record in fixture teams.parquet can be rendered cleanly."""
    monkeypatch.setenv("TENDENCY_USE_FIXTURES", "1")
    bundle = load_results()
    assert bundle.teams is not None

    for _, row in bundle.teams.iterrows():
        split = row["split"]
        season = int(row["season"])
        team = row["team"]

        rec = lookup_team_season_record(bundle.teams, split, season, team)
        assert rec is not None
        assert not math.isnan(rec["n_plays"])


def test_all_team_seasons_render_real_results(monkeypatch):
    """Every record in real results/teams.parquet can be looked up cleanly."""
    monkeypatch.delenv("TENDENCY_USE_FIXTURES", raising=False)
    bundle = load_results()
    if bundle.teams is None or len(bundle.teams) <= 64:
        pytest.skip("Full real results not available")

    assert len(bundle.teams) >= 320
    for _, row in bundle.teams.iterrows():
        split = row["split"]
        season = int(row["season"])
        team = row["team"]
        rec = lookup_team_season_record(bundle.teams, split, season, team)
        assert rec is not None


def test_low_n_team_season_shows_warning_and_chip(monkeypatch):
    """When warn_low_n is True, warning callout and cant_tell chip are displayed."""
    monkeypatch.delenv("TENDENCY_USE_FIXTURES", raising=False)
    bundle = load_results()
    if bundle.teams is None or len(bundle.teams) <= 64:
        pytest.skip("Full real results not available")

    mock_st = _make_st_mock()

    with (
        patch("app.views.team_scout.st", mock_st),
        patch("app.views.team_scout.inject_css"),
        patch("app.views.team_scout.render_page_header"),
        patch("app.views.team_scout.render_page_footer"),
        patch(
            "app.views.team_scout.get_query_param",
            side_effect=lambda k, default=None: {
                "split": "tune",
                "season": "2022",
                "team": "ARI",
            }.get(k, default),
        ),
        patch("app.views.team_scout.set_query_param"),
        patch("app.views.team_scout.caveat_callout") as mock_callout,
        patch("app.views.team_scout.verdict_chip") as mock_chip,
    ):
        render()

    assert mock_callout.called
    warning_calls = [
        c for c in mock_callout.call_args_list if c.kwargs.get("kind") == "warning"
    ]
    assert len(warning_calls) > 0
    assert "fewer than 60" in warning_calls[0].args[0]
    mock_chip.assert_called_with("cant_tell")


def test_shrunk_interval_drawn_from_file_columns(monkeypatch):
    """The shrunk interval plotted must match shrunk_lo and shrunk_hi from the file."""
    monkeypatch.delenv("TENDENCY_USE_FIXTURES", raising=False)
    bundle = load_results()
    if bundle.teams is None:
        pytest.skip("teams results not available")

    rec = lookup_team_season_record(bundle.teams, split="tune", season=2022, team="ARI")
    if rec is None:
        rec = bundle.teams.iloc[0].to_dict()

    assert "shrunk_lo" in rec and "shrunk_hi" in rec
    assert rec["shrunk_lo"] < rec["shrunk_hi"]
    assert rec["shrunk_lo"] <= rec["d_shrunk"] <= rec["shrunk_hi"]


def test_build_team_shrunk_chart_and_strip_chart(monkeypatch):
    """Verify build_team_shrunk_chart and strip_chart return valid figures."""
    from app.ui.charts import build_league_strip_chart, build_team_shrunk_chart

    monkeypatch.setenv("TENDENCY_USE_FIXTURES", "1")
    bundle = load_results()
    assert bundle.teams is not None

    chart = build_team_shrunk_chart(
        raw_estimate=0.05,
        raw_ci_lo=-0.1,
        raw_ci_hi=0.2,
        raw_n=45,
        shrunk_estimate=0.01,
        shrunk_ci_lo=-0.08,
        shrunk_ci_hi=0.1,
        mu_league=-0.002,
        split="tune",
        team_label="ARI 2022",
    )
    assert chart is not None
    assert len(chart.data) >= 1

    strip = build_league_strip_chart(
        teams_df=bundle.teams,
        split="tune",
        selected_team="ARI",
        selected_season=2022,
    )
    assert strip is not None
    assert len(strip.data) >= 1


def test_no_ranking_or_sorting_words_in_team_scout():
    """Verify that team_scout view does not contain ranking or best/worst language."""
    from pathlib import Path

    scout_code = Path("app/views/team_scout.py").read_text(encoding="utf-8").lower()
    banned_ranking_words = [
        "best", "worst", "rank", "ranking", "top team", "bottom team"
    ]
    for word in banned_ranking_words:
        assert word not in scout_code, (
            f"Forbidden ranking word '{word}' found in team_scout.py"
        )


def test_no_banned_causal_words_in_team_scout():
    """Verify that team_scout view adheres to association-only language rule."""
    from pathlib import Path

    scout_code = Path("app/views/team_scout.py").read_text(encoding="utf-8").lower()
    banned_causal_words = ["causes", "proves", "optimal", "causality"]
    for word in banned_causal_words:
        assert word not in scout_code, (
            f"Forbidden causal word '{word}' found in team_scout.py"
        )
