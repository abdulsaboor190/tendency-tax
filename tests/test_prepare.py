"""Tests for data preparation, filters, labels, buckets, and splits."""

from typing import Any

import pandas as pd
import pytest

from src import config
from src.load import prepare, verify_early_tune_counts


def _make_valid_row(**overrides: Any) -> dict[str, Any]:
    """Create a single play row that satisfies all valid filters."""
    row = {
        "game_id": "2016_01_KC_NE",
        "season": 2016,
        "season_type": "REG",
        "week": 1,
        "home_team": "NE",
        "posteam": "NE",
        "defteam": "KC",
        "play_type": "pass",
        "down": 1,
        "ydstogo": 10,
        "yardline_100": 50,
        "qtr": 1,
        "game_seconds_remaining": 3500,
        "half_seconds_remaining": 1700,
        "score_differential": 0,
        "posteam_timeouts_remaining": 3,
        "defteam_timeouts_remaining": 3,
        "spread_line": 3.5,  # NE favored by 3.5
        "qb_dropback": 1,
        "qb_kneel": 0,
        "qb_spike": 0,
        "epa": 0.15,
        "success": 1,
        "wp": 0.55,
        "xpass": 0.60,
        "shotgun": 1,
        "no_huddle": 0,
        "penalty": 0,
    }
    row.update(overrides)
    return row


def test_posteam_spread_sign():
    """Verify posteam_spread sign: positive when possession team is favored."""
    # Case 1: Home possession, home favored -> posteam_spread = +3.5
    row_home_fav = _make_valid_row(
        home_team="NE", posteam="NE", spread_line=3.5
    )
    # Case 2: Away possession, home favored -> posteam_spread = -3.5
    row_away_dog = _make_valid_row(
        home_team="NE", posteam="KC", spread_line=3.5
    )
    # Case 3: Away possession, away favored (-4.0) -> posteam_spread = +4.0
    row_away_fav = _make_valid_row(
        home_team="NE", posteam="KC", spread_line=-4.0
    )

    df = pd.DataFrame([row_home_fav, row_away_dog, row_away_fav])
    prepared = prepare(df)

    assert len(prepared) == 3
    assert prepared.loc[0, "posteam_spread"] == 3.5
    assert prepared.loc[1, "posteam_spread"] == -3.5
    assert prepared.loc[2, "posteam_spread"] == 4.0


def test_call_pass_taken_from_qb_dropback_not_play_type():
    """Verify call_pass is qb_dropback, treating scrambles as pass calls."""
    # Traditional pass
    row_pass = _make_valid_row(play_type="pass", qb_dropback=1)
    # Scramble: play_type is "run" but qb_dropback is 1 -> MUST be call_pass = 1
    row_scramble = _make_valid_row(play_type="run", qb_dropback=1)
    # Traditional designed run
    row_run = _make_valid_row(play_type="run", qb_dropback=0)

    df = pd.DataFrame([row_pass, row_scramble, row_run])
    prepared = prepare(df)

    assert len(prepared) == 3
    assert prepared.loc[0, "call_pass"] == 1
    assert prepared.loc[1, "call_pass"] == 1  # Scramble is a pass call!
    assert prepared.loc[2, "call_pass"] == 0  # Designed run is 0


def test_each_filter_removes_exact_rows():
    """Verify that each filter in prepare() removes exactly the intended rows."""
    test_cases = [
        # (row_override, should_keep, reason)
        ({}, True, "Baseline valid row"),
        ({"season_type": "POST"}, False, "Exclude postseason"),
        ({"play_type": "punt"}, False, "play_type not in {pass, run}"),
        ({"down": 4}, False, "4th down out of scope"),
        ({"epa": None}, False, "epa null"),
        ({"posteam": None}, False, "posteam null"),
        ({"xpass": None}, False, "xpass null"),
        ({"qtr": 5}, False, "qtr 5 overtime out of scope"),
        ({"wp": 0.04}, False, "wp below 0.05"),
        ({"wp": 0.96}, False, "wp above 0.95"),
        ({"qb_kneel": 1}, False, "qb_kneel"),
        ({"qb_spike": 1}, False, "qb_spike"),
        (
            {"qtr": 2, "half_seconds_remaining": 120},
            False,
            "qtr 2 2-minute drill",
        ),
        (
            {"qtr": 4, "half_seconds_remaining": 30},
            False,
            "qtr 4 2-minute drill",
        ),
        (
            {"qtr": 2, "half_seconds_remaining": 121},
            True,
            "qtr 2 before 2-minute drill",
        ),
        (
            {"qtr": 1, "half_seconds_remaining": 60},
            True,
            "qtr 1 under 2 minutes (not excluded)",
        ),
    ]

    rows = []
    expected_keeps = []
    for override, keep, _reason in test_cases:
        rows.append(_make_valid_row(**override))
        expected_keeps.append(keep)

    df = pd.DataFrame(rows)
    prepared = prepare(df)

    assert len(prepared) == sum(expected_keeps)


def test_no_null_in_constructed_columns():
    """Verify no nulls exist in constructed columns and verify bucket shapes."""
    rows = [
        # short yardage, red zone
        _make_valid_row(ydstogo=2, yardline_100=10, down=1, season=2016),
        # medium yardage, midfield
        _make_valid_row(ydstogo=6, yardline_100=50, down=2, season=2022),
        # long yardage, backed up
        _make_valid_row(ydstogo=12, yardline_100=85, down=3, season=2023),
    ]
    df = pd.DataFrame(rows)
    prepared = prepare(df)

    cols_to_check = [
        "call_pass",
        "posteam_spread",
        "dist_bucket",
        "dd_bucket",
        "zone",
        "cell_id",
        "split",
    ]
    for col in cols_to_check:
        assert prepared[col].notna().all(), f"Null found in column '{col}'"

    # Row 0 assertions
    assert prepared.loc[0, "dist_bucket"] == "short"
    assert prepared.loc[0, "zone"] == "red_zone"
    assert prepared.loc[0, "dd_bucket"] == "1_short"
    assert prepared.loc[0, "cell_id"] == "1_short_red_zone"
    assert prepared.loc[0, "split"] == "early"

    # Row 1 assertions
    assert prepared.loc[1, "dist_bucket"] == "medium"
    assert prepared.loc[1, "zone"] == "midfield"
    assert prepared.loc[1, "dd_bucket"] == "2_medium"
    assert prepared.loc[1, "cell_id"] == "2_medium_midfield"
    assert prepared.loc[1, "split"] == "tune"

    # Row 2 assertions
    assert prepared.loc[2, "dist_bucket"] == "long"
    assert prepared.loc[2, "zone"] == "backed_up"
    assert prepared.loc[2, "dd_bucket"] == "3_long"
    assert prepared.loc[2, "cell_id"] == "3_long_backed_up"
    assert prepared.loc[2, "split"] == "tune"


def test_zone_cut_points_none_raises_value_error(monkeypatch: pytest.MonkeyPatch):
    """Verify prepare raises ValueError if config.ZONE_CUT_POINTS is None."""
    monkeypatch.setattr(config, "ZONE_CUT_POINTS", None)
    df = pd.DataFrame([_make_valid_row()])
    with pytest.raises(ValueError) as exc:
        prepare(df)
    assert "ZONE_CUT_POINTS is None" in str(exc.value)


def test_verify_early_tune_counts_matches_exact_numbers():
    """Verify real data play and game counts match exact specification numbers."""
    counts = verify_early_tune_counts()

    expected_per_season = {
        2016: 24152,
        2017: 23692,
        2018: 23188,
        2019: 23582,
        2020: 23879,
        2021: 24453,
        2022: 25190,
        2023: 24932,
    }

    assert counts["season_counts"] == expected_per_season
    assert counts["early_plays"] == 142946
    assert counts["early_games"] == 1552
    assert counts["tune_plays"] == 50122
    assert counts["tune_games"] == 543
