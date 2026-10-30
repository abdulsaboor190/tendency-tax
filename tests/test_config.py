"""Tests for src.config configuration constants."""

from src.config import (
    FEATURES,
    LEAN_PASS_THRESHOLD,
    LEAN_RUN_THRESHOLD,
    MIN_PLAYS_PER_CELL,
    SEASONS,
    SPLITS,
    USECOLS,
    ZONE_CUT_POINTS,
)


def test_each_season_appears_in_exactly_one_split():
    """Verify that every season from SEASONS appears in exactly one split."""
    all_split_seasons: list[int] = []
    for split_name, split_seasons in SPLITS.items():
        assert len(split_seasons) == len(set(split_seasons)), (
            f"Duplicate season found inside split '{split_name}'"
        )
        all_split_seasons.extend(split_seasons)

    # All seasons in SEASONS are covered
    assert sorted(all_split_seasons) == sorted(SEASONS), (
        f"Mismatch between SEASONS and SPLITS. "
        f"Missing: {set(SEASONS) - set(all_split_seasons)}, "
        f"Extra: {set(all_split_seasons) - set(SEASONS)}"
    )

    # Exactly one split per season (no duplicates across splits)
    assert len(all_split_seasons) == len(set(all_split_seasons)), (
        "Some seasons appear in multiple splits!"
    )


def test_features_list():
    """Verify exactly 10 features, with no shotgun or no_huddle."""
    assert len(FEATURES) == 10
    assert "shotgun" not in FEATURES
    assert "no_huddle" not in FEATURES


def test_human_decision_placeholders():
    """Verify human decision placeholders are None with TODOs."""
    assert ZONE_CUT_POINTS is None
    assert MIN_PLAYS_PER_CELL is None


def test_lean_thresholds():
    """Verify lean thresholds."""
    assert LEAN_PASS_THRESHOLD == 0.65
    assert LEAN_RUN_THRESHOLD == 0.35


def test_usecols():
    """Verify USECOLS list contains required 28 columns."""
    assert len(USECOLS) == 28
    assert "epa" in USECOLS
    assert "no_play" not in USECOLS
    assert "qb_dropback" in USECOLS
