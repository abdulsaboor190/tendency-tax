"""Tests for src.config configuration constants."""

from src.config import (
    FEATURES,
    FORBIDDEN_FEATURES,
    LEAN_PASS_THRESHOLD,
    LEAN_RUN_THRESHOLD,
    MIN_PLAYS_PER_CELL,
    MODEL_PARAMS,
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


def test_human_decisions_and_placeholders():
    """Verify ZONE_CUT_POINTS is set and MIN_PLAYS_PER_CELL is None."""
    assert ZONE_CUT_POINTS == (20, 80)
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


def test_model_params_and_forbidden_features():
    """Verify MODEL_PARAMS and FORBIDDEN_FEATURES configuration."""
    assert isinstance(MODEL_PARAMS, dict)
    assert "max_depth" in MODEL_PARAMS
    assert "learning_rate" in MODEL_PARAMS
    assert "l2_regularization" in MODEL_PARAMS
    assert "max_iter" in MODEL_PARAMS
    assert 3 <= MODEL_PARAMS["max_depth"] <= 5

    assert "epa" in FORBIDDEN_FEATURES
    assert "xpass" in FORBIDDEN_FEATURES
    assert "success" in FORBIDDEN_FEATURES
    assert "wp" in FORBIDDEN_FEATURES
    assert "shotgun" in FORBIDDEN_FEATURES
    assert "no_huddle" in FORBIDDEN_FEATURES
