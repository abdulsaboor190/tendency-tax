"""Tests for the confirmatory stage of scripts/run_all.py (Deliverable D4.2).

Ensures all preconditions and pipeline steps work with synthetic or mocked data.
NEVER executes against the real holdout data or sets TENDENCY_ALLOW_HOLDOUT.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from scripts.run_all import run_confirm_pipeline
from src import config
from src.load import find_season_file


def _has_raw_data() -> bool:
    """Check if raw data is available for pipeline execution."""
    try:
        find_season_file(2016)
        return True
    except FileNotFoundError:
        return False


def _make_mock_holdout_split(n_games=30, plays_per_game=25, seed=123):
    """Generate mock holdout dataset."""
    rng = np.random.default_rng(seed)
    n = n_games * plays_per_game
    game_ids = np.repeat([f"2024_{i:03d}" for i in range(n_games)], plays_per_game)
    buckets = ["1_short", "1_med", "1_long", "2_med", "3_med", "3_long"]

    call = rng.binomial(1, 0.55, size=n).astype(float)
    dd_bucket = rng.choice(buckets, size=n)
    cell_ids = [f"{b}_midfield" for b in dd_bucket]

    features = {
        "down": rng.choice([1, 2, 3], size=n),
        "ydstogo": rng.integers(1, 15, size=n),
        "yardline_100": rng.integers(10, 90, size=n),
        "qtr": rng.choice([1, 2, 3, 4], size=n),
        "game_seconds_remaining": rng.integers(120, 3600, size=n),
        "half_seconds_remaining": rng.integers(120, 1800, size=n),
        "score_differential": rng.integers(-14, 14, size=n),
        "posteam_timeouts_remaining": rng.choice([0, 1, 2, 3], size=n),
        "defteam_timeouts_remaining": rng.choice([0, 1, 2, 3], size=n),
        "posteam_spread": rng.normal(0, 3, size=n),
    }

    epa = 0.1 * call + rng.normal(0, 0.4, size=n)

    df_dict = {
        "game_id": game_ids,
        "season": 2024,
        "call_pass": call,
        "epa": epa,
        "dd_bucket": dd_bucket,
        "cell_id": cell_ids,
    }
    df_dict.update(features)
    return pd.DataFrame(df_dict)


def test_confirm_refuses_when_env_var_missing(monkeypatch):
    """Stage refuses when TENDENCY_ALLOW_HOLDOUT != '1'."""
    monkeypatch.delenv(config.HOLDOUT_ENV_VAR, raising=False)
    with pytest.raises(RuntimeError, match="Precondition 1 failed"):
        run_confirm_pipeline()


def test_confirm_refuses_when_tree_is_dirty(monkeypatch):
    """Stage refuses when git working tree is dirty."""
    monkeypatch.setenv(config.HOLDOUT_ENV_VAR, "1")

    with patch("subprocess.run") as mock_run:
        # Mock git status --porcelain returning dirty
        mock_run.return_value.stdout = " M some_file.py\n"
        with pytest.raises(RuntimeError, match="Working tree is dirty"):
            run_confirm_pipeline(skip_git_checks=False)


def test_confirm_refuses_when_primary_json_already_exists(tmp_path, monkeypatch):
    """Stage refuses without --force if primary.json already exists."""
    monkeypatch.setenv(config.HOLDOUT_ENV_VAR, "1")

    # Create dummy existing primary.json in tmp_path
    existing_primary = tmp_path / "primary.json"
    existing_primary.write_text("{}", encoding="utf-8")

    def custom_path_router(path_arg):
        p_str = str(path_arg)
        if "results/primary.json" in p_str or "results\\primary.json" in p_str:
            return existing_primary
        return Path(path_arg)

    with patch("scripts.run_all.Path", side_effect=custom_path_router):
        with pytest.raises(RuntimeError, match="already exists"):
            run_confirm_pipeline(
                force=False,
                skip_git_checks=True,
                prompt_input_func=lambda _: "RUN",
            )


def test_confirm_refuses_when_interactive_confirmation_aborted(tmp_path, monkeypatch):
    """Stage refuses when user input is not 'RUN'."""
    monkeypatch.setenv(config.HOLDOUT_ENV_VAR, "1")

    # Non-existent primary.json
    dummy_primary = tmp_path / "primary.json"

    def custom_path_router(path_arg):
        p_str = str(path_arg)
        if "results/primary.json" in p_str or "results\\primary.json" in p_str:
            return dummy_primary
        return Path(path_arg)

    with patch("scripts.run_all.Path", side_effect=custom_path_router):
        with pytest.raises(RuntimeError, match="Confirmation aborted"):
            run_confirm_pipeline(
                skip_git_checks=True,
                prompt_input_func=lambda _: "NO",
            )


def test_confirm_execution_on_synthetic_data(monkeypatch, tmp_path):
    """Test full execution on synthetic mock holdout data."""
    if not _has_raw_data():
        pytest.skip("Raw play-by-play data not present in data/raw/")
    monkeypatch.setenv(config.HOLDOUT_ENV_VAR, "1")

    mock_holdout = _make_mock_holdout_split(n_games=15, plays_per_game=20, seed=42)

    primary_tmp = tmp_path / "primary.json"
    placebo_tmp = tmp_path / "placebo_holdout.parquet"
    decile_tmp = tmp_path / "decile_gap_holdout.parquet"
    run_log_tmp = tmp_path / "RUN_LOG.md"

    def custom_path_router(path_arg):
        p_str = str(path_arg)
        if "results/primary.json" in p_str or "results\\primary.json" in p_str:
            return primary_tmp
        if "results/placebo_holdout.parquet" in p_str:
            return placebo_tmp
        if "results/decile_gap_holdout.parquet" in p_str:
            return decile_tmp
        if "results/RUN_LOG.md" in p_str:
            return run_log_tmp
        return Path(path_arg)

    with patch("scripts.run_all.Path", side_effect=custom_path_router), \
         patch("scripts.run_all.get_split", return_value=mock_holdout):

        res = run_confirm_pipeline(
            force=False,
            prompt_input_func=lambda _: "RUN",
            n_boot=10,
            n_perm=10,
            n_jobs=1,
            skip_git_checks=True,
        )

        # Check all required top-level keys
        assert "timestamp" in res
        assert "git_commit" in res
        assert "plan_commit" in res
        assert "package_versions" in res
        assert "simulation_parameters" in res
        assert "early" in res
        assert "tune" in res
        assert "holdout" in res
        assert "verdict" in res

        # Check holdout structure
        holdout = res["holdout"]
        assert "n_plays" in holdout
        assert "n_games" in holdout
        assert "spec_b" in holdout
        assert "spec_a_secondary" in holdout
        assert "predictive_scores" in holdout

        spec_b = holdout["spec_b"]
        assert "delta" in spec_b
        assert "delta_ci_95" in spec_b
        assert "delta_se" in spec_b
        assert "p_value_delta" in spec_b
        assert "delta_run" in spec_b
        assert "delta_pass" in spec_b
        assert "placebo" in spec_b

        # Check verdict dict structure
        v = res["verdict"]
        assert "state" in v
        assert "conditions" in v
        assert "failed" in v
        assert "direction" in v
        assert "sentence" in v

        # Check files were created
        assert primary_tmp.exists()
        assert placebo_tmp.exists()
        assert decile_tmp.exists()
        assert run_log_tmp.exists()
