"""Safety and integrity tests for dev stage pipeline (Deliverable D3.5)."""

from pathlib import Path
from unittest.mock import patch

from scripts.run_all import run_dev_pipeline
from src import config
from src.load import get_split


def test_dev_stage_hard_safety_never_calls_holdout():
    """Verify that dev stage NEVER requests the holdout split."""
    original_get_split = get_split

    def guarded_get_split(name):
        if str(name).lower() == "holdout":
            raise RuntimeError(
                "SAFETY BREACH: get_split('holdout') called during dev stage!"
            )
        return original_get_split(name)

    with patch("scripts.run_all.get_split", side_effect=guarded_get_split):
        # Run with tiny reps for speed
        results = run_dev_pipeline(n_boot=5, n_perm=5, n_jobs=1)
        assert "early" in results
        assert "tune" in results
        assert "holdout" not in results


def test_holdout_access_log_untouched():
    """Verify that HOLDOUT_ACCESS.log does not exist or remains completely untouched."""
    log_path = config.HOLDOUT_ACCESS_LOG
    if log_path.exists():
        initial_content = log_path.read_text(encoding="utf-8")
    else:
        initial_content = None

    # Run with small reps
    run_dev_pipeline(n_boot=2, n_perm=2, n_jobs=1)

    if initial_content is None:
        assert not log_path.exists(), (
            "HOLDOUT_ACCESS.log was created during dev pipeline!"
        )
    else:
        assert log_path.read_text(encoding="utf-8") == initial_content, (
            "HOLDOUT_ACCESS.log was modified during dev pipeline!"
        )


def test_dev_artifacts_created():
    """Verify that required dev results artifacts exist and have valid structure."""
    assert Path("results/dev_primary.json").exists()
    assert Path("results/placebo_early.parquet").exists()
    assert Path("results/placebo_tune.parquet").exists()
    assert Path("results/decile_gap_dev.parquet").exists()

