"""Tests for results schema, bundle loader, and fixtures (Deliverable D5.1)."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.schema import (
    ResultsBundle,
    ResultsError,
    generate_manifest,
    load_results,
    validate_parquet_df,
)


def test_load_results_fixtures():
    """Verify load_results works on test fixtures."""
    bundle = load_results(source="fixtures")
    assert isinstance(bundle, ResultsBundle)
    assert bundle.source == "fixtures"
    assert bundle.primary is not None
    assert bundle.dev_primary is not None
    assert len(bundle.model_scores) == 6
    assert bundle.model_diagnostics is not None
    assert bundle.power is not None

    # Parquet checks
    assert bundle.decile_gap_dev is not None
    assert bundle.decile_gap_holdout is not None
    assert bundle.placebo_early is not None
    assert bundle.placebo_tune is not None
    assert bundle.placebo_holdout is not None
    assert bundle.calibration_early is not None
    assert bundle.calibration_tune is not None
    assert bundle.phat_by_bucket is not None
    assert bundle.robustness is not None
    assert bundle.cells is not None
    assert bundle.teams is not None

    # Check shapes of new Phase 5 fake datasets
    assert len(bundle.cells) == 27 * 4
    assert len(bundle.teams) == 64


def test_load_results_real():
    """Verify load_results works on real existing results directory."""
    bundle = load_results(source="real")
    assert isinstance(bundle, ResultsBundle)
    assert bundle.source == "real"
    assert bundle.primary is not None
    assert bundle.dev_primary is not None
    assert len(bundle.model_scores) == 6
    assert bundle.power is not None
    assert bundle.decile_gap_dev is not None
    assert bundle.phat_by_bucket is not None


def test_broken_parquet_missing_column(tmp_path):
    """Verify missing required column raises ResultsError."""
    # Break decile_gap by omitting epa_gap_pass_minus_run
    df = pd.DataFrame({
        "split": ["early"],
        "decile": [1],
        "phat_min": [0.0],
        "phat_max": [0.1],
        "n_pass": [10],
        "n_run": [10],
        "mean_epa_pass": [0.1],
        "mean_epa_run": [0.0],
        # "epa_gap_pass_minus_run" is missing
        "ci_95_low": [-0.1],
        "ci_95_high": [0.1],
    })
    with pytest.raises(ResultsError, match="Missing required columns"):
        validate_parquet_df(df, schema_name="decile_gap", filename="bad_decile.parquet")


def test_broken_parquet_wrong_type(tmp_path):
    """Verify wrong column type raises ResultsError."""
    df = pd.DataFrame({
        "split": ["early"],
        "decile": ["not_an_int"],  # Wrong type
        "phat_min": [0.0],
        "phat_max": [0.1],
        "n_pass": [10],
        "n_run": [10],
        "mean_epa_pass": [0.1],
        "mean_epa_run": [0.0],
        "epa_gap_pass_minus_run": [0.1],
        "ci_95_low": [-0.1],
        "ci_95_high": [0.1],
    })
    with pytest.raises(ResultsError, match="invalid dtype"):
        validate_parquet_df(df, schema_name="decile_gap", filename="bad_decile.parquet")


def test_broken_cell_verdict(tmp_path):
    """Verify verdict outside the allowed four raises ResultsError."""
    df = pd.read_parquet("tests/fixtures/results/cells.parquet").copy()
    df.loc[0, "verdict"] = "super_significant"  # not in ALLOWED_CELL_VERDICTS
    with pytest.raises(ResultsError, match="Invalid cell verdict values"):
        validate_parquet_df(df, schema_name="cells", filename="cells.parquet")


def test_broken_json_missing_key(tmp_path):
    """Verify JSON with missing required key raises ResultsError."""
    # Create invalid power.json missing 'calibration'
    bad_dir = tmp_path / "bad_results"
    bad_dir.mkdir()
    # Copy valid files except power
    for p in Path("tests/fixtures/results").glob("*"):
        if p.is_file() and p.name != "power.json":
            (bad_dir / p.name).write_bytes(p.read_bytes())
    # Bad power
    (bad_dir / "power.json").write_text(
        json.dumps({"is_fixture": True}), encoding="utf-8"
    )

    with pytest.raises(ResultsError, match="Failed schema validation"):
        load_results(base_dir=bad_dir)


def test_manifest_generation(tmp_path):
    """Verify manifest records all metadata and saves to MANIFEST.json."""
    manifest_path = tmp_path / "MANIFEST.json"
    manifest = generate_manifest(
        results_dir=Path("tests/fixtures/results"),
        output_path=manifest_path,
    )
    assert "schema_version" in manifest
    assert "git_hash" in manifest
    assert "files" in manifest
    assert manifest_path.exists()

    files = manifest["files"]
    assert "primary.json" in files
    assert "cells.parquet" in files
    assert files["cells.parquet"]["row_count"] == 27 * 4
    assert files["cells.parquet"]["file_size_bytes"] > 0
    assert "sha256" in files["cells.parquet"]
