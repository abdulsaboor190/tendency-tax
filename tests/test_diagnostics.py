"""Tests for predictability diagnostics module."""

import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.diagnostics import compute_diagnostics_for_split


def test_within_bucket_sd_computed_within_buckets_not_overall():
    """Verify that SD is computed within each dd_bucket, not as the overall SD."""
    # Synthetic dataframe with 2 buckets having very different within-bucket variances
    # Bucket A: constant phat = 0.5 (within SD = 0)
    # Bucket B: values [0.1, 0.9] (within SD ~ 0.565)
    # Overall values: [0.5, 0.5, 0.1, 0.9] (overall SD ~ 0.33)
    df = pd.DataFrame(
        {
            "dd_bucket": ["A", "A", "B", "B"],
            "phat": [0.5, 0.5, 0.1, 0.9],
            "call_pass": [1, 0, 0, 1],
        }
    )

    diag, bucket_stats = compute_diagnostics_for_split(df, "test_split")

    overall_sd = float(np.std(df["phat"], ddof=1))
    bucket_a_sd = diag["within_bucket_sds"]["A"]
    bucket_b_sd = diag["within_bucket_sds"]["B"]

    # Bucket A SD must be 0, Bucket B SD must be ~0.565685
    assert bucket_a_sd == pytest.approx(0.0)
    assert bucket_b_sd == pytest.approx(float(np.std([0.1, 0.9], ddof=1)), abs=1e-5)
    # Weighted SD should be the average of bucket A and B SDs (0.2828), NOT overall SD
    assert diag["weighted_within_bucket_sd"] != pytest.approx(overall_sd)
    assert diag["weighted_within_bucket_sd"] == pytest.approx(
        (bucket_a_sd + bucket_b_sd) / 2, abs=1e-5
    )


def test_lean_shares_between_zero_and_one():
    """Verify lean shares and followed lean shares are strictly bounded in [0, 1]."""
    df = pd.DataFrame(
        {
            "dd_bucket": ["1_short", "2_long", "3_medium"],
            "phat": [0.20, 0.80, 0.50],  # 2 lean plays (0.20, 0.80), 1 neutral
            "call_pass": [0, 1, 0],  # team followed lean on both lean plays
        }
    )

    diag, _ = compute_diagnostics_for_split(df, "test")

    assert 0.0 <= diag["lean_share"] <= 1.0
    assert 0.0 <= diag["followed_lean_share"] <= 1.0
    assert diag["lean_share"] == pytest.approx(2 / 3)
    assert diag["followed_lean_share"] == pytest.approx(1.0)


def test_diagnostics_file_contains_no_epa_derived_numbers():
    """Verify that results/model_diagnostics.json contains NO EPA or outcome figures."""
    assert config.MODEL_DIAGNOSTICS_PATH.exists(), (
        f"Missing {config.MODEL_DIAGNOSTICS_PATH}"
    )

    with open(config.MODEL_DIAGNOSTICS_PATH, "r", encoding="utf-8") as f:
        diag_content = json.load(f)

    # Inspect all keys and nested text for 'epa'
    raw_text = json.dumps(diag_content).lower()
    assert "epa" not in raw_text, (
        "EPA-derived figures or references found in model_diagnostics.json!"
    )
    assert "delta" not in raw_text

    # Verify structure
    assert "early" in diag_content
    assert "tune" in diag_content
    assert "summary" in diag_content

    for sp in ["early", "tune"]:
        sp_data = diag_content[sp]
        assert "weighted_within_bucket_sd" in sp_data
        assert "within_bucket_sds" in sp_data
        assert "distribution" in sp_data
        assert "lean_share" in sp_data
        assert "followed_lean_share" in sp_data
        assert len(sp_data["within_bucket_sds"]) == 9
        assert 0.0 <= sp_data["lean_share"] <= 1.0
        assert 0.0 <= sp_data["followed_lean_share"] <= 1.0


def test_phat_by_bucket_parquet_file():
    """Verify results/phat_by_bucket.parquet exists and has 18 rows (9 per split)."""
    assert config.PHAT_BY_BUCKET_PATH.exists(), (
        f"Missing {config.PHAT_BY_BUCKET_PATH}"
    )

    df_buckets = pd.read_parquet(config.PHAT_BY_BUCKET_PATH)
    assert len(df_buckets) == 18  # 9 early + 9 tune
    expected_cols = ["split", "dd_bucket", "n", "mean_phat", "sd_phat"]
    assert list(df_buckets.columns) == expected_cols
    assert set(df_buckets["split"].unique()) == {"early", "tune"}
    assert (df_buckets["n"] > 0).all()
    assert (df_buckets["sd_phat"] > 0).all()
