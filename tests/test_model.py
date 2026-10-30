"""Tests for model training, hyperparameter search, and phat construction."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss, log_loss

from src import config
from src.load import get_split
from src.model import (
    add_phat,
    baseline_phat,
    fit_early_oof_models,
    fit_model,
    score,
)


@pytest.fixture(scope="module")
def early_data() -> pd.DataFrame:
    """Fixture providing early split data."""
    return get_split("early")


@pytest.fixture(scope="module")
def tune_data() -> pd.DataFrame:
    """Fixture providing tune split data."""
    return get_split("tune")


def test_forbidden_columns_are_not_features():
    """Assert none of the forbidden columns are in config.FEATURES.

    Strict rule: Never epa, xpass, success, wp, shotgun, or no_huddle.
    """
    for forbidden in config.FORBIDDEN_FEATURES:
        assert forbidden not in config.FEATURES, (
            f"Forbidden feature '{forbidden}' found in config.FEATURES!"
        )

    # Check model features strictly match config.FEATURES
    dummy_data = pd.DataFrame(
        {
            col: np.random.randn(20) if col != "down" else [1, 2] * 10
            for col in config.FEATURES
        }
    )
    dummy_data["call_pass"] = [0, 1] * 10
    clf = fit_model(dummy_data)
    assert list(clf.feature_names_in_) == config.FEATURES


def test_forbidden_columns_rejected_if_injected(monkeypatch: pytest.MonkeyPatch):
    """Verify fit_model raises ValueError if a forbidden feature is added."""
    monkeypatch.setattr(config, "FEATURES", list(config.FEATURES) + ["epa"])
    dummy_data = pd.DataFrame({col: np.zeros(10) for col in config.FEATURES})
    dummy_data["call_pass"] = 1
    with pytest.raises(ValueError) as exc:
        fit_model(dummy_data)
    assert "Forbidden feature 'epa' detected" in str(exc.value)


def test_phat_range_and_no_nulls_for_early_and_tune(
    early_data: pd.DataFrame,
    tune_data: pd.DataFrame,
):
    """Verify phat is in [0, 1] and has zero nulls for early and tune splits."""
    early_phat = add_phat(early_data)
    assert "phat" in early_phat.columns
    assert early_phat["phat"].isna().sum() == 0, "Nulls found in early phat!"
    assert (early_phat["phat"] >= 0.0).all()
    assert (early_phat["phat"] <= 1.0).all()

    tune_phat = add_phat(tune_data)
    assert "phat" in tune_phat.columns
    assert tune_phat["phat"].isna().sum() == 0, "Nulls found in tune phat!"
    assert (tune_phat["phat"] >= 0.0).all()
    assert (tune_phat["phat"] <= 1.0).all()


def test_running_add_phat_twice_gives_identical_values(early_data: pd.DataFrame):
    """Verify that add_phat is fully deterministic and produces identical values."""
    run1 = add_phat(early_data)
    run2 = add_phat(early_data)
    pd.testing.assert_series_equal(run1["phat"], run2["phat"])


def test_no_leakage_early_2018(early_data: pd.DataFrame):
    """Verify no leakage for early season 2018.

    Asserts on training indices: the model that produced 2018's phat
    was not trained on 2018 rows.
    """
    oof_models = fit_early_oof_models(early_data)

    assert 2018 in oof_models
    train_indices_2018 = set(oof_models[2018]["train_indices"])
    season_2018_indices = set(early_data.index[early_data["season"] == 2018])

    # Assert complete disjointness of training indices and 2018 indices
    assert train_indices_2018.isdisjoint(season_2018_indices), (
        "Data leakage: model for season 2018 was trained on season 2018 rows!"
    )

    # Check that training indices cover all other early seasons
    other_early_seasons = set(config.SPLITS["early"]) - {2018}
    expected_train_idx = set(
        early_data.index[early_data["season"].isin(other_early_seasons)]
    )
    assert train_indices_2018 == expected_train_idx

    # Check predictions match the model
    mask_2018 = early_data["season"] == 2018
    X_2018 = early_data.loc[mask_2018, config.FEATURES]
    direct_preds = oof_models[2018]["model"].predict_proba(X_2018)[:, 1]

    early_phat = add_phat(early_data)
    phat_2018 = early_phat.loc[mask_2018, "phat"].values
    np.testing.assert_allclose(direct_preds, phat_2018, rtol=1e-7, atol=1e-7)


def test_holdout_not_scored_in_add_phat():
    """Verify that holdout seasons are NOT scored by add_phat (remain NaN)."""
    # Create mock dataframe with holdout seasons
    mock_df = pd.DataFrame(
        {
            "season": [2024, 2024, 2025],
            "down": [1, 2, 3],
            "ydstogo": [10, 5, 2],
            "yardline_100": [50, 40, 20],
            "qtr": [1, 2, 3],
            "game_seconds_remaining": [3600, 1800, 600],
            "half_seconds_remaining": [1800, 900, 300],
            "score_differential": [0, 3, -7],
            "posteam_timeouts_remaining": [3, 2, 1],
            "defteam_timeouts_remaining": [3, 3, 2],
            "posteam_spread": [2.5, -1.0, 3.0],
            "call_pass": [1, 0, 1],
        }
    )
    scored = add_phat(mock_df)
    assert scored["phat"].isna().all(), "Holdout rows should not be scored!"


def test_frozen_final_model_and_metadata():
    """Verify frozen model and metadata files exist and have valid structure."""
    assert config.FROZEN_MODEL_PATH.exists(), (
        f"Missing frozen model file at {config.FROZEN_MODEL_PATH}"
    )
    assert config.FROZEN_MODEL_META_PATH.exists(), (
        f"Missing frozen metadata file at {config.FROZEN_MODEL_META_PATH}"
    )

    # Validate metadata content
    with open(config.FROZEN_MODEL_META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert "sklearn_version" in meta
    assert meta["features"] == config.FEATURES
    assert meta["params"] == config.MODEL_PARAMS
    expected_training_seasons = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
    assert meta["training_seasons"] == expected_training_seasons
    assert meta["n_plays"] == 193068
    assert meta["target"] == config.LABEL

    # Validate model load and inference capabilities
    model = joblib.load(config.FROZEN_MODEL_PATH)
    assert hasattr(model, "predict_proba")
    assert model.n_features_in_ == len(config.FEATURES)


def test_baseline_fallback_works_for_missing_cell():
    """Verify baseline_phat falls back to dd_bucket pass rate when a cell is missing."""
    # Training pool with 2 seasons so leave-one-season-out works
    train_pool = pd.DataFrame(
        {
            "season": [2016, 2016, 2017, 2017],
            "cell_id": [
                "1_short_red_zone",
                "1_short_midfield",
                "1_short_red_zone",
                "1_short_midfield",
            ],
            "dd_bucket": ["1_short", "1_short", "1_short", "1_short"],
            "call_pass": [1, 0, 1, 0],  # dd_bucket rate = 0.5
        }
    )

    # Test frame containing 1_short_backed_up (missing from training data!)
    test_df = pd.DataFrame(
        {
            "season": [2016, 2022],  # one early row, one tune row
            "cell_id": ["1_short_backed_up", "1_short_backed_up"],
            "dd_bucket": ["1_short", "1_short"],
            "call_pass": [1, 0],
        }
    )

    scored = baseline_phat(test_df, early_df=train_pool)
    assert "phat_baseline" in scored.columns
    assert "baseline_phat" in scored.columns

    # Both early and tune rows for the missing cell should fall back to 0.5
    assert scored.loc[0, "phat_baseline"] == pytest.approx(0.5)
    assert scored.loc[1, "phat_baseline"] == pytest.approx(0.5)


def test_score_matches_sklearn_on_tiny_example():
    """Verify score() matches sklearn log_loss and brier_score_loss."""
    df = pd.DataFrame(
        {
            "call_pass": [1, 0, 1, 1, 0, 0, 1, 0, 1, 0],
            "p": [0.9, 0.1, 0.8, 0.7, 0.3, 0.2, 0.65, 0.4, 0.85, 0.15],
        }
    )

    res = score(df, "p")

    # Direct sklearn calculation
    y_true = df["call_pass"].values
    p_clipped = np.clip(df["p"].values, 1e-6, 1.0 - 1e-6)
    expected_ll = float(log_loss(y_true, p_clipped))
    expected_bs = float(brier_score_loss(y_true, df["p"].values))

    assert res.log_loss == pytest.approx(expected_ll, abs=1e-7)
    assert res.brier_score == pytest.approx(expected_bs, abs=1e-7)
    # Check tuple unpacking compatibility
    ll_val, bs_val, calib_tbl = score(df, "p")
    assert ll_val == res.log_loss
    assert bs_val == res.brier_score


def test_calibration_bins_sum_to_full_row_count():
    """Verify calibration bins sum to the exact row count of the input dataframe."""
    np.random.seed(config.RANDOM_SEED)
    n_rows = 1234
    df = pd.DataFrame(
        {
            "call_pass": np.random.randint(0, 2, n_rows),
            "pred_prob": np.random.uniform(0.05, 0.95, n_rows),
        }
    )

    res = score(df, "pred_prob")
    calib = res.calibration

    assert len(calib) == 10, "Expected exactly 10 calibration bins!"
    assert calib["n"].sum() == n_rows, (
        f"Calibration bin counts sum ({calib['n'].sum()}) != total rows ({n_rows})"
    )
    assert (calib["mean_pred"] >= 0.0).all()
    assert (calib["mean_pred"] <= 1.0).all()


def test_model_scores_file_and_gate_g1():
    """Verify results/model_scores.json exists and confirms GATE G1 passes."""
    assert config.MODEL_SCORES_PATH.exists(), (
        f"Missing {config.MODEL_SCORES_PATH}"
    )

    with open(config.MODEL_SCORES_PATH, "r", encoding="utf-8") as f:
        scores = json.load(f)

    assert len(scores) == 6
    tune_scores = {s["predictor"]: s for s in scores if s["split"] == "tune"}

    # GATE G1 check: model log loss must beat baseline log loss on tune split
    model_tune_ll = tune_scores["model"]["log_loss"]
    base_tune_ll = tune_scores["baseline"]["log_loss"]

    assert model_tune_ll < base_tune_ll, (
        f"GATE G1 FAILED: Model log loss ({model_tune_ll}) is not lower than "
        f"baseline log loss ({base_tune_ll}) on TUNE split!"
    )


def test_calibration_parquet_files():
    """Verify calibration parquet files exist and have valid structure."""
    assert config.CALIBRATION_EARLY_PATH.exists()
    assert config.CALIBRATION_TUNE_PATH.exists()

    df_early = pd.read_parquet(config.CALIBRATION_EARLY_PATH)
    assert len(df_early) == 30  # 3 predictors x 10 bins
    expected_cols = [
        "split",
        "predictor",
        "bin",
        "mean_pred",
        "observed_pass_rate",
        "n",
    ]
    assert list(df_early.columns) == expected_cols
    assert set(df_early["predictor"].unique()) == {"model", "baseline", "xpass"}

    df_tune = pd.read_parquet(config.CALIBRATION_TUNE_PATH)
    assert len(df_tune) == 30
    assert list(df_tune.columns) == expected_cols
    assert set(df_tune["predictor"].unique()) == {"model", "baseline", "xpass"}
