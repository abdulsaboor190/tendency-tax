"""Tests for robustness suite (Deliverable D5.2).

Ensures:
  1. Extra-regressor fit agrees with statsmodels to 1e-8.
  2. Each variant filters or substitutes correctly on synthetic data.
  3. Secondary stage refuses without env var and without primary.json.
  4. Robustness output dataframe validates against schema.
  5. Frozen model features and forbidden features are completely unchanged.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from scripts.run_all import run_secondary_pipeline
from src import config
from src.estimate import fit_arrays
from src.robustness import compute_robustness_suite, run_single_variant
from src.schema import validate_parquet_df


def test_extra_regressor_fit_agrees_with_statsmodels():
    """Verify fit_arrays with extra_regressors agrees with statsmodels OLS to 1e-8."""
    np.random.seed(42)
    n = 1000
    call = np.random.binomial(1, 0.55, size=n).astype(float)
    phat = np.random.uniform(0.1, 0.9, size=n)
    groups = np.random.choice(["1_short", "1_med", "2_long", "3_med"], size=n)
    shotgun = np.random.binomial(1, 0.65, size=n).astype(float)
    no_huddle = np.random.binomial(1, 0.15, size=n).astype(float)
    noise = np.random.normal(0, 0.4, size=n)

    epa = (
        0.20 * call
        - 0.10 * phat
        - 0.15 * (call * phat)
        + 0.05 * shotgun
        - 0.08 * no_huddle
        + noise
    )

    df = pd.DataFrame({
        "epa": epa,
        "call_pass": call,
        "phat": phat,
        "dd_bucket": groups,
        "shotgun": shotgun,
        "no_huddle": no_huddle,
    })

    # Statsmodels OLS
    sm_fit = smf.ols(
        "epa ~ call_pass + phat + call_pass:phat + shotgun + no_huddle + C(dd_bucket)",
        data=df,
    ).fit()

    # fit_arrays
    my_fit = fit_arrays(
        epa=epa,
        call=call,
        phat=phat,
        groups=groups,
        spec="B",
        extra_regressors=np.column_stack([shotgun, no_huddle]),
    )

    assert my_fit.b1 == pytest.approx(sm_fit.params["call_pass"], abs=1e-8)
    assert my_fit.b2 == pytest.approx(sm_fit.params["phat"], abs=1e-8)
    assert my_fit.b3 == pytest.approx(sm_fit.params["call_pass:phat"], abs=1e-8)


def test_robustness_variants_logic():
    """Verify each variant filters or substitutes as specified on hand-made data."""
    rng = np.random.default_rng(123)
    n = 200
    df = pd.DataFrame({
        "game_id": np.repeat([f"g{i}" for i in range(10)], 20),
        "season": np.repeat([2022, 2023], 100),
        "down": rng.choice([1, 2, 3], size=n),
        "dd_bucket": rng.choice(["1_short", "2_med", "3_long", "1_med"], size=n),
        "call_pass": rng.binomial(1, 0.5, size=n).astype(float),
        "phat": rng.uniform(0.2, 0.8, size=n),
        "xpass": rng.uniform(0.3, 0.7, size=n),
        "epa": rng.normal(0, 0.5, size=n),
        "success": rng.binomial(1, 0.45, size=n).astype(float),
        "shotgun": rng.binomial(1, 0.6, size=n).astype(float),
        "no_huddle": rng.binomial(1, 0.1, size=n).astype(float),
        "penalty": np.tile([0, 0, 0, 1], 50),  # 50 penalty plays
    })

    # Test penalty filter: 150 plays
    res_no_pen = run_single_variant(df, filter_mask=df["penalty"] == 0, n_boot=10)
    assert res_no_pen["n_plays"] == 150

    # Test down filter: down 1
    d1_count = int((df["down"] == 1).sum())
    res_down_1 = run_single_variant(df, filter_mask=df["down"] == 1, n_boot=10)
    assert res_down_1["n_plays"] == d1_count

    # Test outcome success
    res_succ = run_single_variant(df, outcome_col="success", n_boot=10)
    assert res_succ["n_plays"] == n
    assert not np.isnan(res_succ["delta"])

    # Test alternative expectation xpass
    res_xpass = run_single_variant(df, phat_col="xpass", n_boot=10)
    assert res_xpass["n_plays"] == n
    assert not np.isnan(res_xpass["delta"])


def test_secondary_stage_preconditions(monkeypatch):
    """Secondary stage refuses without env var or without primary.json."""
    # 1. Env var missing
    monkeypatch.delenv(config.HOLDOUT_ENV_VAR, raising=False)
    with pytest.raises(RuntimeError, match="Precondition failed"):
        run_secondary_pipeline()

    # 2. primary.json missing
    monkeypatch.setenv(config.HOLDOUT_ENV_VAR, "1")
    with patch("scripts.run_all.Path.exists", return_value=False):
        with pytest.raises(RuntimeError, match="primary.json.*missing"):
            run_secondary_pipeline()


def test_robustness_suite_schema_validation():
    """Verify compute_robustness_suite output validates against schema."""
    rng = np.random.default_rng(456)
    n = 300
    df = pd.DataFrame({
        "game_id": np.repeat([f"g{i}" for i in range(15)], 20),
        "season": np.repeat([2022, 2023], 150),
        "down": rng.choice([1, 2, 3], size=n),
        "dd_bucket": rng.choice(["1_short", "2_med", "3_long", "1_med"], size=n),
        "call_pass": rng.binomial(1, 0.5, size=n).astype(float),
        "phat": rng.uniform(0.2, 0.8, size=n),
        "xpass": rng.uniform(0.3, 0.7, size=n),
        "epa": rng.normal(0, 0.5, size=n),
        "success": rng.binomial(1, 0.45, size=n).astype(float),
        "shotgun": rng.binomial(1, 0.6, size=n).astype(float),
        "no_huddle": rng.binomial(1, 0.1, size=n).astype(float),
        "penalty": np.zeros(n, dtype=int),
    })

    suite_df = compute_robustness_suite({"tune": df}, n_boot=15, n_jobs=1)
    # Validate against schema
    validate_parquet_df(
        suite_df, schema_name="robustness", filename="robustness.parquet"
    )
    assert len(suite_df) > 0
    assert (suite_df["label"] == "secondary").all()


def test_phat_model_features_strictly_unchanged():
    """Ensure shotgun & no_huddle remain forbidden features and NOT model features."""
    assert "shotgun" in config.FORBIDDEN_FEATURES
    assert "no_huddle" in config.FORBIDDEN_FEATURES
    assert "shotgun" not in config.FEATURES
    assert "no_huddle" not in config.FEATURES
    assert len(config.FEATURES) == 10
