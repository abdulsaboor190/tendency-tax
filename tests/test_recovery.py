"""Tests for simulation recovery and estimator validity (Deliverable D3.4)."""

import numpy as np
import pandas as pd

from src.simulate import (
    estimate_calibration_params,
    fit_spec_b_with_cr1_se,
    run_recovery_simulation,
)


def _make_synthetic_split(n_games=40, plays_per_game=25, seed=42):
    """Generate synthetic play dataset structured like tune split."""
    rng = np.random.default_rng(seed)
    n = n_games * plays_per_game
    game_ids = np.repeat([f"game_{i:03d}" for i in range(n_games)], plays_per_game)
    buckets = ["1_short", "1_med", "1_long", "2_med", "3_med", "3_long"]

    call = rng.binomial(1, 0.55, size=n).astype(float)
    phat = rng.uniform(0.15, 0.85, size=n)
    dd_bucket = rng.choice(buckets, size=n)

    # Base epa
    epa = 0.1 * call + rng.normal(0, 0.5, size=n)

    return pd.DataFrame({
        "game_id": game_ids,
        "season": 2022,
        "epa": epa,
        "call_pass": call,
        "phat": phat,
        "dd_bucket": dd_bucket,
    })


def test_cr1_se_calculation():
    """Verify fit_spec_b_with_cr1_se produces valid positive standard error."""
    df = _make_synthetic_split(n_games=20, plays_per_game=30, seed=123)
    game_ids = df["game_id"].to_numpy()
    _, game_inv = np.unique(game_ids, return_inverse=True)

    b1, b2, b3, se_delta = fit_spec_b_with_cr1_se(
        epa=df["epa"].to_numpy(),
        call=df["call_pass"].to_numpy(),
        phat=df["phat"].to_numpy(),
        groups=df["dd_bucket"].to_numpy(),
        game_inv=game_inv,
        n_games=len(np.unique(game_ids)),
    )

    assert isinstance(b1, float)
    assert isinstance(b2, float)
    assert isinstance(b3, float)
    assert isinstance(se_delta, float)
    assert se_delta > 0.0


def test_recovery_simulation_fast():
    """Fast recovery test in pytest (40 sims, small sample, bounds 0.85-1.0)."""
    df = _make_synthetic_split(n_games=30, plays_per_game=30, seed=777)
    calib = {
        "mu_dd": {b: 0.0 for b in df["dd_bucket"].unique()},
        "c": 0.08,
        "game_re_sd": 0.05,
        "noise_sd": 0.40,
    }

    # Run on 0.00 and 0.05 with 40 sims
    results = run_recovery_simulation(
        tune_df=df,
        calib=calib,
        true_deltas=[0.00, 0.05],
        n_sims=40,
        seed=12345,
        check_bootstrap=False,
    )

    # Verify delta = 0.00
    r0 = results["delta_0.00"]
    assert abs(r0["bias"]) < 0.02
    assert 0.85 <= r0["coverage_95"] <= 1.0

    # Verify delta = 0.05
    r5 = results["delta_0.05"]
    assert abs(r5["bias"]) < 0.02
    assert 0.85 <= r5["coverage_95"] <= 1.0


def test_calibration_param_estimation():
    """Verify calibration parameter estimation returns non-zero positive SDs."""
    df = _make_synthetic_split(n_games=20, plays_per_game=25, seed=99)
    calib = estimate_calibration_params(df)

    assert "mu_dd" in calib
    assert "c" in calib
    assert calib["game_re_sd"] > 0.0
    assert calib["noise_sd"] > 0.0
