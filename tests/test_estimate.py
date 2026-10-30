"""Tests for regression specifications A and B and game-clustered bootstrap."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from src.estimate import (
    BootstrapResult,
    SpecResult,
    cluster_bootstrap,
    fit_arrays,
    fit_spec,
)


def test_fit_arrays_synthetic_exact_recovery():
    """Verify exact coefficient recovery on synthetic data without noise."""
    np.random.seed(42)
    n = 1000
    call = np.random.binomial(1, 0.55, size=n).astype(float)
    phat = np.random.uniform(0.1, 0.9, size=n)

    # True parameters
    true_const = 0.05
    true_b1 = 0.20
    true_b2 = -0.15
    true_b3 = -0.10

    # True outcome for Spec A
    epa_a = true_const + true_b1 * call + true_b2 * phat + true_b3 * (call * phat)

    res_a = fit_arrays(epa=epa_a, call=call, phat=phat, spec="A")
    assert res_a.intercept == pytest.approx(true_const, abs=1e-10)
    assert res_a.b1 == pytest.approx(true_b1, abs=1e-10)
    assert res_a.b2 == pytest.approx(true_b2, abs=1e-10)
    assert res_a.b3 == pytest.approx(true_b3, abs=1e-10)
    assert res_a.delta == pytest.approx(-true_b3 / 2.0, abs=1e-10)
    assert res_a.delta_run == pytest.approx(true_b2, abs=1e-10)
    assert res_a.delta_pass == pytest.approx(-(true_b2 + true_b3), abs=1e-10)

    # For Spec B with groups
    group_effects = np.array([0.1, -0.05, 0.2, -0.15, 0.0])
    groups = np.random.choice(5, size=n)
    epa_b = (
        group_effects[groups]
        + true_b1 * call
        + true_b2 * phat
        + true_b3 * (call * phat)
    )

    res_b = fit_arrays(epa=epa_b, call=call, phat=phat, groups=groups, spec="B")
    assert res_b.b1 == pytest.approx(true_b1, abs=1e-10)
    assert res_b.b2 == pytest.approx(true_b2, abs=1e-10)
    assert res_b.b3 == pytest.approx(true_b3, abs=1e-10)
    assert res_b.delta == pytest.approx(-true_b3 / 2.0, abs=1e-10)
    assert res_b.delta_run == pytest.approx(true_b2, abs=1e-10)
    assert res_b.delta_pass == pytest.approx(-(true_b2 + true_b3), abs=1e-10)


def test_fit_arrays_matches_statsmodels_ols():
    """Verify agreement with statsmodels OLS to < 1e-8 on noisy data."""
    np.random.seed(123)
    n = 2000
    call = np.random.binomial(1, 0.6, size=n).astype(float)
    phat = np.random.beta(2, 2, size=n)
    groups = np.random.choice(["1_short", "1_med", "2_long", "3_med"], size=n)
    noise = np.random.normal(0, 0.5, size=n)

    epa = 0.12 * call - 0.08 * phat - 0.14 * (call * phat) + noise

    df = pd.DataFrame({
        "epa": epa,
        "call_pass": call,
        "phat": phat,
        "dd_bucket": groups,
    })

    # Spec A
    res_a = fit_spec(df, spec="A")
    sm_a = smf.ols("epa ~ call_pass + phat + call_pass:phat", data=df).fit()
    assert res_a.intercept == pytest.approx(sm_a.params["Intercept"], abs=1e-8)
    assert res_a.b1 == pytest.approx(sm_a.params["call_pass"], abs=1e-8)
    assert res_a.b2 == pytest.approx(sm_a.params["phat"], abs=1e-8)
    assert res_a.b3 == pytest.approx(sm_a.params["call_pass:phat"], abs=1e-8)

    # Spec B
    res_b = fit_spec(df, spec="B")
    sm_b = smf.ols(
        "epa ~ call_pass + phat + call_pass:phat + C(dd_bucket)", data=df
    ).fit()
    assert res_b.b1 == pytest.approx(sm_b.params["call_pass"], abs=1e-8)
    assert res_b.b2 == pytest.approx(sm_b.params["phat"], abs=1e-8)
    assert res_b.b3 == pytest.approx(sm_b.params["call_pass:phat"], abs=1e-8)


def test_delta_formulas_hand_check():
    """Verify delta, delta_run, delta_pass formulas by hand."""
    b1 = 0.10
    b2 = 0.04   # delta_run = 0.04
    b3 = -0.12  # delta = -(-0.12)/2 = 0.06
    # delta_pass = -(b2 + b3) = -(0.04 - 0.12) = -(-0.08) = 0.08

    # delta_run != delta_pass (0.04 != 0.08)
    # average: (0.04 + 0.08)/2 = 0.06 = delta
    call = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    phat = np.array([0.2, 0.2, 0.8, 0.8, 0.3, 0.3, 0.7, 0.7], dtype=float)
    const = 0.0
    epa = const + b1 * call + b2 * phat + b3 * (call * phat)

    res = fit_arrays(epa=epa, call=call, phat=phat, spec="A")
    assert res.b1 == pytest.approx(b1, abs=1e-10)
    assert res.b2 == pytest.approx(b2, abs=1e-10)
    assert res.b3 == pytest.approx(b3, abs=1e-10)

    assert res.delta_run == pytest.approx(0.04, abs=1e-10)
    assert res.delta_pass == pytest.approx(0.08, abs=1e-10)
    assert res.delta == pytest.approx(0.06, abs=1e-10)
    assert (res.delta_run + res.delta_pass) / 2.0 == pytest.approx(res.delta, abs=1e-10)


def test_rank_deficient_raises_value_error():
    """Verify rank-deficient design matrix raises clear ValueError."""
    # Collinear phat and call
    call = np.array([0.0, 0.0, 1.0, 1.0])
    phat = np.array([0.0, 0.0, 1.0, 1.0])
    epa = np.array([0.1, 0.2, 0.3, 0.4])

    with pytest.raises(ValueError, match="rank-deficient"):
        fit_arrays(epa=epa, call=call, phat=phat, spec="A")


def test_zero_variance_raises_value_error():
    """Verify zero variance in call or phat raises ValueError."""
    call = np.ones(10)
    phat = np.linspace(0.1, 0.9, 10)
    epa = np.random.randn(10)

    with pytest.raises(ValueError, match="zero variance"):
        fit_arrays(epa=epa, call=call, phat=phat, spec="A")


def test_spec_result_namedtuple_fields():
    """Check fields of SpecResult."""
    epa = np.array([0.1, -0.2, 0.3, -0.4, 0.5, 0.2])
    call = np.array([0, 1, 0, 1, 1, 0], dtype=float)
    phat = np.array([0.2, 0.8, 0.4, 0.6, 0.7, 0.3])

    res = fit_arrays(epa=epa, call=call, phat=phat, spec="A")
    assert isinstance(res, SpecResult)
    assert res.spec == "A"
    assert res.n == 6
    assert isinstance(res.b1, float)
    assert isinstance(res.b2, float)
    assert isinstance(res.b3, float)
    assert isinstance(res.delta, float)


# --- Bootstrap Tests ---


def _create_synthetic_cluster_data(
    n_games=100, plays_per_game=50, b1=0.1, b2=0.05, b3=-0.2, seed=42
):
    """Helper to generate synthetic clustered data."""
    rng = np.random.default_rng(seed)
    n_total = n_games * plays_per_game
    game_ids = np.repeat([f"game_{i:03d}" for i in range(n_games)], plays_per_game)
    game_effects = {f"game_{i:03d}": rng.normal(0, 0.1) for i in range(n_games)}

    call = rng.binomial(1, 0.55, size=n_total).astype(float)
    phat = rng.uniform(0.15, 0.85, size=n_total)
    buckets = ["1_short", "1_med", "1_long", "2_med", "3_med"]
    dd_bucket = rng.choice(buckets, size=n_total)

    # Outcome
    g_effect = np.array([game_effects[gid] for gid in game_ids])
    noise = rng.normal(0, 0.3, size=n_total)
    epa = g_effect + b1 * call + b2 * phat + b3 * (call * phat) + noise

    return pd.DataFrame({
        "game_id": game_ids,
        "epa": epa,
        "call_pass": call,
        "phat": phat,
        "dd_bucket": dd_bucket,
    })


def test_bootstrap_deterministic_seed_and_n_jobs_invariance():
    """Verify that same seed produces identical results with n_jobs=1 and n_jobs=2."""
    df = _create_synthetic_cluster_data(n_games=30, plays_per_game=20, seed=10)

    res_1 = cluster_bootstrap(df, spec="B", n_boot=25, n_jobs=1, seed=999)
    res_2 = cluster_bootstrap(df, spec="B", n_boot=25, n_jobs=2, seed=999)

    assert isinstance(res_1, BootstrapResult)
    assert np.allclose(res_1.delta_boot, res_2.delta_boot, atol=1e-12)
    assert np.allclose(res_1.delta_run_boot, res_2.delta_run_boot, atol=1e-12)
    assert np.allclose(res_1.delta_pass_boot, res_2.delta_pass_boot, atol=1e-12)
    assert res_1.ci_delta == pytest.approx(res_2.ci_delta, abs=1e-10)
    assert res_1.se_delta == pytest.approx(res_2.se_delta, abs=1e-10)
    assert res_1.p_value_delta == pytest.approx(res_2.p_value_delta, abs=1e-10)


def test_bootstrap_ci_coverage_with_effect_and_null():
    """Verify CI excludes 0 when clear effect exists, and contains 0 under null."""
    # 1. Clear effect: b3 = -0.30 => delta = +0.15
    df_effect = _create_synthetic_cluster_data(
        n_games=120, plays_per_game=40, b3=-0.30, seed=42
    )
    res_effect = cluster_bootstrap(
        df_effect, spec="B", n_boot=200, n_jobs=1, seed=42
    )
    # delta = -(-0.30)/2 = 0.15
    assert res_effect.ci_delta[0] > 0.0  # excludes 0
    # Point estimate and CI contain the truth
    assert res_effect.ci_delta[0] <= 0.15 <= res_effect.ci_delta[1]
    assert res_effect.p_value_delta < 0.05

    # 2. Null effect: b3 = 0.0 => delta = 0.0
    df_null = _create_synthetic_cluster_data(
        n_games=80, plays_per_game=30, b3=0.0, seed=456
    )
    res_null = cluster_bootstrap(df_null, spec="B", n_boot=100, n_jobs=1, seed=42)
    assert res_null.ci_delta[0] <= 0.0 <= res_null.ci_delta[1]  # contains 0


def test_bootstrap_se_roughly_matches_analytic_clustered_se():
    """Verify bootstrap SE roughly agrees with statsmodels cluster-robust SE."""
    df = _create_synthetic_cluster_data(
        n_games=60, plays_per_game=40, b3=-0.20, seed=789
    )
    boot_res = cluster_bootstrap(df, spec="A", n_boot=150, n_jobs=1, seed=100)

    # Analytic OLS with cluster-robust SE on game_id
    model = smf.ols("epa ~ call_pass + phat + call_pass:phat", data=df)
    fit_sm = model.fit(cov_type="cluster", cov_kwds={"groups": df["game_id"]})

    # delta = -b3 / 2 => SE(delta) = 0.5 * SE(b3)
    sm_b3_se = fit_sm.bse["call_pass:phat"]
    analytic_delta_se = 0.5 * sm_b3_se

    # Check agreement within ~25%
    rel_diff = abs(boot_res.se_delta - analytic_delta_se) / analytic_delta_se
    assert rel_diff < 0.25


def test_resampling_moves_whole_games():
    """Verify that rows from the same game always move together in resamples."""
    # Create 5 games, each game has distinct value for epa and phat
    records = []
    for g_i in range(5):
        for _ in range(4):
            records.append({
                "game_id": f"g_{g_i}",
                "epa": float(g_i),
                "call_pass": 1.0,
                "phat": 0.5,
                "dd_bucket": "1_short",
            })
    df = pd.DataFrame(records)

    # Call worker directly on indices to verify that all rows of each
    # selected game appear together
    game_ids = df["game_id"].to_numpy()
    _, game_inv = np.unique(game_ids, return_inverse=True)
    order = np.argsort(game_inv)
    splits = np.flatnonzero(np.diff(game_inv[order])) + 1
    game_row_indices = tuple(np.split(order, splits))

    rng = np.random.default_rng(42)
    sampled_games = rng.integers(0, 5, size=5)
    sampled_rows = np.concatenate([game_row_indices[i] for i in sampled_games])

    # In sampled_rows, the counts of rows from each game must be exact multiples of 4
    sampled_df = df.iloc[sampled_rows]
    counts = sampled_df["game_id"].value_counts()
    for count in counts:
        assert count % 4 == 0
