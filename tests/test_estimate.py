"""Tests for regression specifications A and B in src/estimate.py."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from src.estimate import SpecResult, fit_arrays, fit_spec


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
