"""Regression specifications A and B for Tendency Tax estimation.

Implements fast OLS estimation on numpy arrays (fit_arrays) and
DataFrame wrapper (fit_spec).
Outcome: epa.
Call: call_pass (1 = pass call).
Phat: pre-snap pass probability.

Spec B: epa ~ b1*call + b2*phat + b3*(call*phat) + fixed effects
for dd_bucket (9 buckets).
Spec A: epa ~ const + b1*call + b2*phat + b3*(call*phat) (single constant).

Estimands:
  delta = -b3 / 2
  delta_run = b2
  delta_pass = -(b2 + b3)
"""

from typing import Any, NamedTuple

import numpy as np
import pandas as pd


class SpecResult(NamedTuple):
    spec: str
    b1: float
    b2: float
    b3: float
    delta: float
    delta_run: float
    delta_pass: float
    n: int
    intercept: float | None = None
    r2: float | None = None


def _demean_by_group(
    x: np.ndarray, group_codes: np.ndarray, n_groups: int
) -> np.ndarray:
    """Fast within-group demeaning using np.bincount."""
    counts = np.bincount(group_codes, minlength=n_groups)
    if np.any(counts == 0):
        # Handle empty groups if any
        counts = np.maximum(counts, 1)
    sums = np.bincount(group_codes, weights=x, minlength=n_groups)
    means = sums / counts
    return x - means[group_codes]


def fit_arrays(
    epa: np.ndarray,
    call: np.ndarray,
    phat: np.ndarray,
    groups: np.ndarray | None = None,
    spec: str = "B",
) -> SpecResult:
    """Fit regression Spec A or B on numpy arrays.

    Parameters
    ----------
    epa : np.ndarray
        Outcome vector (1D float).
    call : np.ndarray
        Call binary indicator (1D float/int, 1 = pass).
    phat : np.ndarray
        Pre-snap pass probability (1D float).
    groups : np.ndarray | None
        Group labels for fixed effects (required for Spec B, ignored for Spec A).
    spec : str
        "A" (single constant) or "B" (group fixed effects).

    Returns
    -------
    SpecResult
        NamedTuple with regression coefficients and estimands.
    """
    spec_clean = spec.strip().upper()
    if spec_clean not in {"A", "B"}:
        raise ValueError(f"Unknown spec '{spec}'. Must be 'A' or 'B'.")

    # Ensure 1D numpy arrays
    y = np.asarray(epa, dtype=np.float64).ravel()
    c = np.asarray(call, dtype=np.float64).ravel()
    p = np.asarray(phat, dtype=np.float64).ravel()

    n = len(y)
    if len(c) != n or len(p) != n:
        raise ValueError(
            f"Array length mismatch: epa={n}, call={len(c)}, phat={len(p)}"
        )
    if n < 4:
        raise ValueError(f"Insufficient observations (n={n}) to fit regression.")

    interaction = c * p

    # Check for zero variance
    if np.var(c) == 0:
        raise ValueError("Column 'call' has zero variance.")
    if np.var(p) == 0:
        raise ValueError("Column 'phat' has zero variance.")

    intercept_val = None

    if spec_clean == "B":
        if groups is None:
            raise ValueError("groups array must be provided for Spec B fixed effects.")
        g = np.asarray(groups).ravel()
        if len(g) != n:
            raise ValueError(
                f"groups length ({len(g)}) does not match epa length ({n})"
            )

        # Encode groups as integers 0..K-1
        _, group_codes = np.unique(g, return_inverse=True)
        n_groups = len(np.unique(group_codes))

        # Within-group demeaning
        y_tilde = _demean_by_group(y, group_codes, n_groups)
        c_tilde = _demean_by_group(c, group_codes, n_groups)
        p_tilde = _demean_by_group(p, group_codes, n_groups)
        cp_tilde = _demean_by_group(interaction, group_codes, n_groups)

        # Check within-group variances
        if np.var(c_tilde) == 0:
            raise ValueError("Demeaned 'call' has zero variance.")
        if np.var(p_tilde) == 0:
            raise ValueError("Demeaned 'phat' has zero variance.")

        # X design matrix (no constant after demeaning)
        X = np.column_stack([c_tilde, p_tilde, cp_tilde])
        rank = np.linalg.matrix_rank(X)
        if rank < 3:
            raise ValueError(
                f"Design matrix is rank-deficient: rank {rank} < {X.shape[1]}"
            )

        # OLS without constant
        # beta: [b1, b2, b3]
        beta, _, _, _ = np.linalg.lstsq(X, y_tilde, rcond=None)
        b1, b2, b3 = float(beta[0]), float(beta[1]), float(beta[2])

    else:  # Spec A
        ones = np.ones(n, dtype=np.float64)
        X = np.column_stack([ones, c, p, interaction])
        rank = np.linalg.matrix_rank(X)
        if rank < 4:
            raise ValueError(
                f"Design matrix is rank-deficient: rank {rank} < {X.shape[1]}"
            )

        # beta: [const, b1, b2, b3]
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        intercept_val = float(beta[0])
        b1, b2, b3 = float(beta[1]), float(beta[2]), float(beta[3])

    delta = -b3 / 2.0
    delta_run = b2
    delta_pass = -(b2 + b3)

    return SpecResult(
        spec=spec_clean,
        b1=b1,
        b2=b2,
        b3=b3,
        delta=delta,
        delta_run=delta_run,
        delta_pass=delta_pass,
        n=n,
        intercept=intercept_val,
    )


def fit_spec(df: pd.DataFrame, spec: str = "B") -> SpecResult:
    """Thin wrapper taking a DataFrame with epa, call_pass, phat, and dd_bucket.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns:
          - epa: outcome
          - call_pass: binary indicator (1 = pass call)
          - phat: pre-snap pass probability
          - dd_bucket: down x distance bucket (required for Spec B)
    spec : str
        "A" or "B"

    Returns
    -------
    SpecResult
    """
    for col in ["epa", "call_pass", "phat"]:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' missing from DataFrame.")

    spec_clean = spec.strip().upper()
    if spec_clean == "B" and "dd_bucket" not in df.columns:
        raise KeyError("Required column 'dd_bucket' missing from DataFrame for Spec B.")

    epa = df["epa"].to_numpy()
    call = df["call_pass"].to_numpy()
    phat = df["phat"].to_numpy()
    groups = df["dd_bucket"].to_numpy() if "dd_bucket" in df.columns else None

    return fit_arrays(epa=epa, call=call, phat=phat, groups=groups, spec=spec_clean)


def run_pipeline() -> dict[str, Any]:
    """Fit Specs A and B on Early and Tune splits and print summary.

    Does not touch holdout data.
    """
    from src.load import get_split
    from src.model import add_phat

    print("Loading early and tune splits...")
    early_df = add_phat(get_split("early"))
    tune_df = add_phat(get_split("tune"))

    results = {}
    print("\n" + "=" * 60)
    print("REGRESSION ESTIMATION RESULTS (EARLY & TUNE)")
    print("=" * 60)

    for split_name, df in [("early", early_df), ("tune", tune_df)]:
        res_a = fit_spec(df, spec="A")
        res_b = fit_spec(df, spec="B")
        results[f"{split_name}_spec_a"] = res_a._asdict()
        results[f"{split_name}_spec_b"] = res_b._asdict()

        print(f"\n--- Split: {split_name.upper()} (N = {len(df):,}) ---")
        for res in [res_a, res_b]:
            print(
                f"Spec {res.spec}: "
                f"b1(call)={res.b1:+.4f}, "
                f"b2(phat)={res.b2:+.4f}, "
                f"b3(call*phat)={res.b3:+.4f} | "
                f"delta={res.delta:+.4f}, "
                f"delta_run={res.delta_run:+.4f}, "
                f"delta_pass={res.delta_pass:+.4f}"
            )

    return results


if __name__ == "__main__":
    run_pipeline()
