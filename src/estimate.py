"""Regression specifications A and B for Tendency Tax estimation.

Implements fast OLS estimation on numpy arrays (fit_arrays),
DataFrame wrapper (fit_spec), game-clustered bootstrap (cluster_bootstrap),
and within-(season, dd_bucket) placebo permutation tests (placebo).

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

import joblib
import numpy as np
import pandas as pd

from src import config


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


class BootstrapResult(NamedTuple):
    spec: str
    n_boot: int
    delta: float
    delta_run: float
    delta_pass: float
    delta_boot: np.ndarray
    delta_run_boot: np.ndarray
    delta_pass_boot: np.ndarray
    ci_delta: tuple[float, float]
    ci_delta_run: tuple[float, float]
    ci_delta_pass: tuple[float, float]
    se_delta: float
    p_value_delta: float


class PlaceboResult(NamedTuple):
    spec: str
    n_perm: int
    delta_observed: float
    null_distribution: np.ndarray
    pct_2_5: float
    pct_97_5: float
    outside_range: bool
    p_value: float


def _demean_by_group(
    x: np.ndarray, group_codes: np.ndarray, n_groups: int
) -> np.ndarray:
    """Fast within-group demeaning using np.bincount."""
    counts = np.bincount(group_codes, minlength=n_groups)
    # Handle empty groups if any
    counts_safe = np.where(counts == 0, 1, counts)
    sums = np.bincount(group_codes, weights=x, minlength=n_groups)
    means = sums / counts_safe
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


def _bootstrap_worker(
    seed: np.random.SeedSequence,
    n_games: int,
    game_row_indices: tuple[np.ndarray, ...],
    epa: np.ndarray,
    call: np.ndarray,
    phat: np.ndarray,
    groups: np.ndarray | None,
    spec: str,
) -> tuple[float, float, float]:
    """Single bootstrap replicate worker function.

    Resamples game indices with replacement, stacks rows, and fits regression.
    """
    rng = np.random.default_rng(seed)
    sampled_game_idx = rng.integers(0, n_games, size=n_games)
    sampled_row_indices = np.concatenate(
        [game_row_indices[i] for i in sampled_game_idx]
    )

    b_epa = epa[sampled_row_indices]
    b_call = call[sampled_row_indices]
    b_phat = phat[sampled_row_indices]
    b_groups = groups[sampled_row_indices] if groups is not None else None

    res = fit_arrays(
        epa=b_epa,
        call=b_call,
        phat=b_phat,
        groups=b_groups,
        spec=spec,
    )
    return res.delta, res.delta_run, res.delta_pass


def cluster_bootstrap(
    df: pd.DataFrame,
    spec: str = "B",
    n_boot: int = config.BOOTSTRAP_REPS,
    n_jobs: int = -1,
    seed: int = config.RANDOM_SEED,
) -> BootstrapResult:
    """Game-clustered bootstrap for regression specifications A and B.

    Clustering on game_id. Resamples whole games with replacement,
    precomputing per-game row index arrays for speed.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing game_id, epa, call_pass, phat, and dd_bucket (if Spec B).
    spec : str
        "A" or "B"
    n_boot : int
        Number of bootstrap replications.
    n_jobs : int
        Number of parallel workers for joblib.
    seed : int
        Root seed for SeedSequence spawn.

    Returns
    -------
    BootstrapResult
    """
    for col in ["game_id", "epa", "call_pass", "phat"]:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' missing from DataFrame.")

    spec_clean = spec.strip().upper()
    if spec_clean == "B" and "dd_bucket" not in df.columns:
        raise KeyError("Required column 'dd_bucket' missing from DataFrame for Spec B.")

    # 1. Point estimate on full data
    point_res = fit_spec(df, spec=spec_clean)

    # 2. Extract numpy arrays
    epa = df["epa"].to_numpy(dtype=np.float64)
    call = df["call_pass"].to_numpy(dtype=np.float64)
    phat = df["phat"].to_numpy(dtype=np.float64)
    groups = df["dd_bucket"].to_numpy() if spec_clean == "B" else None

    # 3. Precompute per-game row indices
    game_ids = df["game_id"].to_numpy()
    unique_games, game_inv = np.unique(game_ids, return_inverse=True)
    n_games = len(unique_games)

    order = np.argsort(game_inv)
    sorted_inv = game_inv[order]
    splits = np.flatnonzero(np.diff(sorted_inv)) + 1
    grouped_row_indices = np.split(order, splits)
    game_row_indices: tuple[np.ndarray, ...] = tuple(grouped_row_indices)

    # 4. Deterministic RNG stream per replicate
    root_seq = np.random.SeedSequence(seed)
    child_seeds = root_seq.spawn(n_boot)

    # 5. Parallel execution with joblib
    replicates = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_bootstrap_worker)(
            child_seeds[b],
            n_games,
            game_row_indices,
            epa,
            call,
            phat,
            groups,
            spec_clean,
        )
        for b in range(n_boot)
    )

    deltas = np.array([r[0] for r in replicates], dtype=np.float64)
    delta_runs = np.array([r[1] for r in replicates], dtype=np.float64)
    delta_passes = np.array([r[2] for r in replicates], dtype=np.float64)

    # Percentile 95% CIs [2.5, 97.5]
    ci_delta = (
        float(np.percentile(deltas, 2.5)),
        float(np.percentile(deltas, 97.5)),
    )
    ci_delta_run = (
        float(np.percentile(delta_runs, 2.5)),
        float(np.percentile(delta_runs, 97.5)),
    )
    ci_delta_pass = (
        float(np.percentile(delta_passes, 2.5)),
        float(np.percentile(delta_passes, 97.5)),
    )

    # Bootstrap SE for delta
    se_delta = float(np.std(deltas, ddof=1))

    # Two-sided p-value for delta
    n_le_zero = np.sum(deltas <= 0)
    n_ge_zero = np.sum(deltas >= 0)
    p_val = float(
        min(
            1.0,
            2.0 * min((1 + n_le_zero) / (n_boot + 1), (1 + n_ge_zero) / (n_boot + 1)),
        )
    )

    return BootstrapResult(
        spec=spec_clean,
        n_boot=n_boot,
        delta=point_res.delta,
        delta_run=point_res.delta_run,
        delta_pass=point_res.delta_pass,
        delta_boot=deltas,
        delta_run_boot=delta_runs,
        delta_pass_boot=delta_passes,
        ci_delta=ci_delta,
        ci_delta_run=ci_delta_run,
        ci_delta_pass=ci_delta_pass,
        se_delta=se_delta,
        p_value_delta=p_val,
    )


def _placebo_worker(
    seed: np.random.SeedSequence,
    group_row_indices: tuple[np.ndarray, ...],
    epa: np.ndarray,
    call: np.ndarray,
    phat: np.ndarray,
    groups: np.ndarray | None,
    spec: str,
) -> float:
    """Single placebo replicate worker function.

    Permutes phat within each (season, dd_bucket) group, keeps call and epa fixed,
    and returns delta.
    """
    rng = np.random.default_rng(seed)
    phat_perm = phat.copy()
    for idx_array in group_row_indices:
        phat_perm[idx_array] = rng.permutation(phat[idx_array])

    res = fit_arrays(
        epa=epa,
        call=call,
        phat=phat_perm,
        groups=groups,
        spec=spec,
    )
    return res.delta


def placebo(
    df: pd.DataFrame,
    spec: str = "B",
    n_perm: int = config.PLACEBO_REPS,
    n_jobs: int = -1,
    seed: int = config.RANDOM_SEED,
) -> PlaceboResult:
    """Placebo test by permuting phat within each (season, dd_bucket) cell.

    EPA and call are kept fixed. Only phat is shuffled within each cell.
    Spec B fixed effects and call*phat interaction are recomputed under the shuffle.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing season, dd_bucket, epa, call_pass, and phat.
    spec : str
        "A" or "B" (defaults to "B").
    n_perm : int
        Number of permutations (default config.PLACEBO_REPS).
    n_jobs : int
        Number of parallel workers.
    seed : int
        Random seed for SeedSequence.

    Returns
    -------
    PlaceboResult
    """
    for col in ["season", "dd_bucket", "epa", "call_pass", "phat"]:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' missing from DataFrame.")

    spec_clean = spec.strip().upper()

    # 1. Observed point estimate
    obs_res = fit_spec(df, spec=spec_clean)
    delta_obs = obs_res.delta

    # 2. Extract numpy arrays
    epa = df["epa"].to_numpy(dtype=np.float64)
    call = df["call_pass"].to_numpy(dtype=np.float64)
    phat = df["phat"].to_numpy(dtype=np.float64)
    groups = df["dd_bucket"].to_numpy() if spec_clean == "B" else None

    # 3. Precompute row index arrays for each (season, dd_bucket) group
    seasons = df["season"].to_numpy()
    dd_buckets = df["dd_bucket"].to_numpy()
    # Combine season and dd_bucket as string keys
    combo_keys = [f"{s}_{b}" for s, b in zip(seasons, dd_buckets, strict=False)]
    _, group_inv = np.unique(combo_keys, return_inverse=True)

    order = np.argsort(group_inv)
    sorted_inv = group_inv[order]
    splits = np.flatnonzero(np.diff(sorted_inv)) + 1
    grouped_row_indices = np.split(order, splits)
    group_row_indices: tuple[np.ndarray, ...] = tuple(grouped_row_indices)

    # 4. Deterministic RNG stream per permutation
    root_seq = np.random.SeedSequence(seed)
    child_seeds = root_seq.spawn(n_perm)

    # 5. Parallel execution with joblib
    null_deltas = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_placebo_worker)(
            child_seeds[i],
            group_row_indices,
            epa,
            call,
            phat,
            groups,
            spec_clean,
        )
        for i in range(n_perm)
    )
    null_dist = np.array(null_deltas, dtype=np.float64)

    # 6. Null percentiles [2.5, 97.5]
    pct_2_5 = float(np.percentile(null_dist, 2.5))
    pct_97_5 = float(np.percentile(null_dist, 97.5))

    # 7. Outside range flag (strict inequalities)
    outside_range = bool(delta_obs < pct_2_5 or delta_obs > pct_97_5)

    # 8. Two-sided permutation p-value:
    # (1 + #(|null| >= |observed|)) / (n_perm + 1)
    abs_obs = abs(delta_obs)
    n_as_extreme = np.sum(np.abs(null_dist) >= abs_obs)
    p_val = float((1 + n_as_extreme) / (n_perm + 1))

    return PlaceboResult(
        spec=spec_clean,
        n_perm=n_perm,
        delta_observed=delta_obs,
        null_distribution=null_dist,
        pct_2_5=pct_2_5,
        pct_97_5=pct_97_5,
        outside_range=outside_range,
        p_value=p_val,
    )


def run_pipeline() -> dict[str, Any]:
    """Fit Specs A and B on Early and Tune splits and print summary.

    Does not touch holdout data.
    """
    import time

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

    print("\n" + "=" * 60)
    print("TIMING BENCHMARK: CLUSTERED BOOTSTRAP ON TUNE SPLIT")
    print("=" * 60)
    t0 = time.perf_counter()
    n_bench = 100
    boot_tune = cluster_bootstrap(tune_df, spec="B", n_boot=n_bench, n_jobs=-1)
    elapsed = time.perf_counter() - t0
    per_rep = elapsed / n_bench
    proj_tune_1000 = per_rep * 1000
    proj_early_1000 = proj_tune_1000 * (len(early_df) / len(tune_df))

    print(f"Tune bootstrap ({n_bench} reps, n_jobs=-1): {elapsed:.2f}s")
    print(f"Per-replicate time (tune): {per_rep * 1000:.2f} ms")
    print(
        f"Projected 1,000 reps on tune: {proj_tune_1000:.2f}s "
        f"({proj_tune_1000/60:.2f} min)"
    )
    print(
        f"Projected 1,000 reps on early: {proj_early_1000:.2f}s "
        f"({proj_early_1000/60:.2f} min)"
    )
    print(
        f"Tune Spec B Delta: {boot_tune.delta:+.4f}, "
        f"95% CI [{boot_tune.ci_delta[0]:+.4f}, {boot_tune.ci_delta[1]:+.4f}], "
        f"SE: {boot_tune.se_delta:.4f}, p: {boot_tune.p_value_delta:.4f}"
    )

    print("\n" + "=" * 60)
    print("PLACEBO TEST BENCHMARK ON TUNE SPLIT")
    print("=" * 60)
    t0_p = time.perf_counter()
    n_perm_bench = 100
    placebo_tune = placebo(tune_df, spec="B", n_perm=n_perm_bench, n_jobs=-1)
    elapsed_p = time.perf_counter() - t0_p
    print(
        f"Tune Placebo ({n_perm_bench} perms): {elapsed_p:.2f}s | "
        f"Observed delta: {placebo_tune.delta_observed:+.4f}, "
        f"Null 95% range: [{placebo_tune.pct_2_5:+.4f}, {placebo_tune.pct_97_5:+.4f}], "
        f"Outside range: {placebo_tune.outside_range}, "
        f"p-value: {placebo_tune.p_value:.4f}"
    )

    return results


if __name__ == "__main__":
    run_pipeline()
