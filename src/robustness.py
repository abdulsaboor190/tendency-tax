"""Robustness suite for Tendency Tax (Deliverable D5.2).

Computes variants of the primary Spec B specification:
  - controls: Spec B plus shotgun and no_huddle as additive controls
  - outcome: success instead of epa
  - by season: Spec B within each season separately
  - by down: Spec B within each down (1, 2, 3), keeping dd_bucket fixed effects
  - no penalties: drop plays with penalty == 1
  - alternative expectation: xpass used in place of phat
  - primary: reference Spec B

All variants are labeled 'secondary'.
"""

import time
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src import config
from src.estimate import fit_arrays


def _robustness_bootstrap_worker(
    seed: np.random.SeedSequence,
    n_games: int,
    game_row_indices: tuple[np.ndarray, ...],
    outcome: np.ndarray,
    call: np.ndarray,
    phat: np.ndarray,
    groups: np.ndarray,
    extra_regressors: np.ndarray | None,
) -> float | None:
    """Bootstrap replicate worker for robustness variants.

    Returns delta estimate or None if replicate is degenerate.
    """
    rng = np.random.default_rng(seed)
    sampled_game_idx = rng.integers(0, n_games, size=n_games)
    sampled_rows = np.concatenate([game_row_indices[i] for i in sampled_game_idx])

    b_y = outcome[sampled_rows]
    b_c = call[sampled_rows]
    b_p = phat[sampled_rows]
    b_g = groups[sampled_rows]
    b_w = extra_regressors[sampled_rows] if extra_regressors is not None else None

    try:
        res = fit_arrays(
            epa=b_y,
            call=b_c,
            phat=b_p,
            groups=b_g,
            spec="B",
            extra_regressors=b_w,
        )
        return res.delta
    except (ValueError, np.linalg.LinAlgError):
        return None


def run_single_variant(
    df: pd.DataFrame,
    outcome_col: str = "epa",
    phat_col: str = "phat",
    extra_cols: list[str] | None = None,
    filter_mask: np.ndarray | pd.Series | None = None,
    n_boot: int = config.ROBUSTNESS_REPS,
    n_jobs: int = -1,
    seed: int = config.RANDOM_SEED,
) -> dict[str, Any]:
    """Fit a single robustness variant and compute bootstrap statistics."""
    sub_df = df if filter_mask is None else df[filter_mask]
    n_plays = len(sub_df)
    n_games = sub_df["game_id"].nunique()

    outcome = sub_df[outcome_col].to_numpy(dtype=np.float64)
    call = sub_df["call_pass"].to_numpy(dtype=np.float64)
    phat = sub_df[phat_col].to_numpy(dtype=np.float64)
    groups = sub_df["dd_bucket"].to_numpy()
    extra_w = (
        sub_df[extra_cols].to_numpy(dtype=np.float64) if extra_cols else None
    )

    # Point fit on full sample
    point_res = fit_arrays(
        epa=outcome,
        call=call,
        phat=phat,
        groups=groups,
        spec="B",
        extra_regressors=extra_w,
    )
    point_delta = point_res.delta

    # Group row indices by game
    game_ids = sub_df["game_id"].to_numpy()
    unique_games, game_inv = np.unique(game_ids, return_inverse=True)
    num_games = len(unique_games)
    order = np.argsort(game_inv)
    splits = np.flatnonzero(np.diff(game_inv[order])) + 1
    game_row_indices = tuple(np.split(order, splits))

    # Bootstrap
    root_seq = np.random.SeedSequence(seed)
    child_seeds = root_seq.spawn(n_boot)

    raw_boot = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_robustness_bootstrap_worker)(
            child_seeds[b],
            num_games,
            game_row_indices,
            outcome,
            call,
            phat,
            groups,
            extra_w,
        )
        for b in range(n_boot)
    )

    valid_deltas = np.array([d for d in raw_boot if d is not None], dtype=np.float64)
    n_valid = len(valid_deltas)

    if n_valid < 10:
        ci_lo, ci_hi, se, p_val = np.nan, np.nan, np.nan, np.nan
    else:
        ci_lo = float(np.percentile(valid_deltas, 2.5))
        ci_hi = float(np.percentile(valid_deltas, 97.5))
        se = float(np.std(valid_deltas, ddof=1))
        n_le_zero = np.sum(valid_deltas <= 0)
        n_ge_zero = np.sum(valid_deltas >= 0)
        p_val = float(
            min(
                1.0,
                2.0 * min(
                    (1 + n_le_zero) / (n_valid + 1),
                    (1 + n_ge_zero) / (n_valid + 1),
                ),
            )
        )

    return {
        "delta": round(point_delta, 5),
        "ci_lo": round(ci_lo, 5) if not np.isnan(ci_lo) else np.nan,
        "ci_hi": round(ci_hi, 5) if not np.isnan(ci_hi) else np.nan,
        "se": round(se, 5) if not np.isnan(se) else np.nan,
        "p_boot": round(p_val, 5) if not np.isnan(p_val) else np.nan,
        "n_plays": int(n_plays),
        "n_games": int(n_games),
        "n_reps_valid": int(n_valid),
        "label": "secondary",
    }


def compute_robustness_suite(
    dfs: dict[str, pd.DataFrame],
    n_boot: int = config.ROBUSTNESS_REPS,
    n_jobs: int = -1,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Compute all robustness variants across splits.

    Parameters
    ----------
    dfs : dict[str, pd.DataFrame]
        Mapping from split name to prepared DataFrame with scored phat
        and required columns (shotgun, no_huddle, success, xpass, penalty).
    n_boot : int
        Bootstrap replicates per variant.
    n_jobs : int
        Parallel worker count.
    seed : int
        RNG seed.

    Returns
    -------
    pd.DataFrame
        DataFrame matching the schema of results/robustness.parquet.
    """
    records = []

    for split_name, df in dfs.items():
        print(f"\nRobustness suite for split '{split_name}' (N={len(df):,})...")

        # 1. Primary Spec B reference
        t0 = time.perf_counter()
        res_prim = run_single_variant(df, n_boot=n_boot, n_jobs=n_jobs, seed=seed)
        t_prim = time.perf_counter() - t0
        print(f"  [primary] delta={res_prim['delta']:+.4f} ({t_prim:.1f}s)")
        records.append({
            "variant_id": "primary",
            "variant_label": "Primary Spec B (Reference)",
            "split": split_name,
            "subset": "all",
            "outcome": "epa",
            "phat_source": "phat",
            "controls": "dd_bucket",
            **res_prim,
        })

        # 2. Controls: shotgun and no_huddle
        t0 = time.perf_counter()
        res_ctrl = run_single_variant(
            df,
            extra_cols=["shotgun", "no_huddle"],
            n_boot=n_boot,
            n_jobs=n_jobs,
            seed=seed,
        )
        t_ctrl = time.perf_counter() - t0
        print(f"  [controls] delta={res_ctrl['delta']:+.4f} ({t_ctrl:.1f}s)")
        records.append({
            "variant_id": "controls_shotgun_no_huddle",
            "variant_label": "Add shotgun and no_huddle controls",
            "split": split_name,
            "subset": "all",
            "outcome": "epa",
            "phat_source": "phat",
            "controls": "dd_bucket + shotgun + no_huddle",
            **res_ctrl,
        })

        # 3. Outcome: success
        t0 = time.perf_counter()
        res_succ = run_single_variant(
            df,
            outcome_col="success",
            n_boot=n_boot,
            n_jobs=n_jobs,
            seed=seed,
        )
        t_succ = time.perf_counter() - t0
        print(f"  [outcome_success] delta={res_succ['delta']:+.4f} ({t_succ:.1f}s)")
        records.append({
            "variant_id": "outcome_success",
            "variant_label": "Outcome: Success Rate",
            "split": split_name,
            "subset": "all",
            "outcome": "success",
            "phat_source": "phat",
            "controls": "dd_bucket",
            **res_succ,
        })

        # 4. No penalties
        t0 = time.perf_counter()
        mask_no_pen = df["penalty"] == 0
        res_pen = run_single_variant(
            df,
            filter_mask=mask_no_pen,
            n_boot=n_boot,
            n_jobs=n_jobs,
            seed=seed,
        )
        t_pen = time.perf_counter() - t0
        print(f"  [no_penalties] delta={res_pen['delta']:+.4f} ({t_pen:.1f}s)")
        records.append({
            "variant_id": "no_penalties",
            "variant_label": "Exclude Penalty Plays",
            "split": split_name,
            "subset": "penalty == 0",
            "outcome": "epa",
            "phat_source": "phat",
            "controls": "dd_bucket",
            **res_pen,
        })

        # 5. Alternative expectation: xpass
        t0 = time.perf_counter()
        res_xpass = run_single_variant(
            df,
            phat_col="xpass",
            n_boot=n_boot,
            n_jobs=n_jobs,
            seed=seed,
        )
        t_xp = time.perf_counter() - t0
        print(f"  [alt_xpass] delta={res_xpass['delta']:+.4f} ({t_xp:.1f}s)")
        records.append({
            "variant_id": "alt_expectation_xpass",
            "variant_label": "Alternative Expectation: nflfastR xpass",
            "split": split_name,
            "subset": "all",
            "outcome": "epa",
            "phat_source": "xpass",
            "controls": "dd_bucket",
            **res_xpass,
        })

        # 6. By down (1, 2, 3)
        for d in [1, 2, 3]:
            t0 = time.perf_counter()
            mask_d = df["down"] == d
            res_down = run_single_variant(
                df,
                filter_mask=mask_d,
                n_boot=n_boot,
                n_jobs=n_jobs,
                seed=seed,
            )
            t_down = time.perf_counter() - t0
            print(f"  [down_{d}] delta={res_down['delta']:+.4f} ({t_down:.1f}s)")
            records.append({
                "variant_id": f"down_{d}",
                "variant_label": f"Subsample: Down {d}",
                "split": split_name,
                "subset": f"down == {d}",
                "outcome": "epa",
                "phat_source": "phat",
                "controls": "dd_bucket",
                **res_down,
            })

        # 7. By season
        seasons = sorted(df["season"].unique())
        for yr in seasons:
            t0 = time.perf_counter()
            mask_yr = df["season"] == yr
            res_yr = run_single_variant(
                df,
                filter_mask=mask_yr,
                n_boot=n_boot,
                n_jobs=n_jobs,
                seed=seed,
            )
            t_yr = time.perf_counter() - t0
            print(f"  [season_{yr}] delta={res_yr['delta']:+.4f} ({t_yr:.1f}s)")
            records.append({
                "variant_id": f"season_{yr}",
                "variant_label": f"Season {yr} Only",
                "split": split_name,
                "subset": f"season == {yr}",
                "outcome": "epa",
                "phat_source": "phat",
                "controls": "dd_bucket",
                **res_yr,
            })

    out_df = pd.DataFrame(records)
    return out_df
