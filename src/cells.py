"""27-cell grid analysis for Tendency Tax (Deliverable D5.3).

Cells are defined across 3 downs x 3 distance buckets x 3 field zones:
  - downs: 1, 2, 3
  - distance buckets: short (1-3), medium (4-7), long (8+)
  - zones: red_zone (<=20), midfield (21-79), backed_up (>=80)

Each cell is evaluated using Spec A:
  epa ~ call + phat + call*phat + constant (delta = -b3/2)
with game-clustered bootstrap inference and Benjamini-Hochberg FDR correction.

All rows are tagged with label: 'secondary'.
"""

import time
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats

from src import config
from src.estimate import cluster_bootstrap


def add_grid_cell_id(df: pd.DataFrame) -> pd.DataFrame:
    """Add (or overwrite) the 'cell_id' column for the 27-cell grid.

    The cell_id is formed as ``{down}_{dist_bucket}_{zone}`` where:

    - ``dist_bucket`` is derived from ``ydstogo``:
      short (<=3), medium (4-7), long (8+).
    - ``zone`` is derived from ``yardline_100`` using
      config.ZONE_CUT_POINTS: red_zone (<=z0), backed_up (>=z1), midfield
      otherwise.

    This column is independent of the model dd_bucket column and is only
    used for cell-level analysis.
    """
    if config.ZONE_CUT_POINTS is None:
        raise ValueError("config.ZONE_CUT_POINTS is None. Must be configured.")
    z0, z1 = config.ZONE_CUT_POINTS

    def _zone(yl100: float) -> str:
        if yl100 <= z0:
            return "red_zone"
        if yl100 >= z1:
            return "backed_up"
        return "midfield"

    def _dist(ydstogo: float) -> str:
        if ydstogo <= 3:
            return "short"
        if ydstogo <= 7:
            return "medium"
        return "long"

    df = df.copy()
    df["cell_id"] = (
        df["down"].astype(int).astype(str)
        + "_"
        + df["ydstogo"].map(_dist)
        + "_"
        + df["yardline_100"].map(_zone)
    )
    return df


def get_all_27_cell_defs() -> list[dict[str, Any]]:
    """Return the ordered list of all 27 cell definitions."""
    if config.ZONE_CUT_POINTS is None:
        raise ValueError("config.ZONE_CUT_POINTS is None. Must be configured.")

    cells = []
    for down in [1, 2, 3]:
        for dist in ["short", "medium", "long"]:
            for zone in config.ZONE_NAMES:
                cell_id = f"{down}_{dist}_{zone}"
                cells.append({
                    "cell_id": cell_id,
                    "down": int(down),
                    "dist_bucket": dist,
                    "zone": zone,
                })
    return cells


def analyze_cell(
    cell_df: pd.DataFrame,
    cell_def: dict[str, Any],
    split_name: str,
    n_boot: int = config.BOOTSTRAP_REPS,
    n_jobs: int = -1,
    seed: int = config.RANDOM_SEED,
) -> dict[str, Any]:
    """Compute statistics, quantiles, and Spec A bootstrap for a single cell.

    Degenerate cells are marked testable=False with NaN estimates and a plain reason.
    """
    n_plays = int(len(cell_df))
    cell_id = cell_def["cell_id"]
    down = cell_def["down"]
    dist_bucket = cell_def["dist_bucket"]
    zone = cell_def["zone"]

    # Base structure
    base_res = {
        "split": split_name,
        "cell_id": cell_id,
        "down": down,
        "dist_bucket": dist_bucket,
        "zone": zone,
        "n_plays": n_plays,
        "n_games": 0,
        "n_pass": 0,
        "n_run": 0,
        "phat_median": np.nan,
        "phat_p25": np.nan,
        "phat_p75": np.nan,
        "phat_sd": np.nan,
        "delta": np.nan,
        "ci_lo": np.nan,
        "ci_hi": np.nan,
        "se": np.nan,
        "p_boot": np.nan,
        "q_bh": np.nan,
        "testable": False,
        "below_min_plays": False,
        "verdict": "cant_tell",
        "reason": "",
        "label": "secondary",
    }

    # Minimum plays check
    if config.MIN_PLAYS_PER_CELL is not None:
        if n_plays < int(config.MIN_PLAYS_PER_CELL):
            base_res["below_min_plays"] = True
            base_res["reason"] = f"fewer than {config.MIN_PLAYS_PER_CELL} plays"
            return base_res

    if n_plays == 0:
        base_res["reason"] = "no plays in cell"
        return base_res

    n_games = int(cell_df["game_id"].nunique())
    n_pass = int((cell_df["call_pass"] == 1.0).sum())
    n_run = int((cell_df["call_pass"] == 0.0).sum())

    phat_vals = cell_df["phat"].to_numpy(dtype=np.float64)
    phat_sd = float(np.std(phat_vals, ddof=1)) if n_plays > 1 else 0.0

    base_res.update({
        "n_games": n_games,
        "n_pass": n_pass,
        "n_run": n_run,
        "phat_median": round(float(np.median(phat_vals)), 5),
        "phat_p25": round(float(np.percentile(phat_vals, 25)), 5),
        "phat_p75": round(float(np.percentile(phat_vals, 75)), 5),
        "phat_sd": round(phat_sd, 5),
    })

    # Degeneracy checks
    if n_pass == 0:
        base_res["reason"] = "no pass plays in cell"
        return base_res
    if n_run == 0:
        base_res["reason"] = "no run plays in cell"
        return base_res
    if n_games < 2:
        base_res["reason"] = "fewer than 2 games in cell"
        return base_res
    if phat_sd == 0.0:
        base_res["reason"] = "zero phat variance in cell"
        return base_res

    # Reset index before running bootstrap
    df_cell_clean = cell_df.reset_index(drop=True)

    try:
        boot_res = cluster_bootstrap(
            df_cell_clean,
            spec="A",
            n_boot=n_boot,
            n_jobs=n_jobs,
            seed=seed,
        )
        base_res.update({
            "delta": round(float(boot_res.delta), 5),
            "ci_lo": round(float(boot_res.ci_delta[0]), 5),
            "ci_hi": round(float(boot_res.ci_delta[1]), 5),
            "se": round(float(boot_res.se_delta), 5),
            "p_boot": round(float(boot_res.p_value_delta), 5),
            "testable": True,
            "reason": "",
        })
    except (ValueError, np.linalg.LinAlgError) as err:
        base_res["testable"] = False
        base_res["reason"] = f"regression rank-deficient or singular: {err}"

    return base_res


def apply_fdr_and_verdicts(cell_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply Benjamini-Hochberg FDR correction and assign verdicts within a split."""
    testable_indices = [
        i for i, r in enumerate(cell_records)
        if r["testable"] and not np.isnan(r["p_boot"])
    ]

    # Compute q_bh for testable cells
    if testable_indices:
        p_vals = np.array([cell_records[i]["p_boot"] for i in testable_indices])
        try:
            q_vals = scipy.stats.false_discovery_control(p_vals, method="bh")
        except AttributeError:
            from statsmodels.stats.multitest import multipletests
            _, q_vals, _, _ = multipletests(p_vals, method="fdr_bh")

        for idx, q_val in zip(testable_indices, q_vals, strict=True):
            cell_records[idx]["q_bh"] = round(float(q_val), 5)

    # Assign verdicts in pre-registered order
    for r in cell_records:
        if r["below_min_plays"]:
            r["verdict"] = "cant_tell"
            continue
        if not r["testable"]:
            r["verdict"] = "cant_tell"
            continue

        q = r["q_bh"]
        d = r["delta"]
        ci_lo = r["ci_lo"]
        ci_hi = r["ci_hi"]
        margin = config.CELL_EQUIV_MARGIN

        if q < 0.05 and d > 0:
            r["verdict"] = "premium"
            r["reason"] = "FDR q < 0.05 and positive delta"
        elif q < 0.05 and d < 0:
            r["verdict"] = "reverse"
            r["reason"] = "FDR q < 0.05 and negative delta"
        elif margin is not None and (-margin < ci_lo and ci_hi < margin):
            r["verdict"] = "none"
            r["reason"] = f"95% CI entirely within equivalence margin (+/-{margin})"
        else:
            r["verdict"] = "cant_tell"
            r["reason"] = (
                "interval includes zero and is too wide to rule out a meaningful effect"
            )

    return cell_records


def compute_cells_grid(
    dfs: dict[str, pd.DataFrame],
    n_boot: int = config.BOOTSTRAP_REPS,
    n_jobs: int = -1,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Compute the 27-cell grid analysis across splits (early, tune, holdout, all).

    Parameters
    ----------
    dfs : dict[str, pd.DataFrame]
        Dictionary with keys 'early', 'tune', 'holdout', and optionally 'all'.
        If 'all' is not present, it will be concatenated from the provided splits.
    n_boot : int
        Bootstrap replicates per cell.
    n_jobs : int
        Parallel worker count.
    seed : int
        RNG seed.

    Returns
    -------
    pd.DataFrame
        DataFrame matching the schema of results/cells.parquet.
    """
    if config.ZONE_CUT_POINTS is None:
        raise ValueError(
            "config.ZONE_CUT_POINTS is None. Must configure field zone cut points."
        )

    splits_data = dict(dfs)
    if "all" not in splits_data:
        splits_data["all"] = pd.concat(list(dfs.values()), ignore_index=True)

    cell_defs = get_all_27_cell_defs()
    all_split_records = []

    for split_name in ["early", "tune", "holdout", "all"]:
        if split_name not in splits_data:
            continue
        df_split = splits_data[split_name]
        print(
            f"\nComputing 27-cell grid for split '{split_name}'"
            f" (N={len(df_split):,})..."
        )
        t0 = time.perf_counter()

        split_cells = []
        for c_def in cell_defs:
            c_id = c_def["cell_id"]
            mask = df_split["cell_id"] == c_id
            cell_sub_df = df_split[mask]
            rec = analyze_cell(
                cell_sub_df,
                cell_def=c_def,
                split_name=split_name,
                n_boot=n_boot,
                n_jobs=n_jobs,
                seed=seed,
            )
            split_cells.append(rec)

        # FDR correction and verdicts for this split
        split_cells_with_verdicts = apply_fdr_and_verdicts(split_cells)
        n_testable = sum(1 for r in split_cells_with_verdicts if r["testable"])
        n_premium = sum(
            1 for r in split_cells_with_verdicts if r["verdict"] == "premium"
        )
        n_reverse = sum(
            1 for r in split_cells_with_verdicts if r["verdict"] == "reverse"
        )
        elapsed = time.perf_counter() - t0
        print(
            f"  Split '{split_name}' completed in {elapsed:.1f}s: "
            f"{n_testable}/27 testable cells, {n_premium} premium, {n_reverse} reverse."
        )

        all_split_records.extend(split_cells_with_verdicts)

    return pd.DataFrame(all_split_records)

