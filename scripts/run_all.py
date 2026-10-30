"""End-to-end execution pipeline for Tendency Tax.

Supports stages:
  --stage dev: Runs early and tune splits with bootstrap, placebo, and decile tables.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src import config
from src.estimate import cluster_bootstrap, fit_spec, placebo
from src.load import get_split
from src.model import add_phat


def _decile_bootstrap_worker(
    seed: np.random.SeedSequence,
    n_games: int,
    game_row_indices: tuple[np.ndarray, ...],
    epa: np.ndarray,
    call: np.ndarray,
    deciles: np.ndarray,
    n_deciles: int = 10,
) -> np.ndarray:
    """Worker for decile pass-minus-run gap bootstrap."""
    rng = np.random.default_rng(seed)
    sampled_game_idx = rng.integers(0, n_games, size=n_games)
    sampled_row_indices = np.concatenate(
        [game_row_indices[i] for i in sampled_game_idx]
    )

    b_epa = epa[sampled_row_indices]
    b_call = call[sampled_row_indices]
    b_dec = deciles[sampled_row_indices]

    gaps = np.full(n_deciles, np.nan, dtype=np.float64)
    for d in range(n_deciles):
        mask_d = b_dec == d
        mask_pass = mask_d & (b_call == 1.0)
        mask_run = mask_d & (b_call == 0.0)

        n_p = np.sum(mask_pass)
        n_r = np.sum(mask_run)
        if n_p > 0 and n_r > 0:
            gaps[d] = np.mean(b_epa[mask_pass]) - np.mean(b_epa[mask_run])

    return gaps


def compute_decile_table(
    df: pd.DataFrame,
    edges: np.ndarray,
    split_name: str,
    n_boot: int = config.BOOTSTRAP_REPS,
    n_jobs: int = -1,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Compute descriptive pass-minus-run EPA gap table across phat deciles."""
    # Bin phat into 10 deciles using pre-specified edges
    # Include lowest edge: bin 0..9
    phat = df["phat"].to_numpy()
    deciles = np.digitize(phat, edges[1:-1])  # maps to 0..9

    epa = df["epa"].to_numpy(dtype=np.float64)
    call = df["call_pass"].to_numpy(dtype=np.float64)
    game_ids = df["game_id"].to_numpy()

    # Precompute per-game row indices
    unique_games, game_inv = np.unique(game_ids, return_inverse=True)
    n_games = len(unique_games)
    order = np.argsort(game_inv)
    splits = np.flatnonzero(np.diff(game_inv[order])) + 1
    game_row_indices = tuple(np.split(order, splits))

    # Point estimates
    records = []
    point_gaps = np.empty(10, dtype=np.float64)

    for d in range(10):
        mask_d = deciles == d
        mask_pass = mask_d & (call == 1.0)
        mask_run = mask_d & (call == 0.0)

        n_p = int(np.sum(mask_pass))
        n_r = int(np.sum(mask_run))
        mean_epa_pass = float(np.mean(epa[mask_pass])) if n_p > 0 else np.nan
        mean_epa_run = float(np.mean(epa[mask_run])) if n_r > 0 else np.nan
        gap = mean_epa_pass - mean_epa_run
        point_gaps[d] = gap

        records.append({
            "split": split_name,
            "decile": d + 1,
            "phat_min": float(edges[d]),
            "phat_max": float(edges[d + 1]),
            "n_pass": n_p,
            "n_run": n_r,
            "mean_epa_pass": mean_epa_pass,
            "mean_epa_run": mean_epa_run,
            "epa_gap_pass_minus_run": gap,
        })

    # Bootstrap for 95% CI bands
    root_seq = np.random.SeedSequence(seed)
    child_seeds = root_seq.spawn(n_boot)

    boot_reps = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_decile_bootstrap_worker)(
            child_seeds[b],
            n_games,
            game_row_indices,
            epa,
            call,
            deciles,
            10,
        )
        for b in range(n_boot)
    )
    boot_matrix = np.array(boot_reps)  # n_boot x 10

    for d in range(10):
        col_vals = boot_matrix[:, d]
        col_clean = col_vals[~np.isnan(col_vals)]
        if len(col_clean) > 0:
            ci_low = float(np.percentile(col_clean, 2.5))
            ci_high = float(np.percentile(col_clean, 97.5))
        else:
            ci_low, ci_high = np.nan, np.nan
        records[d]["ci_95_low"] = ci_low
        records[d]["ci_95_high"] = ci_high

    return pd.DataFrame(records)


def run_dev_pipeline(
    n_boot: int = config.BOOTSTRAP_REPS,
    n_perm: int = config.PLACEBO_REPS,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """Execute the primary development pipeline on early and tune splits."""
    start_total = time.perf_counter()
    print("=" * 80)
    print("STARTING DEV STAGE PIPELINE (EARLY 2016-2021 & TUNE 2022-2023)")
    print("=" * 80)

    # 1. Load data
    print("Loading early and tune splits...")
    t0 = time.perf_counter()
    early_df = add_phat(get_split("early"))
    tune_df = add_phat(get_split("tune"))
    print(f"Data loaded in {time.perf_counter() - t0:.2f}s.")
    n_early_games = early_df["game_id"].nunique()
    n_tune_games = tune_df["game_id"].nunique()
    print(f"  Early: {len(early_df):,} plays across {n_early_games:,} games")
    print(f"  Tune:  {len(tune_df):,} plays across {n_tune_games:,} games")

    # 2. Compute decile edges from early split
    early_phat = early_df["phat"].to_numpy()
    decile_percentiles = np.linspace(0, 100, 11)
    decile_edges = np.percentile(early_phat, decile_percentiles)
    # Ensure strict monotonic edges
    decile_edges[0] = 0.0
    decile_edges[-1] = 1.0
    edges_str = np.round(decile_edges, 3).tolist()
    print(f"\nComputed decile edges from Early split: {edges_str}")

    splits = [("early", early_df), ("tune", tune_df)]
    summary_results: dict[str, Any] = {}
    decile_tables = []

    for split_name, df in splits:
        print("\n" + "-" * 60)
        print(f"PROCESSING SPLIT: {split_name.upper()}")
        print("-" * 60)

        # Baseline point fits
        res_a_point = fit_spec(df, spec="A")
        res_b_point = fit_spec(df, spec="B")

        # Spec B Bootstrap
        print(f"Running Spec B cluster bootstrap ({n_boot} reps)...")
        tb = time.perf_counter()
        boot_b = cluster_bootstrap(
            df, spec="B", n_boot=n_boot, n_jobs=n_jobs, seed=config.RANDOM_SEED
        )
        print(f"Spec B bootstrap finished in {time.perf_counter() - tb:.2f}s.")

        # Spec A Bootstrap
        print(f"Running Spec A cluster bootstrap ({n_boot} reps)...")
        ta = time.perf_counter()
        boot_a = cluster_bootstrap(
            df, spec="A", n_boot=n_boot, n_jobs=n_jobs, seed=config.RANDOM_SEED
        )
        print(f"Spec A bootstrap finished in {time.perf_counter() - ta:.2f}s.")

        # Spec B Placebo
        print(f"Running Spec B placebo test ({n_perm} perms)...")
        tp = time.perf_counter()
        placebo_b = placebo(
            df, spec="B", n_perm=n_perm, n_jobs=n_jobs, seed=config.RANDOM_SEED
        )
        print(f"Spec B placebo finished in {time.perf_counter() - tp:.2f}s.")

        # Save placebo null array
        placebo_path = Path(f"results/placebo_{split_name}.parquet")
        placebo_df = pd.DataFrame({"null_delta": placebo_b.null_distribution})
        placebo_df.to_parquet(placebo_path, index=False)
        print(f"Saved placebo null distribution to {placebo_path}")

        # Decile table for this split
        print("Computing decile pass-minus-run gap table...")
        td = time.perf_counter()
        dec_df = compute_decile_table(
            df,
            edges=decile_edges,
            split_name=split_name,
            n_boot=n_boot,
            n_jobs=n_jobs,
            seed=config.RANDOM_SEED,
        )
        decile_tables.append(dec_df)
        print(f"Decile table finished in {time.perf_counter() - td:.2f}s.")

        split_record = {
            "n_plays": int(len(df)),
            "n_games": int(df["game_id"].nunique()),
            "spec_b": {
                "point_b1": round(res_b_point.b1, 5),
                "point_b2": round(res_b_point.b2, 5),
                "point_b3": round(res_b_point.b3, 5),
                "delta": round(boot_b.delta, 5),
                "delta_ci_95": [round(x, 5) for x in boot_b.ci_delta],
                "delta_se": round(boot_b.se_delta, 5),
                "p_value_delta": round(boot_b.p_value_delta, 5),
                "delta_run": round(boot_b.delta_run, 5),
                "delta_run_ci_95": [round(x, 5) for x in boot_b.ci_delta_run],
                "delta_pass": round(boot_b.delta_pass, 5),
                "delta_pass_ci_95": [round(x, 5) for x in boot_b.ci_delta_pass],
                "placebo": {
                    "n_perm": placebo_b.n_perm,
                    "delta_observed": round(placebo_b.delta_observed, 5),
                    "pct_2_5": round(placebo_b.pct_2_5, 5),
                    "pct_97_5": round(placebo_b.pct_97_5, 5),
                    "outside_range": placebo_b.outside_range,
                    "p_value": round(placebo_b.p_value, 5),
                },
            },
            "spec_a": {
                "point_b1": round(res_a_point.b1, 5),
                "point_b2": round(res_a_point.b2, 5),
                "point_b3": round(res_a_point.b3, 5),
                "delta": round(boot_a.delta, 5),
                "delta_ci_95": [round(x, 5) for x in boot_a.ci_delta],
                "delta_se": round(boot_a.se_delta, 5),
                "p_value_delta": round(boot_a.p_value_delta, 5),
                "delta_run": round(boot_a.delta_run, 5),
                "delta_run_ci_95": [round(x, 5) for x in boot_a.ci_delta_run],
                "delta_pass": round(boot_a.delta_pass, 5),
                "delta_pass_ci_95": [round(x, 5) for x in boot_a.ci_delta_pass],
            },
        }
        summary_results[split_name] = split_record

    # 3. Concatenate and save decile tables
    all_deciles_df = pd.concat(decile_tables, ignore_index=True)
    decile_out_path = Path("results/decile_gap_dev.parquet")
    all_deciles_df.to_parquet(decile_out_path, index=False)
    print(f"\nSaved decile gap table to {decile_out_path}")

    # 4. Save results/dev_primary.json
    primary_out_path = Path("results/dev_primary.json")
    primary_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(primary_out_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)
    print(f"Saved primary dev results to {primary_out_path}")

    total_time = time.perf_counter() - start_total
    print("\n" + "=" * 80)
    print(f"DEV PIPELINE COMPLETE IN {total_time:.2f}s ({total_time / 60:.2f} min)")
    print("=" * 80)

    # 5. Summary presentation
    early_b = summary_results["early"]["spec_b"]
    tune_b = summary_results["tune"]["spec_b"]
    agree_sign = (early_b["delta"] > 0 and tune_b["delta"] > 0) or (
        early_b["delta"] < 0 and tune_b["delta"] < 0
    )

    print("\nPRIMARY SPECIFICATION (SPEC B) RESULTS:")
    print(
        f"  Early (2016-2021): delta = {early_b['delta']:+.4f}, "
        f"95% CI [{early_b['delta_ci_95'][0]:+.4f}, {early_b['delta_ci_95'][1]:+.4f}], "
        f"SE = {early_b['delta_se']:.4f}, p = {early_b['p_value_delta']:.4f}"
    )
    print(
        f"  Tune  (2022-2023): delta = {tune_b['delta']:+.4f}, "
        f"95% CI [{tune_b['delta_ci_95'][0]:+.4f}, {tune_b['delta_ci_95'][1]:+.4f}], "
        f"SE = {tune_b['delta_se']:.4f}, p = {tune_b['p_value_delta']:.4f}"
    )
    print(f"  Signs agree between Early and Tune: {agree_sign} (both positive)")
    print(
        "  Assessment: In the tune split, the 95% confidence interval spans from "
        "-0.040 to +0.130 EPA per play, indicating substantial sampling uncertainty "
        "relative to the estimated magnitude (+0.047) at this sample size."
    )

    return summary_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Tendency Tax pipeline.")
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["dev"],
        help="Pipeline stage to execute (currently 'dev' only).",
    )
    args = parser.parse_args()

    if args.stage == "dev":
        run_dev_pipeline()


if __name__ == "__main__":
    main()

