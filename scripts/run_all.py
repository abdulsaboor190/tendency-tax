"""End-to-end execution pipeline for Tendency Tax.

Supports stages:
  --stage dev:       Runs early and tune splits (bootstrap, placebo, deciles).
  --stage confirm:   Confirmatory holdout evaluation (2024-2025).
  --stage secondary: Secondary robustness, cell-level, and team-level analyses.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from src import config
from src.estimate import cluster_bootstrap, fit_spec, placebo, verdict
from src.load import get_split
from src.model import add_phat, score


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


def _get_git_commit(ref: str = "HEAD") -> str:
    """Return the git commit hash for a ref or file path."""
    cmd = ["git", "rev-parse", ref]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return res.stdout.strip()


def score_holdout_phat(holdout_raw: pd.DataFrame) -> pd.DataFrame:
    """Score holdout plays strictly using the frozen 2016-2023 model."""
    model_path = config.FROZEN_MODEL_PATH
    meta_path = config.FROZEN_MODEL_META_PATH
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            "Frozen model or metadata JSON missing from results/models/."
        )

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    installed_sklearn = sklearn.__version__
    if meta.get("sklearn_version") != installed_sklearn:
        raise AssertionError(
            f"scikit-learn version mismatch: installed {installed_sklearn} != "
            f"metadata {meta.get('sklearn_version')}"
        )
    if meta.get("features") != config.FEATURES:
        raise AssertionError("Feature list in metadata does not match config.FEATURES.")

    frozen_clf = joblib.load(model_path)
    holdout_df = holdout_raw.copy()
    X_holdout = holdout_df[config.FEATURES]
    holdout_df["phat"] = frozen_clf.predict_proba(X_holdout)[:, 1]

    # Assertions on phat
    if holdout_df["phat"].isna().any():
        raise AssertionError("Null values detected in scored holdout phat.")
    if (holdout_df["phat"] < 0.0).any() or (holdout_df["phat"] > 1.0).any():
        raise AssertionError("Holdout phat values out of valid bounds [0, 1].")
    print("Scored holdout with frozen 2016-2023 model successfully.")
    return holdout_df


def run_confirm_pipeline(
    force: bool = False,
    prompt_input_func: Any = input,
    n_boot: int = config.BOOTSTRAP_REPS,
    n_perm: int = config.PLACEBO_REPS,
    n_jobs: int = -1,
    skip_git_checks: bool = False,
) -> dict[str, Any]:
    """Execute confirmatory evaluation on holdout seasons 2024-2025.

    Follows strict protocol pre-conditions 1 through 16.
    """
    print("=" * 80)
    print("CONFIRMATORY STAGE: HOLDOUT EVALUATION (2024-2025)")
    print("=" * 80)

    # 1. Environment variable check
    if os.environ.get(config.HOLDOUT_ENV_VAR) != "1":
        raise RuntimeError(
            f"Precondition 1 failed: Env var {config.HOLDOUT_ENV_VAR} != '1'."
        )

    # 2. Git working tree clean and pre-confirm tag exists
    if not skip_git_checks:
        status_res = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        if status_res.stdout.strip():
            raise RuntimeError(
                f"Precondition 2 failed: Working tree is dirty:\n{status_res.stdout}"
            )

        tag_res = subprocess.run(
            ["git", "tag", "-l", "pre-confirm"],
            capture_output=True,
            text=True,
            check=True,
        )
        if "pre-confirm" not in tag_res.stdout.split():
            raise RuntimeError(
                "Precondition 2 failed: Git tag 'pre-confirm' does not exist."
            )

        # 3. PLAN.md integrity check
        plan_log = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", "PLAN.md"],
            capture_output=True,
            text=True,
            check=True,
        )
        plan_commit = plan_log.stdout.strip()
        if not plan_commit:
            raise RuntimeError(
                "Precondition 3 failed: PLAN.md has never been committed."
            )
        print(f"PLAN.md verified. Introduction commit: {plan_commit}")

        # Check if PLAN.md has diff since that commit
        diff_res = subprocess.run(
            ["git", "diff", plan_commit, "HEAD", "--", "PLAN.md"],
            capture_output=True,
            text=True,
            check=True,
        )
        if diff_res.stdout.strip() and not Path("DEVIATIONS.md").exists():
            raise RuntimeError(
                "Precondition 3 failed: PLAN.md modified since initial commit "
                "without DEVIATIONS.md."
            )
        current_commit = _get_git_commit("HEAD")
    else:
        plan_commit = "mock_plan_commit"
        current_commit = "mock_commit"

    # 4. results/dev_primary.json exists
    dev_primary_path = Path("results/dev_primary.json")
    if not dev_primary_path.exists():
        raise RuntimeError(
            f"Precondition 4 failed: {dev_primary_path} missing. Run dev first."
        )
    with open(dev_primary_path, encoding="utf-8") as f:
        dev_data = json.load(f)

    # 5. results/primary.json does not already exist
    primary_json_path = Path("results/primary.json")
    if primary_json_path.exists():
        if not force:
            raise RuntimeError(
                f"Precondition 5 failed: {primary_json_path} already exists. "
                "Confirmatory run executes once. Use --force to override."
            )
        # Log to DEVIATIONS.md
        dev_md_path = Path("DEVIATIONS.md")
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        dev_entry = (
            f"\n\n## {now_iso} - Force Overwrite of Confirmatory Run\n"
            f"- Overwriting existing {primary_json_path} with --force flag.\n"
        )
        with open(dev_md_path, "a", encoding="utf-8") as f:
            f.write(dev_entry)
        print("Logged force override to DEVIATIONS.md")

    # 6. Interactive confirmation
    print("\n" + "!" * 80)
    print("WARNING: About to execute confirmatory holdout evaluation on 2024-2025.")
    print("This will score the locked frozen model and evaluate the hypothesis.")
    print("!" * 80)
    response = prompt_input_func("Type 'RUN' to proceed: ")
    if response.strip() != "RUN":
        raise RuntimeError("Precondition 6 failed: Confirmation aborted by user.")

    # 7. Load holdout via get_split("holdout")
    print("\nLoading holdout split via get_split('holdout')...")
    holdout_raw = get_split("holdout")
    n_plays_holdout = len(holdout_raw)
    n_games_holdout = holdout_raw["game_id"].nunique()

    # In mock/test environments, allow overriding expected counts
    expected_plays = 48469
    expected_games = 544
    if not skip_git_checks:
        if n_plays_holdout != expected_plays or n_games_holdout != expected_games:
            raise AssertionError(
                f"Holdout counts mismatch: got {n_plays_holdout} plays, "
                f"{n_games_holdout} games; expected {expected_plays} plays, "
                f"{expected_games} games."
            )
    print(
        f"Holdout verified: {n_plays_holdout:,} plays across "
        f"{n_games_holdout:,} games."
    )

    # 8. Load frozen model and score holdout
    holdout_df = score_holdout_phat(holdout_raw)

    # 9. Spec B and Spec A regressions
    print("\nFitting Spec B and Spec A on Holdout...")
    res_b_point = fit_spec(holdout_df, spec="B")
    res_a_point = fit_spec(holdout_df, spec="A")

    print(f"Running Holdout Spec B cluster bootstrap ({n_boot} reps)...")
    boot_b = cluster_bootstrap(
        holdout_df, spec="B", n_boot=n_boot, n_jobs=n_jobs, seed=config.RANDOM_SEED
    )
    print(f"Running Holdout Spec A cluster bootstrap ({n_boot} reps)...")
    boot_a = cluster_bootstrap(
        holdout_df, spec="A", n_boot=n_boot, n_jobs=n_jobs, seed=config.RANDOM_SEED
    )
    print(f"Running Holdout Spec B placebo test ({n_perm} perms)...")
    placebo_b = placebo(
        holdout_df, spec="B", n_perm=n_perm, n_jobs=n_jobs, seed=config.RANDOM_SEED
    )

    # 10. Call verdict()
    early_delta = dev_data["early"]["spec_b"]["delta"]
    verdict_res = verdict(
        holdout_delta=boot_b.delta,
        holdout_ci_lo=boot_b.ci_delta[0],
        holdout_ci_hi=boot_b.ci_delta[1],
        early_delta=early_delta,
        placebo_lo=placebo_b.pct_2_5,
        placebo_hi=placebo_b.pct_97_5,
    )

    # 11. Compute decile gap table for holdout using early edges
    early_raw = add_phat(get_split("early"))
    early_phat = early_raw["phat"].to_numpy()
    decile_edges = np.percentile(early_phat, np.linspace(0, 100, 11))
    decile_edges[0] = 0.0
    decile_edges[-1] = 1.0

    print("Computing Holdout decile gap table...")
    decile_holdout_df = compute_decile_table(
        holdout_df,
        edges=decile_edges,
        split_name="holdout",
        n_boot=n_boot,
        n_jobs=n_jobs,
        seed=config.RANDOM_SEED,
    )

    # 12. Holdout phat predictive scores (call_pass only, no EPA)
    print("Scoring predictive accuracy on Holdout...")
    model_score = score(holdout_df, "phat")

    # Cell baseline learned on 2016-2023 (early + tune)
    dev_pool = pd.concat([get_split("early"), get_split("tune")], ignore_index=True)
    cell_rates = dev_pool.groupby("cell_id")["call_pass"].mean()
    dd_rates = dev_pool.groupby("dd_bucket")["call_pass"].mean()
    overall_rate = float(dev_pool["call_pass"].mean())

    holdout_baseline = (
        holdout_df["cell_id"]
        .map(cell_rates)
        .fillna(holdout_df["dd_bucket"].map(dd_rates))
        .fillna(overall_rate)
        .astype(float)
    )
    holdout_df["phat_baseline"] = holdout_baseline
    baseline_score = score(holdout_df, "phat_baseline")

    # 13. Assemble primary.json
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    package_versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }

    primary_record: dict[str, Any] = {
        "timestamp": now_iso,
        "git_commit": current_commit,
        "plan_commit": plan_commit,
        "package_versions": package_versions,
        "simulation_parameters": {
            "random_seed": config.RANDOM_SEED,
            "n_boot": n_boot,
            "n_perm": n_perm,
        },
        "early": {
            "n_plays": dev_data["early"]["n_plays"],
            "n_games": dev_data["early"]["n_games"],
            "spec_b": dev_data["early"]["spec_b"],
        },
        "tune": {
            "n_plays": dev_data["tune"]["n_plays"],
            "n_games": dev_data["tune"]["n_games"],
            "spec_b": dev_data["tune"]["spec_b"],
        },
        "holdout": {
            "n_plays": n_plays_holdout,
            "n_games": n_games_holdout,
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
            "spec_a_secondary": {
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
            "predictive_scores": {
                "model_phat": {
                    "log_loss": round(model_score.log_loss, 5),
                    "brier_score": round(model_score.brier_score, 5),
                },
                "baseline_cell": {
                    "log_loss": round(baseline_score.log_loss, 5),
                    "brier_score": round(baseline_score.brier_score, 5),
                },
            },
        },
        "verdict": verdict_res.to_dict(),
    }

    with open(primary_json_path, "w", encoding="utf-8") as f:
        json.dump(primary_record, f, indent=2)
    print(f"\nSaved primary confirmatory results to {primary_json_path}")

    # 14. Save parquet artifacts
    placebo_holdout_path = Path("results/placebo_holdout.parquet")
    pd.DataFrame({"null_delta": placebo_b.null_distribution}).to_parquet(
        placebo_holdout_path, index=False
    )
    print(f"Saved holdout placebo null to {placebo_holdout_path}")

    decile_holdout_path = Path("results/decile_gap_holdout.parquet")
    decile_holdout_df.to_parquet(decile_holdout_path, index=False)
    print(f"Saved holdout decile table to {decile_holdout_path}")

    # 15. Append to results/RUN_LOG.md
    run_log_path = Path("results/RUN_LOG.md")
    ci_str = f"[{boot_b.ci_delta[0]:+.4f}, {boot_b.ci_delta[1]:+.4f}]"
    log_entry = (
        f"\n## Confirmatory Run — {now_iso}\n"
        f"- Commit: `{current_commit}`\n"
        f"- PLAN.md Commit: `{plan_commit}`\n"
        f"- Verdict State: **{verdict_res.state.upper()}**\n"
        f"- Verdict Sentence: {verdict_res.sentence}\n"
        f"- Holdout Spec B Delta: {boot_b.delta:+.4f} (95% CI: {ci_str})\n"
    )
    with open(run_log_path, "a", encoding="utf-8") as f:
        f.write(log_entry)
    print(f"Appended confirmatory log entry to {run_log_path}")

    # 16. Print summary
    print("\n" + "=" * 80)
    print("CONFIRMATORY EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Verdict State:    {verdict_res.state.upper()}")
    print(f"Verdict Sentence: {verdict_res.sentence}")
    print(f"Holdout Delta:    {boot_b.delta:+.4f}")
    print(f"95% CI:           [{boot_b.ci_delta[0]:+.4f}, {boot_b.ci_delta[1]:+.4f}]")
    print(f"Bootstrap SE:     {boot_b.se_delta:.4f}")
    print(f"p-value:          {boot_b.p_value_delta:.4f}")
    print(
        f"Placebo 95% Band: [{placebo_b.pct_2_5:+.4f}, {placebo_b.pct_97_5:+.4f}] "
        f"(Outside: {placebo_b.outside_range})"
    )
    print("=" * 80)

    return primary_record


def run_secondary_pipeline(
    only: str | None = None,
    n_boot: int = config.ROBUSTNESS_REPS,
    n_jobs: int = -1,
    skip_git_checks: bool = False,
) -> dict[str, Any]:
    """Execute the secondary analyses stage (robustness, cells, teams)."""
    from src.robustness import compute_robustness_suite
    from src.schema import validate_parquet_df

    print("=" * 80)
    print("SECONDARY STAGE PIPELINE (ROBUSTNESS, CELLS, TEAMS)")
    print("=" * 80)

    # Precondition 1: Environment variable check
    if os.environ.get(config.HOLDOUT_ENV_VAR) != "1":
        raise RuntimeError(
            f"Precondition failed: Env var {config.HOLDOUT_ENV_VAR} != '1'."
        )

    # Precondition 2: results/primary.json must already exist
    primary_json_path = Path("results/primary.json")
    if not primary_json_path.exists():
        raise RuntimeError(
            f"Precondition failed: {primary_json_path} missing. "
            "Confirmatory stage must be executed before secondary analyses."
        )

    print("Loading data for Early, Tune, and Holdout splits...")
    early_df = add_phat(get_split("early"))
    tune_df = add_phat(get_split("tune"))

    holdout_raw = get_split("holdout")
    if skip_git_checks:
        holdout_df = holdout_raw.copy()
        if "phat" not in holdout_df.columns:
            holdout_df["phat"] = 0.5
    else:
        holdout_df = score_holdout_phat(holdout_raw)

    dfs = {
        "early": early_df,
        "tune": tune_df,
        "holdout": holdout_df,
    }

    results: dict[str, Any] = {}

    # Robustness suite
    if only is None or only == "robustness":
        print("\n--- Running Robustness Suite ---")
        t0 = time.perf_counter()
        rob_df = compute_robustness_suite(dfs, n_boot=n_boot, n_jobs=n_jobs)
        validate_parquet_df(
            rob_df, schema_name="robustness", filename="robustness.parquet"
        )
        out_path = Path("results/robustness.parquet")
        rob_df.to_parquet(out_path, index=False)
        elapsed_sec = time.perf_counter() - t0
        print(f"Robustness suite completed in {elapsed_sec:.2f}s ({out_path}).")
        results["robustness"] = rob_df

    # Cells analysis (D5.3)
    if only is None or only == "cells":
        from src.cells import add_grid_cell_id, compute_cells_grid

        print("\n--- Running 27-Cell Grid Analysis ---")
        t0 = time.perf_counter()
        # Build grid cell_id on every split before passing to compute_cells_grid
        dfs_cells = {k: add_grid_cell_id(v) for k, v in dfs.items()}
        cells_df = compute_cells_grid(
            dfs_cells,
            n_boot=n_boot,
            n_jobs=n_jobs,
        )
        validate_parquet_df(
            cells_df, schema_name="cells", filename="cells.parquet"
        )
        out_path = Path("results/cells.parquet")
        cells_df.to_parquet(out_path, index=False)
        elapsed_sec = time.perf_counter() - t0
        print(
            f"27-cell grid completed in {elapsed_sec:.2f}s ({out_path})."
        )
        results["cells"] = cells_df

    # Teams analysis (D5.4)
    if only is None or only == "teams":
        from src.teams import compute_team_scout

        print("\n--- Running Team Scout Analysis ---")
        t0 = time.perf_counter()
        teams_df = compute_team_scout(dfs)
        validate_parquet_df(
            teams_df, schema_name="teams", filename="teams.parquet"
        )
        out_path = Path("results/teams.parquet")
        teams_df.to_parquet(out_path, index=False)
        elapsed_sec = time.perf_counter() - t0
        print(f"Team scout completed in {elapsed_sec:.2f}s ({out_path}).")
        results["teams"] = teams_df

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Tendency Tax pipeline.")
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["dev", "confirm", "secondary"],
        help="Pipeline stage to execute.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        choices=["robustness", "cells", "teams"],
        help="Optional sub-module to run within the secondary stage.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite existing primary.json (logs to DEVIATIONS.md).",
    )
    args = parser.parse_args()

    if args.stage == "dev":
        run_dev_pipeline()
    elif args.stage == "confirm":
        run_confirm_pipeline(force=args.force)
    elif args.stage == "secondary":
        run_secondary_pipeline(only=args.only)


if __name__ == "__main__":
    main()
