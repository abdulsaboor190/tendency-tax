"""Simulation recovery test and power estimation for Tendency Tax.

Simulates known effects (true_delta in {0.00, 0.03, 0.05, 0.10}) using the real
tune split design (call_pass, phat, dd_bucket, game_id).
EPA parameters (mu_dd, c, noise SD, game random effect SD) are calibrated
from the early split.
Estimator: Spec B with CR1 game-clustered sandwich standard errors.
"""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import config
from src.estimate import _demean_by_group, cluster_bootstrap
from src.load import get_split
from src.model import add_phat


def fit_spec_b_with_cr1_se(
    epa: np.ndarray,
    call: np.ndarray,
    phat: np.ndarray,
    groups: np.ndarray,
    game_inv: np.ndarray,
    n_games: int,
) -> tuple[float, float, float, float]:
    """Fit Spec B and compute CR1 game-clustered analytic standard error for delta.

    Parameters
    ----------
    epa : np.ndarray
        Simulated EPA vector.
    call : np.ndarray
        Call indicator (1=pass, 0=run).
    phat : np.ndarray
        Pre-snap pass probability.
    groups : np.ndarray
        Group labels for dd_bucket.
    game_inv : np.ndarray
        Integer array (0..n_games-1) mapping each row to a unique game.
    n_games : int
        Number of unique games.

    Returns
    -------
    tuple[float, float, float, float]
        b1, b2, b3, se_delta
    """
    n = len(epa)
    interaction = call * phat

    # Groups encoding
    _, group_codes = np.unique(groups, return_inverse=True)
    n_groups = len(np.unique(group_codes))

    # Within-group demeaning
    y_tilde = _demean_by_group(epa, group_codes, n_groups)
    c_tilde = _demean_by_group(call, group_codes, n_groups)
    p_tilde = _demean_by_group(phat, group_codes, n_groups)
    cp_tilde = _demean_by_group(interaction, group_codes, n_groups)

    # Design matrix X: [c_tilde, p_tilde, cp_tilde] (N x 3)
    X = np.column_stack([c_tilde, p_tilde, cp_tilde])

    # OLS coefficients
    # (X'X)^(-1) X' y
    XtX = X.T @ X
    inv_XtX = np.linalg.inv(XtX)
    beta = inv_XtX @ (X.T @ y_tilde)
    b1, b2, b3 = float(beta[0]), float(beta[1]), float(beta[2])

    # Residuals
    residuals = y_tilde - X @ beta

    # Clustered sandwich meat:
    # S = sum_{g} (X_g' e_g) (X_g' e_g)'
    # X_e = X * residuals[:, None]
    X_e = X * residuals[:, np.newaxis]

    # Sum X_e within each game cluster using np.bincount
    meat_g = np.empty((n_games, 3), dtype=np.float64)
    for k in range(3):
        meat_g[:, k] = np.bincount(game_inv, weights=X_e[:, k], minlength=n_games)

    meat = meat_g.T @ meat_g

    # CR1 small-sample adjustment: (G / (G - 1)) * ((N - 1) / (N - K))
    # where G = n_games, K = 3
    df_adj = (n_games / (n_games - 1.0)) * ((n - 1.0) / (n - 3.0))
    V_clust = df_adj * (inv_XtX @ meat @ inv_XtX)

    # delta = -b3 / 2 => Var(delta) = 0.25 * Var(b3)
    var_b3 = V_clust[2, 2]
    var_delta = 0.25 * var_b3
    se_delta = float(np.sqrt(max(0.0, var_delta)))

    return b1, b2, b3, se_delta


def estimate_calibration_params(early_df: pd.DataFrame) -> dict[str, Any]:
    """Estimate mu_dd, c, noise SD, and game random-effect SD from early split."""
    # mu_dd: mean EPA per dd_bucket
    mu_dd_map = early_df.groupby("dd_bucket")["epa"].mean().to_dict()

    # Pass-minus-run gap c (controlling for dd_bucket)
    # epa ~ dd_bucket + c * call_pass
    group_means = early_df["dd_bucket"].map(mu_dd_map).to_numpy()
    epa_adj = early_df["epa"].to_numpy() - group_means
    call = early_df["call_pass"].to_numpy()

    # Regression of epa_adj on call with demeaned call
    mean_call_by_bucket = early_df.groupby("dd_bucket")["call_pass"].transform("mean")
    call_demeaned = call - mean_call_by_bucket.to_numpy()
    c_est = float(
        np.dot(call_demeaned, epa_adj) / np.dot(call_demeaned, call_demeaned)
    )

    # Residuals after dd_bucket and call effect:
    resid = epa_adj - c_est * call_demeaned

    # Game-level random effects
    early_df_temp = pd.DataFrame({"game_id": early_df["game_id"], "resid": resid})
    game_means = early_df_temp.groupby("game_id")["resid"].mean().to_numpy()
    game_re_sd = float(np.std(game_means, ddof=1))

    # Intra-game noise
    # var(resid) = var(game_re) + var(noise)
    total_var = float(np.var(resid, ddof=1))
    noise_var = max(0.01, total_var - game_re_sd**2)
    noise_sd = float(np.sqrt(noise_var))

    return {
        "mu_dd": mu_dd_map,
        "c": c_est,
        "game_re_sd": game_re_sd,
        "noise_sd": noise_sd,
    }


def run_recovery_simulation(
    tune_df: pd.DataFrame,
    calib: dict[str, Any],
    true_deltas: list[float] = [0.00, 0.03, 0.05, 0.10],
    n_sims: int = 300,
    seed: int = config.RANDOM_SEED,
    check_bootstrap: bool = True,
) -> dict[str, Any]:
    """Run simulation recovery across true_delta values."""
    n_rows = len(tune_df)
    call = tune_df["call_pass"].to_numpy(dtype=np.float64)
    phat = tune_df["phat"].to_numpy(dtype=np.float64)
    groups = tune_df["dd_bucket"].to_numpy()

    # Surprise variable: (1 - phat) if call == 1 else phat
    surprise = np.where(call == 1.0, 1.0 - phat, phat)

    # Base EPA: mu_dd + c * call
    mu_dd_vals = np.array(
        [calib["mu_dd"].get(g, 0.0) for g in groups], dtype=np.float64
    )
    base_epa = mu_dd_vals + calib["c"] * call

    # Game mapping
    game_ids = tune_df["game_id"].to_numpy()
    unique_games, game_inv = np.unique(game_ids, return_inverse=True)
    n_games = len(unique_games)

    root_seq = np.random.SeedSequence(seed)
    delta_seqs = root_seq.spawn(len(true_deltas))

    out_records = {}

    for d_idx, true_d in enumerate(true_deltas):
        sim_seqs = delta_seqs[d_idx].spawn(n_sims)
        deltas_est = np.empty(n_sims, dtype=np.float64)
        ses_est = np.empty(n_sims, dtype=np.float64)
        covered = np.empty(n_sims, dtype=bool)
        significant_correct = np.empty(n_sims, dtype=bool)

        for s in range(n_sims):
            rng = np.random.default_rng(sim_seqs[s])

            # Draw game random effects and play noise
            game_effects = rng.normal(0.0, calib["game_re_sd"], size=n_games)
            noise = rng.normal(0.0, calib["noise_sd"], size=n_rows)

            epa_sim = base_epa + true_d * surprise + game_effects[game_inv] + noise

            # Fit with CR1 SE
            _, _, b3, se_d = fit_spec_b_with_cr1_se(
                epa=epa_sim,
                call=call,
                phat=phat,
                groups=groups,
                game_inv=game_inv,
                n_games=n_games,
            )
            d_est = -b3 / 2.0
            deltas_est[s] = d_est
            ses_est[s] = se_d

            # 95% Normal interval: [d_est - 1.96 * se_d, d_est + 1.96 * se_d]
            ci_low = d_est - 1.96 * se_d
            ci_high = d_est + 1.96 * se_d
            covered[s] = (ci_low <= true_d <= ci_high)

            # Power definition: interval excludes 0 AND estimate has right sign
            if true_d == 0.0:
                # Under null, power is rate of rejecting 0 (Type I error)
                significant_correct[s] = (ci_low > 0.0 or ci_high < 0.0)
            elif true_d > 0.0:
                significant_correct[s] = (ci_low > 0.0)
            else:
                significant_correct[s] = (ci_high < 0.0)

        mean_est = float(np.mean(deltas_est))
        bias = float(mean_est - true_d)
        mean_se = float(np.mean(ses_est))
        coverage = float(np.mean(covered))
        power = float(np.mean(significant_correct))

        record = {
            "true_delta": true_d,
            "n_sims": n_sims,
            "mean_estimate": round(mean_est, 5),
            "bias": round(bias, 5),
            "mean_se": round(mean_se, 5),
            "coverage_95": round(coverage, 4),
            "power": round(power, 4),
        }

        # On the first simulated dataset, run cluster_bootstrap and compare SE
        if check_bootstrap:
            # Recreate first simulation dataset
            rng0 = np.random.default_rng(sim_seqs[0])
            ge0 = rng0.normal(0.0, calib["game_re_sd"], size=n_games)
            eps0 = rng0.normal(0.0, calib["noise_sd"], size=n_rows)
            epa0 = base_epa + true_d * surprise + ge0[game_inv] + eps0

            sim_df = tune_df.copy()
            sim_df["epa"] = epa0
            boot_res = cluster_bootstrap(
                sim_df, spec="B", n_boot=100, n_jobs=-1, seed=42
            )
            boot_se = boot_res.se_delta
            analytic_se_0 = ses_est[0]
            rel_diff = abs(boot_se - analytic_se_0) / analytic_se_0
            record["bootstrap_se_check"] = {
                "analytic_se": round(analytic_se_0, 5),
                "bootstrap_se": round(boot_se, 5),
                "relative_diff": round(rel_diff, 4),
                "within_15_pct": bool(rel_diff <= 0.15),
            }

        out_records[f"delta_{true_d:.2f}"] = record

    return out_records


def main() -> None:
    """Run full simulation recovery across 300 simulations per true_delta."""
    print("Loading Early and Tune splits...")
    early_df = add_phat(get_split("early"))
    tune_df = add_phat(get_split("tune"))

    print("\nEstimating simulation calibration parameters from Early split...")
    calib = estimate_calibration_params(early_df)
    print(f"  Pass-minus-run gap (c): {calib['c']:+.4f}")
    print(f"  Game random-effect SD:  {calib['game_re_sd']:.4f}")
    print(f"  Play-level noise SD:    {calib['noise_sd']:.4f}")

    print("\nRunning full simulation recovery (300 reps per true_delta)...")
    t0 = time.perf_counter()
    results = run_recovery_simulation(
        tune_df=tune_df,
        calib=calib,
        true_deltas=[0.00, 0.03, 0.05, 0.10],
        n_sims=300,
        seed=config.RANDOM_SEED,
        check_bootstrap=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"Simulation completed in {elapsed:.1f}s.")

    # Determine expected SE at holdout size and smallest effect with >= 80% power
    all_ses = [v["mean_se"] for v in results.values()]
    expected_se_holdout = float(np.mean(all_ses))

    # Required effect size for 80% power (z_alpha = 1.96, z_beta = 0.84):
    # delta_80 = (1.96 + 0.84) * SE = 2.80 * SE
    min_delta_80_formula = round(2.80 * expected_se_holdout, 4)

    # From simulation records
    min_delta_80_sim = None
    for d_key, r in results.items():
        if r["power"] >= 0.80 and r["true_delta"] > 0:
            min_delta_80_sim = r["true_delta"]
            break

    summary = {
        "calibration": {
            "c_pass_minus_run": round(calib["c"], 4),
            "game_re_sd": round(calib["game_re_sd"], 4),
            "noise_sd": round(calib["noise_sd"], 4),
        },
        "simulation_results": results,
        "holdout_expected_se": round(expected_se_holdout, 4),
        "smallest_delta_80_power_formula": min_delta_80_formula,
        "smallest_delta_80_power_sim": min_delta_80_sim,
    }

    output_path = Path("results/power.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved results to {output_path}")

    # Print report table
    print("\n" + "=" * 80)
    print("SIMULATION RECOVERY & POWER REPORT (TUNE SPLIT DESIGN)")
    print("=" * 80)
    print(
        f"{'True Delta':<12} | {'Mean Est':<10} | {'Bias':<10} | "
        f"{'Mean SE':<10} | {'95% Coverage':<14} | {'Power':<8}"
    )
    print("-" * 80)
    for r in results.values():
        print(
            f"{r['true_delta']:<12.2f} | "
            f"{r['mean_estimate']:<10.4f} | "
            f"{r['bias']:<10.4f} | "
            f"{r['mean_se']:<10.4f} | "
            f"{r['coverage_95']:<14.2%} | "
            f"{r['power']:<8.2%}"
        )
    print("=" * 80)
    print(f"Expected SE at holdout size: {expected_se_holdout:.4f} EPA/play")
    print(f"Smallest effect with 80% power: ~{min_delta_80_formula:.4f} EPA/play")


if __name__ == "__main__":
    main()
