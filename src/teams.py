"""Team scout data and empirical-Bayes shrinkage (Deliverable D5.4).

Definitions:
  - A "lean play" has phat >= config.LEAN_HIGH (0.65) or phat <= config.LEAN_LOW (0.35).
  - "Followed" = pass call (call_pass == 1) when phat >= 0.65,
                 or run call (call_pass == 0) when phat <= 0.35.
  - "Broke" = the opposite call on a lean play (run call when phat >= 0.65,
              or pass call when phat <= 0.35).
  - d_raw = mean EPA on broke plays MINUS mean EPA on followed plays.
            A POSITIVE d_raw means going against the situational lean is associated
            with higher EPA (same sign convention as primary delta).
  - se_raw = Welch standard error:
             sqrt(var_break / n_break + var_follow / n_follow).
  - Team-seasons with n_break < 2 or n_follow < 2 have d_raw = NaN, se_raw = NaN,
    and are excluded from shrinkage fitting, but remain listed in the table.
  - warn_low_n = True when n_break < 60.
  - All rows are tagged with label: "secondary".

Empirical-Bayes Shrinkage (Normal-Normal):
  Fit SEPARATELY within each split (early, tune, holdout):
  - Inverse-variance weights: w_i = 1 / se_raw_i^2
  - League mean mu: mu = sum(w_i * d_raw_i) / sum(w_i)
  - DerSimonian-Laird between-team variance tau^2:
      Q = sum(w_i * (d_raw_i - mu)^2)
      c = sum(w_i) - sum(w_i^2) / sum(w_i)
      tau^2 = max(0.0, (Q - (k - 1)) / c)
  - shrink_factor_i = tau^2 / (tau^2 + se_raw_i^2)
  - d_shrunk_i = mu + shrink_factor_i * (d_raw_i - mu)
  - posterior_se_i = sqrt(tau^2 * se_raw_i^2 / (tau^2 + se_raw_i^2))
  - shrunk_lo_i = d_shrunk_i - 1.96 * posterior_se_i
  - shrunk_hi_i = d_shrunk_i + 1.96 * posterior_se_i
  If tau^2 == 0, shrink_factor is 0.0, d_shrunk is mu, and posterior_se is 0.0.

Franchise Code Consistency:
  Historical relocations (OAK -> LV, SD -> LAC, STL -> LA) are mapped to their
  current franchise abbreviations if found. In nflverse data (2016-present),
  teams are already largely normalized to modern 3-letter codes, but any
  legacy abbreviations are explicitly mapped by FRANCHISE_MAPPING.
"""

from typing import Any

import numpy as np
import pandas as pd

from src import config

# Franchise normalization mapping for relocated teams
FRANCHISE_MAPPING: dict[str, str] = {
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LA",
}


def normalize_team_codes(teams: pd.Series) -> pd.Series:
    """Normalize team franchise codes to modern standard abbreviations."""
    return teams.replace(FRANCHISE_MAPPING)


def dersimonian_laird_tau2(
    d_raw: np.ndarray,
    se_raw: np.ndarray,
) -> tuple[float, float]:
    """Compute league mean mu and DerSimonian-Laird between-study variance tau^2.

    Parameters
    ----------
    d_raw : np.ndarray
        Point estimates for each team-season (k valid teams).
    se_raw : np.ndarray
        Welch standard errors for each team-season.

    Returns
    -------
    tuple[float, float]
        (mu, tau2) where mu is inverse-variance weighted mean and tau2 is floored at 0.
    """
    k = len(d_raw)
    if k < 2:
        # Cannot estimate variance with fewer than 2 units
        mu = float(d_raw[0]) if k == 1 else 0.0
        return mu, 0.0

    w = 1.0 / (se_raw**2)
    sum_w = float(np.sum(w))
    mu = float(np.sum(w * d_raw) / sum_w)

    # Cochran's Q
    q = float(np.sum(w * (d_raw - mu) ** 2))
    sum_w2 = float(np.sum(w**2))
    c = sum_w - (sum_w2 / sum_w)

    if c <= 0:
        tau2 = 0.0
    else:
        tau2 = max(0.0, (q - (k - 1)) / c)

    return mu, tau2


def apply_eb_shrinkage(df_team_seasons: pd.DataFrame) -> pd.DataFrame:
    """Apply empirical-Bayes shrinkage to team-seasons within a single split."""
    df = df_team_seasons.copy()

    # Pre-populate shrinkage columns
    df["mu_league"] = np.nan
    df["tau2"] = np.nan
    df["shrink_factor"] = np.nan
    df["d_shrunk"] = np.nan
    df["shrunk_lo"] = np.nan
    df["shrunk_hi"] = np.nan

    # Eligible for shrinkage: d_raw and se_raw are finite and non-null
    mask_valid = df["d_raw"].notna() & df["se_raw"].notna() & (df["se_raw"] > 0)
    valid_indices = df[mask_valid].index

    if len(valid_indices) == 0:
        return df

    d_valid = df.loc[valid_indices, "d_raw"].to_numpy(dtype=np.float64)
    se_valid = df.loc[valid_indices, "se_raw"].to_numpy(dtype=np.float64)

    mu, tau2 = dersimonian_laird_tau2(d_valid, se_valid)

    # Assign mu and tau2 to all rows in the split (or at least valid rows)
    # The requirement states: "Store mu and tau2 in each row."
    df["mu_league"] = round(mu, 5)
    df["tau2"] = round(tau2, 5)

    if tau2 == 0.0:
        shrink_factor = np.zeros_like(se_valid)
        d_shrunk = np.full_like(d_valid, mu)
        post_se = np.zeros_like(se_valid)
    else:
        var_tot = tau2 + se_valid**2
        shrink_factor = tau2 / var_tot
        d_shrunk = mu + shrink_factor * (d_valid - mu)
        post_var = (tau2 * se_valid**2) / var_tot
        post_se = np.sqrt(post_var)

    shrunk_lo = d_shrunk - 1.96 * post_se
    shrunk_hi = d_shrunk + 1.96 * post_se

    df.loc[valid_indices, "shrink_factor"] = np.round(shrink_factor, 5)
    df.loc[valid_indices, "d_shrunk"] = np.round(d_shrunk, 5)
    df.loc[valid_indices, "shrunk_lo"] = np.round(shrunk_lo, 5)
    df.loc[valid_indices, "shrunk_hi"] = np.round(shrunk_hi, 5)

    return df


def compute_team_scout_split(
    df_split: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """Compute team-season breakdown and EB shrinkage for a single split."""
    df = df_split.copy()
    df["team"] = normalize_team_codes(df["posteam"].astype(str))

    lean_high = config.LEAN_HIGH
    lean_low = config.LEAN_LOW

    # Classify lean plays
    is_lean_pass = df["phat"] >= lean_high
    is_lean_run = df["phat"] <= lean_low
    is_lean = is_lean_pass | is_lean_run

    df["is_lean"] = is_lean
    # Followed: pass when lean pass, run when lean run
    df["is_follow"] = (
        (is_lean_pass & (df["call_pass"] == 1.0))
        | (is_lean_run & (df["call_pass"] == 0.0))
    )
    # Broke: run when lean pass, pass when lean run
    df["is_break"] = (
        (is_lean_pass & (df["call_pass"] == 0.0))
        | (is_lean_run & (df["call_pass"] == 1.0))
    )

    records: list[dict[str, Any]] = []

    # Group by season, team
    grouped = df.groupby(["season", "team"], sort=True)

    for (season, team), group in grouped:
        n_plays = int(len(group))
        lean_group = group[group["is_lean"]]
        n_lean = int(len(lean_group))
        lean_share = round(float(n_lean / n_plays), 5) if n_plays > 0 else 0.0

        follow_group = lean_group[lean_group["is_follow"]]
        break_group = lean_group[lean_group["is_break"]]

        n_follow = int(len(follow_group))
        n_break = int(len(break_group))
        follow_rate = round(float(n_follow / n_lean), 5) if n_lean > 0 else np.nan

        epa_follow = (
            round(float(follow_group["epa"].mean()), 5)
            if n_follow > 0
            else np.nan
        )
        epa_break = (
            round(float(break_group["epa"].mean()), 5)
            if n_break > 0
            else np.nan
        )

        # d_raw = mean EPA on broke plays MINUS mean EPA on followed plays
        # Excluded if n_break < 2 or n_follow < 2
        if n_break >= 2 and n_follow >= 2:
            d_raw = float(epa_break - epa_follow)
            var_break = float(break_group["epa"].var(ddof=1))
            var_follow = float(follow_group["epa"].var(ddof=1))
            se_raw = float(np.sqrt(var_break / n_break + var_follow / n_follow))
            d_raw = round(d_raw, 5)
            se_raw = round(se_raw, 5)
        else:
            d_raw = np.nan
            se_raw = np.nan

        warn_low_n = bool(n_break < 60)

        records.append({
            "split": split_name,
            "season": int(season),
            "team": str(team),
            "n_plays": n_plays,
            "n_lean": n_lean,
            "lean_share": lean_share,
            "n_follow": n_follow,
            "n_break": n_break,
            "follow_rate": follow_rate,
            "epa_follow": epa_follow,
            "epa_break": epa_break,
            "d_raw": d_raw,
            "se_raw": se_raw,
            "warn_low_n": warn_low_n,
            "label": "secondary",
        })

    df_res = pd.DataFrame(records)
    # Apply empirical-Bayes shrinkage within this split
    df_res = apply_eb_shrinkage(df_res)
    return df_res


def compute_team_scout(
    dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compute team scout data and EB shrinkage across early, tune, holdout splits.

    Parameters
    ----------
    dfs : dict[str, pd.DataFrame]
        Dictionary with splits e.g. 'early', 'tune', 'holdout'.

    Returns
    -------
    pd.DataFrame
        DataFrame matching the schema of results/teams.parquet.
    """
    split_dfs = []
    for split_name in ["early", "tune", "holdout"]:
        if split_name not in dfs:
            continue
        print(f"Computing team scout table for split '{split_name}'...")
        split_df = compute_team_scout_split(dfs[split_name], split_name=split_name)
        split_dfs.append(split_df)

    if not split_dfs:
        raise ValueError("No splits provided for compute_team_scout.")

    final_df = pd.concat(split_dfs, ignore_index=True)

    # Reorder columns to match schema exactly
    ordered_cols = [
        "split", "season", "team", "n_plays", "n_lean", "lean_share",
        "n_follow", "n_break", "follow_rate", "epa_follow", "epa_break",
        "d_raw", "se_raw", "mu_league", "tau2", "shrink_factor",
        "d_shrunk", "shrunk_lo", "shrunk_hi", "warn_low_n", "label"
    ]
    return final_df[ordered_cols]
