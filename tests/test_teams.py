"""Tests for team scout data and empirical-Bayes shrinkage (Deliverable D5.4).

Covers:
  1. Shrinkage pulls larger-SE teams closer to mu.
  2. tau2 = 0 gives shrink_factor 0 and d_shrunk = mu.
  3. Hand-computed DerSimonian-Laird example matches.
  4. Sign convention test on a frame where breaking clearly pays.
  5. Teams with tiny n are listed and flagged (warn_low_n=True).
  6. Stage refuses without preconditions (env var, primary.json).
  7. Output validates against schema (results/teams.parquet).
  8. Relocated franchise codes are mapped properly.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config
from src.schema import validate_parquet_df
from src.teams import (
    apply_eb_shrinkage,
    compute_team_scout,
    compute_team_scout_split,
    dersimonian_laird_tau2,
    normalize_team_codes,
)

# ---------------------------------------------------------------------------
# 1. Hand-computed DerSimonian-Laird reference test
# ---------------------------------------------------------------------------

def test_hand_computed_dersimonian_laird_matches():
    """Verify DerSimonian-Laird tau2 and mu against explicit hand calculation."""
    # 3 studies/teams:
    # d = [1.0, 2.0, 3.0]
    # se = [0.5, 0.5, 0.5] => var = 0.25 => w = [4.0, 4.0, 4.0]
    # sum_w = 12.0
    # mu = (4*1 + 4*2 + 4*3) / 12 = 2.0
    # Q = 4*(1-2)^2 + 4*(2-2)^2 + 4*(3-2)^2 = 4*1 + 0 + 4*1 = 8.0
    # k = 3 => k - 1 = 2
    # sum_w2 = 16 + 16 + 16 = 48
    # c = sum_w - (sum_w2 / sum_w) = 12 - (48 / 12) = 12 - 4 = 8.0
    # tau2 = (Q - (k-1)) / c = (8.0 - 2) / 8.0 = 6.0 / 8.0 = 0.75
    d = np.array([1.0, 2.0, 3.0])
    se = np.array([0.5, 0.5, 0.5])

    mu, tau2 = dersimonian_laird_tau2(d, se)
    assert pytest.approx(mu, rel=1e-6) == 2.0
    assert pytest.approx(tau2, rel=1e-6) == 0.75


# ---------------------------------------------------------------------------
# 2. tau2 = 0 gives shrink_factor 0 and d_shrunk = mu
# ---------------------------------------------------------------------------

def test_tau2_zero_gives_shrink_factor_zero_and_d_shrunk_mu():
    """When tau2 is 0 (or floored at 0), shrink_factor is 0 and d_shrunk equals mu."""
    # Identical estimates => Q = 0 => Q - (k-1) < 0 => tau2 floored at 0
    df = pd.DataFrame({
        "split": ["early", "early"],
        "season": [2016, 2016],
        "team": ["KC", "BUF"],
        "n_plays": [500, 500],
        "n_lean": [200, 200],
        "lean_share": [0.4, 0.4],
        "n_follow": [140, 140],
        "n_break": [60, 60],
        "follow_rate": [0.7, 0.7],
        "epa_follow": [0.0, 0.0],
        "epa_break": [0.2, 0.2],
        "d_raw": [0.2, 0.2],
        "se_raw": [0.1, 0.1],
        "warn_low_n": [False, False],
        "label": ["secondary", "secondary"],
    })

    res = apply_eb_shrinkage(df)
    assert (res["tau2"] == 0.0).all()
    assert (res["shrink_factor"] == 0.0).all()
    assert (res["d_shrunk"] == res["mu_league"]).all()
    assert pytest.approx(res["mu_league"].iloc[0], rel=1e-4) == 0.2


# ---------------------------------------------------------------------------
# 3. Shrinkage pulls larger-SE teams closer to mu
# ---------------------------------------------------------------------------

def test_shrinkage_pulls_larger_se_closer_to_mu():
    """Teams with larger SE have lower shrink_factor and are pulled closer to mu."""
    # Two teams with same d_raw deviation from mu, but different SE
    df = pd.DataFrame({
        "split": ["early", "early", "early"],
        "season": [2016, 2016, 2016],
        "team": ["A", "B", "C"],
        "n_plays": [500, 500, 500],
        "n_lean": [200, 200, 200],
        "lean_share": [0.4, 0.4, 0.4],
        "n_follow": [140, 140, 140],
        "n_break": [60, 60, 60],
        "follow_rate": [0.7, 0.7, 0.7],
        "epa_follow": [0.0, 0.0, 0.0],
        "epa_break": [0.0, 0.5, 0.5],
        "d_raw": [0.0, 0.5, 0.5],
        "se_raw": [0.05, 0.05, 0.25],  # C has 5x larger SE than B
        "warn_low_n": [False, False, False],
        "label": ["secondary", "secondary", "secondary"],
    })

    res = apply_eb_shrinkage(df)
    mu = res["mu_league"].iloc[0]

    # C has larger SE, so its shrink factor should be strictly smaller than B's
    sf_b = res.loc[res["team"] == "B", "shrink_factor"].iloc[0]
    sf_c = res.loc[res["team"] == "C", "shrink_factor"].iloc[0]
    assert sf_c < sf_b, f"Expected sf_c < sf_b, got {sf_c} vs {sf_b}"

    # Distance to mu: |d_shrunk - mu| should be smaller for C than B
    dist_b = abs(res.loc[res["team"] == "B", "d_shrunk"].iloc[0] - mu)
    dist_c = abs(res.loc[res["team"] == "C", "d_shrunk"].iloc[0] - mu)
    assert dist_c < dist_b, f"Expected dist_c < dist_b, got {dist_c} vs {dist_b}"


# ---------------------------------------------------------------------------
# 4. Sign convention test on a frame where breaking clearly pays
# ---------------------------------------------------------------------------

def test_sign_convention_when_breaking_pays():
    """d_raw is POSITIVE when broke plays have higher EPA than followed plays."""
    n = 100
    df = pd.DataFrame({
        "game_id": [f"g_{i}" for i in range(n)],
        "season": [2020] * n,
        "posteam": ["KC"] * n,
        "phat": [0.8] * n,          # All lean pass (phat >= 0.65)
        "call_pass": [1.0] * 50 + [0.0] * 50,  # 50 followed (pass), 50 broke (run)
        "epa": [0.0] * 50 + [1.0] * 50,        # Followed EPA = 0.0, Broke EPA = 1.0
    })

    res = compute_team_scout_split(df, split_name="early")
    assert len(res) == 1
    row = res.iloc[0]
    assert row["n_follow"] == 50
    assert row["n_break"] == 50
    assert pytest.approx(row["epa_follow"], rel=1e-4) == 0.0
    assert pytest.approx(row["epa_break"], rel=1e-4) == 1.0
    assert pytest.approx(row["d_raw"], rel=1e-4) == 1.0, (
        f"Expected d_raw = +1.0, got {row['d_raw']}"
    )


# ---------------------------------------------------------------------------
# 5. Teams with tiny n are listed and flagged (warn_low_n=True)
# ---------------------------------------------------------------------------

def test_tiny_n_listed_and_flagged():
    """Team-seasons with n_break < 2 have NaN d_raw; n_break < 60 sets warn_low_n."""
    df = pd.DataFrame({
        "game_id": ["g1", "g2", "g3", "g4"],
        "season": [2021, 2021, 2021, 2021],
        "posteam": ["TINY", "TINY", "TINY", "TINY"],
        "phat": [0.8, 0.8, 0.8, 0.8],
        "call_pass": [1.0, 1.0, 1.0, 0.0],  # 3 follow, 1 break
        "epa": [0.1, 0.2, 0.1, 0.5],
    })

    res = compute_team_scout_split(df, split_name="tune")
    assert len(res) == 1
    row = res.iloc[0]
    assert row["team"] == "TINY"
    assert row["n_break"] == 1
    assert bool(row["warn_low_n"]) is True
    # Excluded from d_raw estimation because n_break < 2
    assert np.isnan(row["d_raw"])
    assert np.isnan(row["se_raw"])


# ---------------------------------------------------------------------------
# 6. Secondary stage refuses without its preconditions
# ---------------------------------------------------------------------------

def test_secondary_stage_refuses_without_env_var(monkeypatch):
    """run_secondary_pipeline raises RuntimeError when holdout env var is absent."""
    from scripts.run_all import run_secondary_pipeline

    monkeypatch.delenv(config.HOLDOUT_ENV_VAR, raising=False)
    with pytest.raises(RuntimeError, match=config.HOLDOUT_ENV_VAR):
        run_secondary_pipeline(only="teams")


def test_secondary_stage_refuses_without_primary_json(monkeypatch):
    """run_secondary_pipeline raises RuntimeError when primary.json is missing."""
    from scripts.run_all import run_secondary_pipeline

    monkeypatch.setenv(config.HOLDOUT_ENV_VAR, "1")

    results_path = Path("results/primary.json")
    existed = results_path.exists()
    if existed:
        results_path.rename(results_path.with_suffix(".json.bak"))

    try:
        with pytest.raises(RuntimeError, match="primary.json"):
            run_secondary_pipeline(only="teams")
    finally:
        if existed:
            bak = results_path.with_suffix(".json.bak")
            if bak.exists():
                bak.rename(results_path)


# ---------------------------------------------------------------------------
# 7. Output validates against schema
# ---------------------------------------------------------------------------

def test_teams_output_validates_against_schema():
    """compute_team_scout output validates against the schema in src/schema.py."""
    # Synthetic plays across 2 splits
    rng = np.random.default_rng(config.RANDOM_SEED)
    rows = []
    for team in ["KC", "BUF", "BAL"]:
        for _ in range(150):
            phat = rng.uniform(0.1, 0.9)
            call_pass = float(rng.binomial(1, 0.55))
            rows.append({
                "game_id": f"g_{rng.integers(1, 20)}",
                "season": 2018,
                "posteam": team,
                "phat": phat,
                "call_pass": call_pass,
                "epa": float(rng.normal(0, 0.5)),
            })
    df_synth = pd.DataFrame(rows)

    dfs = {"early": df_synth, "tune": df_synth}
    teams_df = compute_team_scout(dfs)

    # Validate against schema
    validate_parquet_df(teams_df, schema_name="teams", filename="teams.parquet")

    # Check non-evaluative reporting (no best/worst ranking columns)
    forbidden = ["best", "worst", "rank", "ranking", "tier", "grade"]
    for col in teams_df.columns:
        assert not any(f in col.lower() for f in forbidden), (
            f"Evaluative label found in column: {col}"
        )


# ---------------------------------------------------------------------------
# 8. Relocated franchise codes are mapped properly
# ---------------------------------------------------------------------------

def test_franchise_code_normalization():
    """Verify OAK -> LV, SD -> LAC, STL -> LA normalization."""
    teams = pd.Series(["OAK", "SD", "STL", "KC", "GB"])
    normalized = normalize_team_codes(teams)
    expected = ["LV", "LAC", "LA", "KC", "GB"]
    assert list(normalized) == expected
