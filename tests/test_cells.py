"""Tests for the 27-cell grid analysis (Deliverable D5.3).

Covers:
  1. Planted positive effect => verdict 'premium' and small q_bh.
  2. Planted null cell => not 'premium'.
  3. Low-n cells => 'cant_tell' with below_min_plays=True when threshold set.
  4. Degenerate cells (no runs / no passes / zero variance) => testable=False.
  5. BH correction matches scipy reference.
  6. CELL_EQUIV_MARGIN=None never produces 'none' verdict.
  7. Secondary stage refuses without env var.
  8. Secondary stage refuses without primary.json.
  9. Output validates against cells schema.
 10. add_grid_cell_id assigns correct zone/dist bucket labels.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import scipy.stats

from src import config
from src.cells import (
    add_grid_cell_id,
    analyze_cell,
    apply_fdr_and_verdicts,
    compute_cells_grid,
    get_all_27_cell_defs,
)
from src.schema import validate_parquet_df

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(config.RANDOM_SEED)


def _make_synthetic_df(
    n_plays: int = 400,
    n_games: int = 20,
    pass_rate: float = 0.55,
    effect: float = 0.0,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Return a synthetic play-level DataFrame suitable for cell analysis."""
    rng = np.random.default_rng(seed)
    game_ids = [f"game_{i:03d}" for i in range(n_games)]
    game_col = rng.choice(game_ids, size=n_plays)
    call_pass = rng.binomial(1, pass_rate, size=n_plays).astype(float)
    phat = rng.uniform(0.2, 0.8, size=n_plays)
    noise = rng.normal(0, 0.5, size=n_plays)
    # Spec A: epa ~ call + phat + call*phat + constant; delta = -b3 / 2
    # Negative interaction b3 yields positive delta (predictability cost).
    epa = 0.1 * call_pass + 0.2 * phat - effect * call_pass * phat + noise
    return pd.DataFrame(
        {
            "game_id": game_col,
            "call_pass": call_pass,
            "phat": phat,
            "epa": epa,
            "down": 2,
            "ydstogo": 5,
            "yardline_100": 40,
            "dd_bucket": "2_med",
        }
    )


def _make_full_grid_df(
    n_per_cell: int = 200,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Build a DataFrame covering all 27 cells with add_grid_cell_id applied."""
    rng = np.random.default_rng(seed)
    rows = []
    for down in [1, 2, 3]:
        for ydstogo, dist in [(2, "short"), (5, "medium"), (10, "long")]:
            for yardline_100, zone in [
                (10, "red_zone"), (50, "midfield"), (85, "backed_up")
            ]:
                n = n_per_cell
                n_games = 20
                game_ids = [f"g_{down}_{dist}_{zone}_{i}" for i in range(n_games)]
                rows.append(pd.DataFrame({
                    "game_id": rng.choice(game_ids, size=n),
                    "call_pass": rng.binomial(1, 0.55, size=n).astype(float),
                    "phat": rng.uniform(0.2, 0.8, size=n),
                    "epa": rng.normal(0, 0.5, size=n),
                    "down": down,
                    "ydstogo": ydstogo,
                    "yardline_100": yardline_100,
                    "dd_bucket": "2_med",
                }))
    df = pd.concat(rows, ignore_index=True)
    return add_grid_cell_id(df)


# ---------------------------------------------------------------------------
# 1. Planted positive effect => premium
# ---------------------------------------------------------------------------

def test_planted_positive_effect_gets_premium():
    """A cell with a large planted positive interaction gets verdict 'premium'."""
    df = _make_synthetic_df(n_plays=800, n_games=40, effect=3.0, seed=1)
    cell_def = {
        "cell_id": "2_medium_midfield",
        "down": 2,
        "dist_bucket": "medium",
        "zone": "midfield",
    }
    rec = analyze_cell(df, cell_def, split_name="early", n_boot=200, n_jobs=1)
    assert rec["testable"], "Expected testable=True for large-effect cell"
    assert rec["delta"] > 0, "Expected positive delta"

    # Wrap in apply_fdr_and_verdicts to get verdict assigned
    records = [rec]
    records = apply_fdr_and_verdicts(records)
    # With a single testable cell and small p_boot, q_bh == p_boot
    assert records[0]["verdict"] == "premium", (
        f"Expected 'premium', got '{records[0]['verdict']}' (delta={rec['delta']:.4f}, "
        f"p_boot={rec['p_boot']:.4f})"
    )


# ---------------------------------------------------------------------------
# 2. Planted null => not premium
# ---------------------------------------------------------------------------

def test_planted_null_not_premium():
    """A cell with zero planted effect does not receive the 'premium' verdict."""
    df = _make_synthetic_df(n_plays=600, n_games=30, effect=0.0, seed=7)
    cell_def = {
        "cell_id": "1_short_midfield",
        "down": 1,
        "dist_bucket": "short",
        "zone": "midfield",
    }
    rec = analyze_cell(df, cell_def, split_name="early", n_boot=200, n_jobs=1)
    records = apply_fdr_and_verdicts([rec])
    # Either cant_tell (if not significant) or reverse — never premium for null
    assert records[0]["verdict"] != "premium", (
        f"Null cell should not be 'premium'; got '{records[0]['verdict']}'"
    )


# ---------------------------------------------------------------------------
# 3. Low-n cells => below_min_plays
# ---------------------------------------------------------------------------

def test_low_n_cell_below_min_plays():
    """When MIN_PLAYS_PER_CELL is set, tiny cells return below_min_plays=True."""
    df = _make_synthetic_df(n_plays=5, n_games=3)
    cell_def = {
        "cell_id": "2_medium_midfield",
        "down": 2,
        "dist_bucket": "medium",
        "zone": "midfield",
    }
    with patch.object(config, "MIN_PLAYS_PER_CELL", 50):
        rec = analyze_cell(df, cell_def, split_name="early", n_boot=50, n_jobs=1)
    assert rec["below_min_plays"] is True
    assert rec["verdict"] == "cant_tell"
    assert "fewer than" in rec["reason"]


# ---------------------------------------------------------------------------
# 4. Degenerate cells never crash and return testable=False
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario,modification", [
    ("no_passes", lambda df: df.assign(call_pass=0.0)),
    ("no_runs", lambda df: df.assign(call_pass=1.0)),
    ("zero_phat_variance", lambda df: df.assign(phat=0.5)),
    ("too_few_games", lambda df: df.assign(game_id="single_game")),
    ("empty_df", lambda df: df.iloc[:0].copy()),
])
def test_degenerate_cells_not_testable(scenario, modification):
    """Degenerate cells always return testable=False and never crash."""
    df = _make_synthetic_df(n_plays=100, n_games=10)
    df_mod = modification(df)
    cell_def = {
        "cell_id": "3_long_red_zone",
        "down": 3,
        "dist_bucket": "long",
        "zone": "red_zone",
    }
    rec = analyze_cell(df_mod, cell_def, split_name="tune", n_boot=50, n_jobs=1)
    assert rec["testable"] is False, (
        f"Expected testable=False for scenario '{scenario}'"
    )
    assert rec["verdict"] == "cant_tell"
    assert np.isnan(rec["delta"]), (
        f"Expected NaN delta for degenerate scenario '{scenario}'"
    )


# ---------------------------------------------------------------------------
# 5. BH correction matches scipy reference
# ---------------------------------------------------------------------------

def test_bh_correction_matches_scipy_reference():
    """apply_fdr_and_verdicts BH q-values match scipy.stats.false_discovery_control."""
    # Create 5 testable records with known p-values
    p_vals = [0.001, 0.01, 0.04, 0.20, 0.50]
    records = []
    for i, p in enumerate(p_vals):
        records.append({
            "split": "early",
            "cell_id": f"cell_{i}",
            "down": 1,
            "dist_bucket": "short",
            "zone": "midfield",
            "n_plays": 200,
            "n_games": 20,
            "n_pass": 100,
            "n_run": 100,
            "phat_median": 0.5,
            "phat_p25": 0.3,
            "phat_p75": 0.7,
            "phat_sd": 0.1,
            "delta": 0.1,
            "ci_lo": 0.01,
            "ci_hi": 0.2,
            "se": 0.05,
            "p_boot": p,
            "q_bh": np.nan,
            "testable": True,
            "below_min_plays": False,
            "verdict": "cant_tell",
            "reason": "",
            "label": "secondary",
        })

    records = apply_fdr_and_verdicts(records)
    expected_q = scipy.stats.false_discovery_control(
        np.array(p_vals), method="bh"
    )
    computed_q = [r["q_bh"] for r in records]
    np.testing.assert_allclose(
        computed_q, expected_q, rtol=1e-4,
        err_msg="BH q-values do not match scipy reference."
    )


# ---------------------------------------------------------------------------
# 6. CELL_EQUIV_MARGIN=None => no 'none' verdict
# ---------------------------------------------------------------------------

def test_no_none_verdict_when_margin_is_none():
    """When CELL_EQUIV_MARGIN is None, the 'none' verdict is never assigned."""
    with patch.object(config, "CELL_EQUIV_MARGIN", None):
        df_grid = _make_full_grid_df(n_per_cell=100)
        cells_df = compute_cells_grid(
            {"early": df_grid},
            n_boot=100,
            n_jobs=1,
        )
    assert "none" not in cells_df["verdict"].values, (
        "'none' verdict appeared even though CELL_EQUIV_MARGIN is None"
    )


# ---------------------------------------------------------------------------
# 7. Secondary stage refuses without env var
# ---------------------------------------------------------------------------

def test_secondary_stage_refuses_without_env_var(tmp_path, monkeypatch):
    """run_secondary_pipeline raises RuntimeError when holdout env var is absent."""
    from scripts.run_all import run_secondary_pipeline

    monkeypatch.delenv(config.HOLDOUT_ENV_VAR, raising=False)
    with pytest.raises(RuntimeError, match=config.HOLDOUT_ENV_VAR):
        run_secondary_pipeline(only="cells")


# ---------------------------------------------------------------------------
# 8. Secondary stage refuses without primary.json
# ---------------------------------------------------------------------------

def test_secondary_stage_refuses_without_primary_json(tmp_path, monkeypatch):
    """run_secondary_pipeline raises RuntimeError when primary.json is missing."""
    from scripts.run_all import run_secondary_pipeline

    monkeypatch.setenv(config.HOLDOUT_ENV_VAR, "1")

    # Ensure primary.json does not exist for this test
    results_path = Path("results/primary.json")
    existed = results_path.exists()
    if existed:
        # Temporarily rename
        results_path.rename(results_path.with_suffix(".json.bak"))

    try:
        with pytest.raises(RuntimeError, match="primary.json"):
            run_secondary_pipeline(only="cells")
    finally:
        if existed:
            bak = results_path.with_suffix(".json.bak")
            if bak.exists():
                bak.rename(results_path)


# ---------------------------------------------------------------------------
# 9. Output validates against cells schema
# ---------------------------------------------------------------------------

def test_cells_output_validates_against_schema():
    """compute_cells_grid output passes schema validation for the cells parquet."""
    df_grid = _make_full_grid_df(n_per_cell=150, seed=42)
    cells_df = compute_cells_grid(
        {"early": df_grid, "tune": df_grid},
        n_boot=100,
        n_jobs=1,
    )
    # Should not raise
    validate_parquet_df(cells_df, schema_name="cells", filename="cells.parquet")

    # Check expected columns present
    expected_cols = {
        "split", "cell_id", "down", "dist_bucket", "zone",
        "n_plays", "n_games", "n_pass", "n_run",
        "phat_median", "phat_p25", "phat_p75", "phat_sd",
        "delta", "ci_lo", "ci_hi", "se", "p_boot", "q_bh",
        "testable", "below_min_plays", "verdict", "reason", "label",
    }
    assert expected_cols.issubset(set(cells_df.columns)), (
        f"Missing columns: {expected_cols - set(cells_df.columns)}"
    )

    # All rows labeled secondary
    assert (cells_df["label"] == "secondary").all()

    # Verdicts are in the allowed set
    allowed = {"premium", "reverse", "none", "cant_tell"}
    bad = set(cells_df["verdict"].unique()) - allowed
    assert not bad, f"Unexpected verdict values: {bad}"


# ---------------------------------------------------------------------------
# 10. add_grid_cell_id assigns correct labels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("yl100,ydstogo,down,expected_zone,expected_dist", [
    (10, 2, 1, "red_zone", "short"),      # yl<=20, ydstogo<=3
    (20, 7, 2, "red_zone", "medium"),     # yl==20 boundary, ydstogo<=7
    (50, 5, 3, "midfield", "medium"),     # midfield
    (80, 10, 1, "backed_up", "long"),     # yl==80 boundary, ydstogo>=8
    (95, 15, 2, "backed_up", "long"),     # deep backed up
    (21, 3, 3, "midfield", "short"),      # just past red_zone boundary
    (79, 8, 1, "midfield", "long"),       # just before backed_up boundary
])
def test_add_grid_cell_id_labels(yl100, ydstogo, down, expected_zone, expected_dist):
    """add_grid_cell_id assigns the correct zone and distance bucket."""
    df = pd.DataFrame(
        {"down": [down], "ydstogo": [ydstogo], "yardline_100": [yl100]}
    )
    result = add_grid_cell_id(df)
    expected_id = f"{down}_{expected_dist}_{expected_zone}"
    assert result["cell_id"].iloc[0] == expected_id, (
        f"Expected '{expected_id}', got '{result['cell_id'].iloc[0]}'"
    )


# ---------------------------------------------------------------------------
# 11. 27 cell definitions: correct count and no duplicates
# ---------------------------------------------------------------------------

def test_27_cell_defs_count_and_uniqueness():
    """get_all_27_cell_defs returns exactly 27 unique cell IDs."""
    defs = get_all_27_cell_defs()
    assert len(defs) == 27, f"Expected 27 cell defs, got {len(defs)}"
    ids = [d["cell_id"] for d in defs]
    assert len(set(ids)) == 27, "Duplicate cell IDs found"
    # Verify structure
    for d in defs:
        assert set(d.keys()) == {"cell_id", "down", "dist_bucket", "zone"}
        assert d["down"] in {1, 2, 3}
        assert d["dist_bucket"] in {"short", "medium", "long"}
        assert d["zone"] in {"red_zone", "midfield", "backed_up"}
