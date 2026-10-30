"""Results schema, validation, bundle loader, and manifest tracking for Tendency Tax.

Defines:
  1. Pydantic models for JSON artifacts:
     - primary.json
     - dev_primary.json
     - model_scores.json
     - model_diagnostics.json
     - power.json
  2. Column-level schema definitions and validation for Parquet artifacts:
     - decile_gap (dev / holdout)
     - placebo (early / tune / holdout)
     - calibration (early / tune)
     - phat_by_bucket
     - robustness
     - cells
     - teams
  3. load_results(source='real' | 'fixtures') -> ResultsBundle
  4. generate_manifest() -> dict / results/MANIFEST.json

Sign Convention:
  For delta and d_raw, POSITIVE means plays against situational expectations
  are associated with HIGHER EPA (consistent with a surprise premium / tendency tax).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------------
# Custom Error
# ---------------------------------------------------------------------------
class ResultsError(Exception):
    """Raised when results files are missing or fail validation."""

    def __init__(self, filename: str, reason: str, command: str) -> None:
        self.filename = filename
        self.reason = reason
        self.command = command
        super().__init__(
            f"Results validation failed for '{filename}': {reason}. "
            f"Please run '{command}'."
        )


# ---------------------------------------------------------------------------
# Pydantic Models for JSON Artifacts
# ---------------------------------------------------------------------------

class BaseResultModel(BaseModel):
    """Base model allowing fixture markers."""
    model_config = ConfigDict(extra="ignore")
    is_fixture: bool | None = None


# --- Placebo & Spec B / Spec A Result Models ---
class PlaceboSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")
    n_perm: int
    delta_observed: float
    pct_2_5: float
    pct_97_5: float
    outside_range: bool
    p_value: float


class RegressionSpecResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    point_b1: float
    point_b2: float
    point_b3: float
    delta: float
    delta_ci_95: list[float]
    delta_se: float
    p_value_delta: float
    delta_run: float
    delta_run_ci_95: list[float]
    delta_pass: float
    delta_pass_ci_95: list[float]
    placebo: PlaceboSummary | None = None


class SplitRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")
    n_plays: int
    n_games: int
    spec_b: RegressionSpecResult
    spec_a: RegressionSpecResult | None = None


# --- Dev Primary JSON ---
class DevPrimaryResult(BaseResultModel):
    early: SplitRecord
    tune: SplitRecord


# --- Primary JSON ---
class VerdictConditions(BaseModel):
    model_config = ConfigDict(extra="ignore")
    c1_ci_excludes_zero: bool
    c2_sign_matches_early: bool
    c3_outside_placebo: bool


class VerdictData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    state: Literal["supported", "not supported / inconclusive"]
    conditions: VerdictConditions
    failed: list[str]
    direction: Literal["positive", "negative", "zero"]
    sentence: str


class PredictiveMetric(BaseModel):
    model_config = ConfigDict(extra="ignore")
    log_loss: float
    brier_score: float


class PredictiveScores(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model_phat: PredictiveMetric
    baseline_cell: PredictiveMetric


class HoldoutRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")
    n_plays: int
    n_games: int
    spec_b: RegressionSpecResult
    spec_a_secondary: RegressionSpecResult | None = None
    predictive_scores: PredictiveScores | None = None


class PrimaryResult(BaseResultModel):
    timestamp: str
    git_commit: str
    plan_commit: str
    package_versions: dict[str, str]
    simulation_parameters: dict[str, Any]
    early: SplitRecord
    tune: SplitRecord
    holdout: HoldoutRecord
    verdict: VerdictData


# --- Model Scores JSON ---
class ModelScoreItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    split: str
    predictor: str
    log_loss: float
    brier_score: float


# --- Model Diagnostics JSON ---
class SplitDistribution(BaseModel):
    model_config = ConfigDict(extra="ignore")
    min: float
    p05: float
    p25: float
    p50: float
    p75: float
    p95: float
    max: float


class SplitDiagnostics(BaseModel):
    model_config = ConfigDict(extra="ignore")
    split: str
    n_plays: int
    weighted_within_bucket_sd: float
    within_bucket_sds: dict[str, float]
    distribution: SplitDistribution
    lean_share: float
    followed_lean_share: float


class ModelDiagnosticsResult(BaseResultModel):
    early: SplitDiagnostics
    tune: SplitDiagnostics
    summary: str


# --- Power JSON ---
class BootstrapCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")
    analytic_se: float
    bootstrap_se: float
    relative_diff: float
    within_15_pct: bool


class SimulationDeltaResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    true_delta: float
    n_sims: int
    mean_estimate: float
    bias: float
    mean_se: float
    coverage_95: float
    power: float
    bootstrap_se_check: BootstrapCheck | None = None


class CalibrationPower(BaseModel):
    model_config = ConfigDict(extra="ignore")
    c_pass_minus_run: float
    game_re_sd: float
    noise_sd: float


class PowerResult(BaseResultModel):
    calibration: CalibrationPower
    simulation_results: dict[str, SimulationDeltaResult]
    holdout_expected_se: float
    smallest_delta_80_power_formula: float
    smallest_delta_80_power_sim: float | None = None


# ---------------------------------------------------------------------------
# Parquet Column Schemas and Validators
# ---------------------------------------------------------------------------

PARQUET_SCHEMAS: dict[str, dict[str, type | str]] = {
    "decile_gap": {
        "split": "string",
        "decile": "integer",
        "phat_min": "float",
        "phat_max": "float",
        "n_pass": "integer",
        "n_run": "integer",
        "mean_epa_pass": "float",
        "mean_epa_run": "float",
        "epa_gap_pass_minus_run": "float",
        "ci_95_low": "float",
        "ci_95_high": "float",
    },
    "placebo": {
        "null_delta": "float",
    },
    "calibration": {
        "split": "string",
        "predictor": "string",
        "bin": "integer",
        "mean_pred": "float",
        "observed_pass_rate": "float",
        "n": "integer",
    },
    "phat_by_bucket": {
        "split": "string",
        "dd_bucket": "string",
        "n": "integer",
        "mean_phat": "float",
        "sd_phat": "float",
    },
    "robustness": {
        "variant_id": "string",
        "variant_label": "string",
        "split": "string",
        "subset": "string",
        "outcome": "string",
        "phat_source": "string",
        "controls": "string",
        "delta": "float",
        "ci_lo": "float",
        "ci_hi": "float",
        "se": "float",
        "p_boot": "float",
        "n_plays": "integer",
        "n_games": "integer",
        "n_reps_valid": "integer",
        "label": "string",
    },
    "cells": {
        "split": "string",
        "cell_id": "string",
        "down": "integer",
        "dist_bucket": "string",
        "zone": "string",
        "n_plays": "integer",
        "n_games": "integer",
        "n_pass": "integer",
        "n_run": "integer",
        "phat_median": "float",
        "phat_p25": "float",
        "phat_p75": "float",
        "phat_sd": "float",
        "delta": "float",
        "ci_lo": "float",
        "ci_hi": "float",
        "se": "float",
        "p_boot": "float",
        "q_bh": "float",
        "testable": "bool",
        "below_min_plays": "bool",
        "verdict": "string",
        "reason": "string",
        "label": "string",
    },
    "teams": {
        "split": "string",
        "season": "integer",
        "team": "string",
        "n_plays": "integer",
        "n_lean": "integer",
        "lean_share": "float",
        "n_follow": "integer",
        "n_break": "integer",
        "follow_rate": "float",
        "epa_follow": "float",
        "epa_break": "float",
        "d_raw": "float",
        "se_raw": "float",
        "mu_league": "float",
        "tau2": "float",
        "shrink_factor": "float",
        "d_shrunk": "float",
        "shrunk_lo": "float",
        "shrunk_hi": "float",
        "warn_low_n": "bool",
        "label": "string",
    },
}

ALLOWED_CELL_VERDICTS = {"premium", "reverse", "none", "cant_tell"}


def _is_compatible_type(dtype: Any, expected_kind: str) -> bool:
    """Check if pandas dtype matches expected logical type."""
    if expected_kind == "string":
        return (
            pd.api.types.is_string_dtype(dtype)
            or pd.api.types.is_object_dtype(dtype)
        )
    if expected_kind == "integer":
        return pd.api.types.is_integer_dtype(dtype)
    if expected_kind == "float":
        return pd.api.types.is_float_dtype(dtype)
    if expected_kind == "bool":
        return pd.api.types.is_bool_dtype(dtype)
    return False


def validate_parquet_df(
    df: pd.DataFrame, schema_name: str, filename: str = ""
) -> None:
    """Validate DataFrame against declared parquet schema.

    Raises ResultsError if columns missing, types mismatch, or constraints fail.
    """
    if schema_name not in PARQUET_SCHEMAS:
        raise ValueError(f"Unknown parquet schema '{schema_name}'")

    schema = PARQUET_SCHEMAS[schema_name]
    cmd = "make dev-run" if "dev" in filename else "make data"

    # Check missing columns
    missing = [c for c in schema if c not in df.columns]
    if missing:
        raise ResultsError(
            filename=filename or schema_name,
            reason=f"Missing required columns: {missing}",
            command=cmd,
        )

    # Check dtypes
    for col, expected_kind in schema.items():
        if not _is_compatible_type(df[col].dtype, expected_kind):
            err_msg = (
                f"Column '{col}' has invalid dtype {df[col].dtype}, "
                f"expected {expected_kind}"
            )
            raise ResultsError(
                filename=filename or schema_name,
                reason=err_msg,
                command=cmd,
            )

    # Value constraints
    if schema_name == "cells":
        invalid_verdicts = set(df["verdict"].dropna().unique()) - ALLOWED_CELL_VERDICTS
        if invalid_verdicts:
            err_msg = (
                f"Invalid cell verdict values found: {invalid_verdicts}. "
                f"Allowed: {ALLOWED_CELL_VERDICTS}"
            )
            raise ResultsError(
                filename=filename or schema_name,
                reason=err_msg,
                command="make secondary",
            )
        invalid_labels = set(df["label"].dropna().unique()) - {"secondary"}
        if invalid_labels:
            raise ResultsError(
                filename=filename or schema_name,
                reason=f"Cells label must be 'secondary', found: {invalid_labels}",
                command="make secondary",
            )

    if schema_name == "robustness":
        invalid_labels = set(df["label"].dropna().unique()) - {"secondary"}
        if invalid_labels:
            raise ResultsError(
                filename=filename or schema_name,
                reason=f"Robustness label must be 'secondary', found: {invalid_labels}",
                command="make secondary",
            )


# ---------------------------------------------------------------------------
# Results Bundle and Loader
# ---------------------------------------------------------------------------

@dataclass
class ResultsBundle:
    """Container holding all validated analysis artifacts."""
    source: str
    primary: PrimaryResult | None
    dev_primary: DevPrimaryResult | None
    model_scores: list[ModelScoreItem]
    model_diagnostics: ModelDiagnosticsResult
    power: PowerResult
    decile_gap_dev: pd.DataFrame
    decile_gap_holdout: pd.DataFrame | None
    placebo_early: pd.DataFrame
    placebo_tune: pd.DataFrame
    placebo_holdout: pd.DataFrame | None
    calibration_early: pd.DataFrame
    calibration_tune: pd.DataFrame
    phat_by_bucket: pd.DataFrame
    robustness: pd.DataFrame | None = None
    cells: pd.DataFrame | None = None
    teams: pd.DataFrame | None = None


def load_results(
    source: Literal["real", "fixtures"] = "real",
    base_dir: Path | None = None,
) -> ResultsBundle:
    """Validate and load all results into a typed ResultsBundle.

    Parameters
    ----------
    source : 'real' | 'fixtures'
        Whether to load from results/ or tests/fixtures/results/
    base_dir : Path | None
        Override base directory path.

    Returns
    -------
    ResultsBundle
    """
    if base_dir is not None:
        target_dir = Path(base_dir)
    elif source == "fixtures":
        target_dir = Path("tests/fixtures/results")
    else:
        target_dir = Path("results")

    if not target_dir.exists():
        raise ResultsError(
            filename=str(target_dir),
            reason="Directory not found",
            command="make dev-run" if source == "real" else "generate fixtures",
        )

    # Helper to load and validate JSON
    def _load_json(filename: str, model_cls: type[BaseModel], command: str) -> Any:
        fpath = target_dir / filename
        if not fpath.exists():
            raise ResultsError(filename, "File does not exist", command)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return [model_cls.model_validate(item) for item in data]
            return model_cls.model_validate(data)
        except Exception as e:
            raise ResultsError(
                filename, f"Failed schema validation: {e}", command
            ) from e

    # Helper to load and validate Parquet
    def _load_parquet(filename: str, schema_name: str, command: str) -> pd.DataFrame:
        fpath = target_dir / filename
        if not fpath.exists():
            raise ResultsError(filename, "File does not exist", command)
        try:
            df = pd.read_parquet(fpath)
            validate_parquet_df(df, schema_name=schema_name, filename=filename)
            return df
        except ResultsError:
            raise
        except Exception as e:
            raise ResultsError(filename, f"Invalid parquet file: {e}", command) from e

    # Load required JSONs
    model_scores = _load_json(
        "model_scores.json", ModelScoreItem, "make dev-run"
    )
    model_diagnostics = _load_json(
        "model_diagnostics.json", ModelDiagnosticsResult, "make diagnostics"
    )
    power = _load_json("power.json", PowerResult, "make power")

    # dev_primary vs primary
    primary = None
    dev_primary = None
    if (target_dir / "primary.json").exists():
        primary = _load_json("primary.json", PrimaryResult, "make confirm")
    if (target_dir / "dev_primary.json").exists():
        dev_primary = _load_json("dev_primary.json", DevPrimaryResult, "make dev-run")

    if primary is None and dev_primary is None:
        raise ResultsError(
            "primary.json / dev_primary.json",
            "No primary result found",
            "make dev-run",
        )

    # Load core parquets
    decile_gap_dev = _load_parquet(
        "decile_gap_dev.parquet", "decile_gap", "make dev-run"
    )
    placebo_early = _load_parquet(
        "placebo_early.parquet", "placebo", "make dev-run"
    )
    placebo_tune = _load_parquet(
        "placebo_tune.parquet", "placebo", "make dev-run"
    )
    calibration_early = _load_parquet(
        "calibration_early.parquet", "calibration", "make dev-run"
    )
    calibration_tune = _load_parquet(
        "calibration_tune.parquet", "calibration", "make dev-run"
    )
    phat_by_bucket = _load_parquet(
        "phat_by_bucket.parquet", "phat_by_bucket", "make diagnostics"
    )

    # Optional / holdout parquets
    decile_gap_holdout = None
    if (target_dir / "decile_gap_holdout.parquet").exists():
        decile_gap_holdout = _load_parquet(
            "decile_gap_holdout.parquet", "decile_gap", "make confirm"
        )

    placebo_holdout = None
    if (target_dir / "placebo_holdout.parquet").exists():
        placebo_holdout = _load_parquet(
            "placebo_holdout.parquet", "placebo", "make confirm"
        )

    # Secondary / future Phase 5 parquets if present
    robustness = None
    if (target_dir / "robustness.parquet").exists():
        robustness = _load_parquet("robustness.parquet", "robustness", "make secondary")

    cells = None
    if (target_dir / "cells.parquet").exists():
        cells = _load_parquet("cells.parquet", "cells", "make secondary")

    teams = None
    if (target_dir / "teams.parquet").exists():
        teams = _load_parquet("teams.parquet", "teams", "make secondary")

    return ResultsBundle(
        source=source,
        primary=primary,
        dev_primary=dev_primary,
        model_scores=model_scores,
        model_diagnostics=model_diagnostics,
        power=power,
        decile_gap_dev=decile_gap_dev,
        decile_gap_holdout=decile_gap_holdout,
        placebo_early=placebo_early,
        placebo_tune=placebo_tune,
        placebo_holdout=placebo_holdout,
        calibration_early=calibration_early,
        calibration_tune=calibration_tune,
        phat_by_bucket=phat_by_bucket,
        robustness=robustness,
        cells=cells,
        teams=teams,
    )


# ---------------------------------------------------------------------------
# Manifest Tracking Support
# ---------------------------------------------------------------------------

def generate_manifest(
    results_dir: Path = Path("results"),
    output_path: Path = Path("results/MANIFEST.json"),
    schema_version: str = "1.0",
) -> dict[str, Any]:
    """Inspect and generate results/MANIFEST.json for all result files.

    Records:
      - filename
      - schema_version
      - row_count (for parquet) or entry_count (for json)
      - file_size_bytes
      - sha256_hash
      - generated_at
      - git_hash
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results dir '{results_path}' does not exist.")

    # Get current git commit hash
    try:
        git_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
            .strip()
        )
    except Exception:
        git_hash = "unknown"

    manifest_entries: dict[str, Any] = {}

    for file_path in sorted(results_path.glob("*")):
        if (
            file_path.is_file()
            and file_path.name != "MANIFEST.json"
            and not file_path.name.startswith(".")
        ):
            size_bytes = file_path.stat().st_size
            mtime = datetime.datetime.fromtimestamp(
                file_path.stat().st_mtime, tz=datetime.timezone.utc
            ).isoformat()

            # Compute sha256
            hasher = hashlib.sha256()
            with open(file_path, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            file_hash = hasher.hexdigest()

            count = None
            if file_path.suffix == ".parquet":
                try:
                    df = pd.read_parquet(file_path)
                    count = len(df)
                except Exception:
                    count = None
            elif file_path.suffix == ".json":
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    count = len(data) if isinstance(data, (list, dict)) else 1
                except Exception:
                    count = None

            manifest_entries[file_path.name] = {
                "schema_version": schema_version,
                "row_count": count,
                "file_size_bytes": size_bytes,
                "sha256": file_hash,
                "generated_at": mtime,
                "git_hash": git_hash,
            }

    manifest = {
        "schema_version": schema_version,
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_hash": git_hash,
        "files": manifest_entries,
    }

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    return manifest
