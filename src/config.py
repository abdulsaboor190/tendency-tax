"""Configuration constants for Tendency Tax.

Constants only, no business logic.
"""

from pathlib import Path
from typing import Any

# Paths
RAW_DATA_DIR: Path = Path("data/raw")
CACHE_DIR: Path = Path("data/cache")
RESULTS_DIR: Path = Path("results")
MODELS_DIR: Path = RESULTS_DIR / "models"
HOLDOUT_ACCESS_LOG: Path = Path("results/HOLDOUT_ACCESS.log")
HOLDOUT_ENV_VAR: str = "TENDENCY_ALLOW_HOLDOUT"
FROZEN_MODEL_PATH: Path = MODELS_DIR / "phat_model_2016_2023.joblib"
FROZEN_MODEL_META_PATH: Path = MODELS_DIR / "phat_model_2016_2023.json"
MODEL_SCORES_PATH: Path = RESULTS_DIR / "model_scores.json"
CALIBRATION_EARLY_PATH: Path = RESULTS_DIR / "calibration_early.parquet"
CALIBRATION_TUNE_PATH: Path = RESULTS_DIR / "calibration_tune.parquet"
MODEL_DIAGNOSTICS_PATH: Path = RESULTS_DIR / "model_diagnostics.json"
PHAT_BY_BUCKET_PATH: Path = RESULTS_DIR / "phat_by_bucket.parquet"

# Columns to load across project
USECOLS: list[str] = [
    "game_id",
    "season",
    "season_type",
    "week",
    "home_team",
    "posteam",
    "defteam",
    "play_type",
    "down",
    "ydstogo",
    "yardline_100",
    "qtr",
    "game_seconds_remaining",
    "half_seconds_remaining",
    "score_differential",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "spread_line",
    "qb_dropback",
    "qb_kneel",
    "qb_spike",
    "epa",
    "success",
    "wp",
    "xpass",
    "shotgun",
    "no_huddle",
    "penalty",
]

# Random Seed
RANDOM_SEED: int = 42

# Seasons & Splits
SEASONS: list[int] = list(range(2016, 2026))  # 2016 through 2025 inclusive

SPLITS: dict[str, list[int]] = {
    "early": [2016, 2017, 2018, 2019, 2020, 2021],
    "tune": [2022, 2023],
    "holdout": [2024, 2025],
}

SEASON_TO_SPLIT: dict[int, str] = {
    season: split for split, seasons in SPLITS.items() for season in seasons
}

# Label
LABEL: str = "qb_dropback"  # 1 = pass / dropback, 0 = designed run

# Features (Pre-snap situation only, exactly 10 features, excludes shotgun/no_huddle)
FEATURES: list[str] = [
    "down",
    "ydstogo",
    "yardline_100",
    "qtr",
    "game_seconds_remaining",
    "half_seconds_remaining",
    "score_differential",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "posteam_spread",
]

# Filters
SEASON_TYPE: str = "REG"  # Regular season only
VALID_PLAY_TYPES: tuple[str, ...] = ("pass", "run")
VALID_DOWNS: tuple[int, ...] = (1, 2, 3)
WP_MIN: float = 0.05
WP_MAX: float = 0.95
EXCLUDE_KNEELS: bool = True
EXCLUDE_SPIKES: bool = True
EXCLUDE_TWO_MINUTE_DRILL_SECONDS: int = 120
EXCLUDE_TWO_MINUTE_QUARTERS: tuple[int, ...] = (2, 4)

# Distance Buckets
# short: 1-3 yards, medium: 4-7 yards, long: 8+ yards
DISTANCE_BUCKETS: dict[str, tuple[int, int | None]] = {
    "short": (1, 3),
    "medium": (4, 7),
    "long": (8, None),
}

# Lean Thresholds & Simulation Parameters
LEAN_PASS_THRESHOLD: float = 0.65  # phat >= 0.65 -> lean pass
LEAN_RUN_THRESHOLD: float = 0.35   # phat <= 0.35 -> lean run
LEAN_HIGH: float = 0.65            # Alias for LEAN_PASS_THRESHOLD
LEAN_LOW: float = 0.35             # Alias for LEAN_RUN_THRESHOLD
BOOTSTRAP_REPS: int = 1000
PLACEBO_REPS: int = 500
ROBUSTNESS_REPS: int = 500

# Field-zone cut points in yardline_100 (Human Decision 1: [20, 80])
# 3 zones: red_zone (<=20), midfield (21-79), backed_up (>=80)
ZONE_CUT_POINTS: tuple[int, int] | None = (20, 80)
ZONE_NAMES: tuple[str, ...] = ("red_zone", "midfield", "backed_up")

# TODO: Human Decision (2) - Minimum plays per cell
MIN_PLAYS_PER_CELL: Any = None

# TODO: Human Decision (3) - Equivalence margin for cell delta (EPA/play)
CELL_EQUIV_MARGIN: float | None = None

# Model Hyperparameters (selected via log loss on tune split)
MODEL_PARAMS: dict[str, Any] = {
    "max_depth": 5,
    "learning_rate": 0.1,
    "l2_regularization": 1.0,
    "max_iter": 100,
}

# Forbidden columns that must never be used as model features
FORBIDDEN_FEATURES: tuple[str, ...] = (
    "epa",
    "xpass",
    "success",
    "wp",
    "shotgun",
    "no_huddle",
)
