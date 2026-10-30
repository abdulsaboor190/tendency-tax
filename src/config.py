"""Configuration constants for Tendency Tax.

Constants only, no business logic.
"""

from typing import Any

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
BOOTSTRAP_REPS: int = 1000
PLACEBO_REPS: int = 500

# Human Decisions (Blanks to be filled by human)
# TODO: Human Decision (1) - Field-zone cut points in yardline_100
ZONE_CUT_POINTS: Any = None

# TODO: Human Decision (2) - Minimum plays per cell
MIN_PLAYS_PER_CELL: Any = None
