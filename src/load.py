"""Data loading and schema validation module.

AGENTS.md is binding:
- Holdout seasons (2024-2025) must never have their rows/data/EPA read.
- Read headers only for 2024-2025.
- Pathlib is used throughout to avoid hardcoded OS-specific path separators.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import config


def find_season_file(year: int, raw_dir: Path | str | None = None) -> Path:
    """Locate the play-by-play file for a given season in raw_dir.

    Accepts either .csv.gz or .csv. If both exist, prefers .csv.gz.
    Raises FileNotFoundError if neither exists.
    """
    directory = Path(raw_dir) if raw_dir is not None else config.RAW_DATA_DIR

    gz_file = directory / f"play_by_play_{year}.csv.gz"
    csv_file = directory / f"play_by_play_{year}.csv"

    if gz_file.exists():
        return gz_file
    elif csv_file.exists():
        return csv_file
    else:
        raise FileNotFoundError(
            f"No play-by-play data file found for season {year} in folder '{directory}'"
        )


def inventory(raw_dir: Path | str | None = None) -> list[dict[str, Any]]:
    """List all ten seasons with filename found, size in MB, and format.

    Prints a formatted table. Reports any missing season and checks
    if any file for season 2026 is present (without deleting/moving).
    """
    directory = Path(raw_dir) if raw_dir is not None else config.RAW_DATA_DIR
    records: list[dict[str, Any]] = []

    print(f"\nData Inventory for {directory}:")
    header_line = (
        f"{'Season':<8} | {'Filename':<28} | {'Size (MB)':<10} | "
        f"{'Format':<8} | {'Status'}"
    )
    separator = "-" * len(header_line)
    print(separator)
    print(header_line)
    print(separator)

    missing_seasons: list[int] = []

    for year in config.SEASONS:
        try:
            file_path = find_season_file(year, raw_dir=directory)
            size_mb = file_path.stat().st_size / (1024 * 1024)
            fmt = "csv.gz" if file_path.name.endswith(".csv.gz") else "csv"
            status = "FOUND"
            record = {
                "season": year,
                "filename": file_path.name,
                "path": file_path,
                "size_mb": round(size_mb, 2),
                "format": fmt,
                "status": status,
            }
        except FileNotFoundError:
            missing_seasons.append(year)
            record = {
                "season": year,
                "filename": None,
                "path": None,
                "size_mb": 0.0,
                "format": None,
                "status": "MISSING",
            }

        records.append(record)
        filename_str = record["filename"] or "<MISSING>"
        format_str = record["format"] or "N/A"
        size_str = f"{record['size_mb']:.2f}" if record["filename"] else "0.00"
        row_str = (
            f"{year:<8} | {filename_str:<28} | {size_str:<10} | "
            f"{format_str:<8} | {record['status']}"
        )
        print(row_str)

    print(separator)

    if missing_seasons:
        print(f"[WARNING] Missing season files: {missing_seasons}")

    # Check for 2026 files without deleting or moving anything
    future_2026_gz = directory / "play_by_play_2026.csv.gz"
    future_2026_csv = directory / "play_by_play_2026.csv"
    if future_2026_gz.exists() or future_2026_csv.exists():
        found_2026 = future_2026_gz if future_2026_gz.exists() else future_2026_csv
        print(f"[NOTE] Unexpected season 2026 file detected: {found_2026.name}")

    return records


def read_season(year: int, raw_dir: Path | str | None = None) -> pd.DataFrame:
    """Read a season's play-by-play data loading only config.USECOLS columns.

    SAFETY / HOLDOUT PROTECTION:
    Rows for seasons in config.SPLITS['holdout'] (2024-2025) must never be read.
    Raises PermissionError if a holdout season is requested.
    """
    holdout_seasons = set(config.SPLITS.get("holdout", [2024, 2025]))
    if year in holdout_seasons:
        raise PermissionError(
            f"Holdout protection: reading data rows from holdout season {year} "
            "is strictly forbidden by AGENTS.md rule 1."
        )

    file_path = find_season_file(year, raw_dir=raw_dir)
    return pd.read_csv(file_path, usecols=config.USECOLS)


def prepare(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Filter raw play-by-play dataframe and construct labels, buckets, and splits.

    Filters (in exact specified order):
    - season_type == REG
    - play_type in {pass, run}
    - down in 1..3
    - epa, posteam, xpass non-null
    - qtr in 1..4
    - wp between 0.05 and 0.95
    - not a kneel (qb_kneel == 0) and not a spike (qb_spike == 0)
    - exclude plays in last 2 mins of qtr 2 & 4 (half_seconds_remaining <= 120)

    Constructs:
    - call_pass: int(qb_dropback)
    - posteam_spread: spread_line if posteam == home_team else -spread_line
    - dist_bucket: short (1-3), medium (4-7), long (8+)
    - dd_bucket: {down}_{dist_bucket} (9 buckets)
    - zone: red_zone (<=20), midfield (21-79), backed_up (>=80) using ZONE_CUT_POINTS
    - cell_id: {dd_bucket}_{zone} (27 cells)
    - split: early / tune / holdout
    """
    if config.ZONE_CUT_POINTS is None:
        raise ValueError(
            "config.ZONE_CUT_POINTS is None. Zone cut points must be configured."
        )

    # 1. Filters in order
    two_min_qtrs = config.EXCLUDE_TWO_MINUTE_QUARTERS
    two_min_secs = config.EXCLUDE_TWO_MINUTE_DRILL_SECONDS

    mask = (
        (raw_df["season_type"] == config.SEASON_TYPE)
        & (raw_df["play_type"].isin(config.VALID_PLAY_TYPES))
        & (raw_df["down"].isin(config.VALID_DOWNS))
        & (raw_df["epa"].notna())
        & (raw_df["posteam"].notna())
        & (raw_df["xpass"].notna())
        & (raw_df["qtr"].isin([1, 2, 3, 4]))
        & (raw_df["wp"] >= config.WP_MIN)
        & (raw_df["wp"] <= config.WP_MAX)
        & (raw_df["qb_kneel"] == 0)
        & (raw_df["qb_spike"] == 0)
        & ~(
            raw_df["qtr"].isin(two_min_qtrs)
            & (raw_df["half_seconds_remaining"] <= two_min_secs)
        )
    )
    df = raw_df.loc[mask].copy()

    # 2. Columns to build
    df["call_pass"] = df["qb_dropback"].astype(int)
    df["posteam_spread"] = np.where(
        df["posteam"] == df["home_team"],
        df["spread_line"],
        -df["spread_line"],
    )

    # dist_bucket: short (ydstogo 1-3), medium (4-7), long (8+)
    dist_conditions = [df["ydstogo"] <= 3, df["ydstogo"] <= 7]
    dist_choices = ["short", "medium"]
    df["dist_bucket"] = np.select(dist_conditions, dist_choices, default="long")

    # dd_bucket: 9 buckets, down (1,2,3) x dist_bucket
    df["dd_bucket"] = df["down"].astype(int).astype(str) + "_" + df["dist_bucket"]

    # zone: red_zone, midfield, backed_up
    z_low, z_high = config.ZONE_CUT_POINTS
    zone_conditions = [df["yardline_100"] <= z_low, df["yardline_100"] < z_high]
    zone_choices = ["red_zone", "midfield"]
    df["zone"] = np.select(zone_conditions, zone_choices, default="backed_up")

    # cell_id: 27 cells
    df["cell_id"] = df["dd_bucket"] + "_" + df["zone"]

    # split: early / tune / holdout
    df["split"] = df["season"].map(config.SEASON_TO_SPLIT)

    return df.reset_index(drop=True)


def assert_schema(path: Path | str) -> None:
    """Read only the header of a file and assert schema properties:

    - The file has exactly 372 columns
    - Every column in config.USECOLS is present
    - The column 'no_play' is NOT present

    Raises AssertionError listing missing or unexpected column names.
    Reads header only (nrows=0), preserving holdout protection for 2024-2025.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found for schema check: {file_path}")

    # Read header only
    header_df = pd.read_csv(file_path, nrows=0)
    columns = set(header_df.columns)
    col_count = len(header_df.columns)

    errors: list[str] = []

    if col_count != 372:
        errors.append(f"Expected exactly 372 columns, found {col_count}")

    missing_usecols = [col for col in config.USECOLS if col not in columns]
    if missing_usecols:
        errors.append(f"Missing required USECOLS: {sorted(missing_usecols)}")

    if "no_play" in columns:
        errors.append("Unexpected column 'no_play' is present in file")

    if errors:
        error_msg = f"Schema assertion failed for '{file_path.name}':\n" + "\n".join(
            f"  - {err}" for err in errors
        )
        raise AssertionError(error_msg)


def verify_early_tune_counts(raw_dir: Path | str | None = None) -> dict[str, Any]:
    """Verify play and game counts for early (2016-2021) and tune (2022-2023) splits.

    Strictly skips holdout seasons (2024-2025).
    """
    directory = Path(raw_dir) if raw_dir is not None else config.RAW_DATA_DIR
    early_seasons = config.SPLITS["early"]
    tune_seasons = config.SPLITS["tune"]

    season_counts: dict[int, int] = {}
    early_plays = 0
    early_games: set[str] = set()
    tune_plays = 0
    tune_games: set[str] = set()

    print("\nVerifying EARLY and TUNE split counts...")
    print(f"{'Season':<8} | {'Plays':<10} | {'Split'}")
    print("-" * 32)

    for year in early_seasons + tune_seasons:
        raw_df = read_season(year, raw_dir=directory)
        prep_df = prepare(raw_df)
        n_plays = len(prep_df)
        season_counts[year] = n_plays
        split_name = config.SEASON_TO_SPLIT[year]
        print(f"{year:<8} | {n_plays:<10} | {split_name}")

        if split_name == "early":
            early_plays += n_plays
            early_games.update(prep_df["game_id"].unique())
        elif split_name == "tune":
            tune_plays += n_plays
            tune_games.update(prep_df["game_id"].unique())

    print("-" * 32)
    print(f"Early (2016-2021): {early_plays:,} plays across {len(early_games):,} games")
    print(f"Tune  (2022-2023): {tune_plays:,} plays across {len(tune_games):,} games")

    return {
        "season_counts": season_counts,
        "early_plays": early_plays,
        "early_games": len(early_games),
        "tune_plays": tune_plays,
        "tune_games": len(tune_games),
    }


def validate_all(raw_dir: Path | str | None = None) -> None:
    """Run inventory and assert_schema across all seasons."""
    inv = inventory(raw_dir=raw_dir)
    print("\nValidating schema for all seasons (header check only)...")
    for item in inv:
        file_path = item["path"]
        if file_path is None:
            raise FileNotFoundError(
                f"Cannot validate schema: missing file for season {item['season']}"
            )
        assert_schema(file_path)
        msg = f"  [PASS] {file_path.name} ({item['format']}, 372 cols, USECOLS ok)"
        print(msg)

    print("\nAll files successfully validated!")


if __name__ == "__main__":
    validate_all()
