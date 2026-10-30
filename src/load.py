"""Data loading and schema validation module.

AGENTS.md is binding:
- Holdout seasons (2024-2025) must never have their rows/data/EPA read.
- Read headers only for 2024-2025.
- Pathlib is used throughout to avoid hardcoded OS-specific path separators.
"""

import hashlib
import inspect
import json
import os
import subprocess
from datetime import datetime, timezone
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


def _read_season_raw(year: int, raw_dir: Path | str | None = None) -> pd.DataFrame:
    """Read a raw season file using USECOLS.

    Used by the caching pipeline.
    """
    file_path = find_season_file(year, raw_dir=raw_dir)
    return pd.read_csv(file_path, usecols=config.USECOLS)


def read_season(year: int, raw_dir: Path | str | None = None) -> pd.DataFrame:
    """Read a season's play-by-play data loading only config.USECOLS columns.

    SAFETY / HOLDOUT PROTECTION:
    Rows for seasons in config.SPLITS['holdout'] (2024-2025) must never be read.
    Raises PermissionError if a holdout season is requested directly.
    """
    holdout_seasons = set(config.SPLITS.get("holdout", [2024, 2025]))
    if year in holdout_seasons:
        raise PermissionError(
            f"Holdout protection: reading data rows from holdout season {year} "
            "is strictly forbidden by AGENTS.md rule 1."
        )

    return _read_season_raw(year, raw_dir=raw_dir)


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


def _is_plan_committed() -> bool:
    """Check via git log whether PLAN.md has been committed."""
    try:
        proc = subprocess.run(
            ["git", "log", "-n", "1", "--", "PLAN.md"],
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.returncode == 0 and "commit " in proc.stdout
    except Exception:
        return False


def _get_current_git_commit() -> str:
    """Get the current git commit hash."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _log_holdout_access(caller_name: str) -> None:
    """Append entry to results/HOLDOUT_ACCESS.log with caller and commit."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    commit_hash = _get_current_git_commit()
    log_line = f"{timestamp} | caller={caller_name} | commit={commit_hash}\n"
    with open(config.HOLDOUT_ACCESS_LOG, "a", encoding="utf-8") as f:
        f.write(log_line)


def _compute_split_hash(
    split_name: str,
    raw_dir: Path | str | None = None,
) -> str:
    """Compute combined SHA-256 hash of raw files for the split and src/config.py."""
    directory = Path(raw_dir) if raw_dir is not None else config.RAW_DATA_DIR
    hasher = hashlib.sha256()

    seasons = config.SPLITS[split_name]
    for year in sorted(seasons):
        file_path = find_season_file(year, raw_dir=directory)
        hasher.update(file_path.name.encode("utf-8"))
        with open(file_path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                hasher.update(chunk)

    config_file = Path("src/config.py")
    if config_file.exists():
        hasher.update(config_file.read_bytes())

    return hasher.hexdigest()


def _ensure_split_cached(
    split_name: str,
    raw_dir: Path | str | None = None,
    cache_dir: Path | str | None = None,
) -> Path:
    """Ensure the parquet cache for split_name is built and up-to-date."""
    directory = Path(raw_dir) if raw_dir is not None else config.RAW_DATA_DIR
    target_cache_dir = Path(cache_dir) if cache_dir is not None else config.CACHE_DIR
    target_cache_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = target_cache_dir / f"{split_name}.parquet"
    meta_path = target_cache_dir / f"{split_name}.meta.json"

    current_hash = _compute_split_hash(split_name, raw_dir=directory)

    if parquet_path.exists() and meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("hash") == current_hash:
                return parquet_path
        except Exception:
            pass

    # Rebuild cache for this split
    seasons = config.SPLITS[split_name]
    dfs: list[pd.DataFrame] = []
    for year in seasons:
        raw_df = _read_season_raw(year, raw_dir=directory)
        prep_df = prepare(raw_df)
        dfs.append(prep_df)

    combined = pd.concat(dfs, ignore_index=True)
    combined.to_parquet(parquet_path, index=False, engine="pyarrow")

    meta_content = {
        "split": split_name,
        "hash": current_hash,
        "plays": len(combined),
        "games": int(combined["game_id"].nunique()),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_content, f, indent=2)

    return parquet_path


def get_split(
    name: str,
    raw_dir: Path | str | None = None,
    cache_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Retrieve prepared data for a specific split.

    - 'early' and 'tune': return the prepared rows for that split.
    - 'holdout': raise a clear error UNLESS BOTH:
      (a) environment variable TENDENCY_ALLOW_HOLDOUT equals '1'
      (b) `git log -- PLAN.md` shows PLAN.md has been committed
      When allowed, returns holdout rows AND appends one line to
      results/HOLDOUT_ACCESS.log with timestamp, caller name, and commit hash.
    - Any other name: raises ValueError.
    """
    if name not in ("early", "tune", "holdout"):
        raise ValueError(
            f"Invalid split name '{name}'. Must be one of: 'early', 'tune', 'holdout'"
        )

    if name == "holdout":
        env_allowed = os.environ.get(config.HOLDOUT_ENV_VAR) == "1"
        plan_committed = _is_plan_committed()

        if not (env_allowed and plan_committed):
            reasons: list[str] = []
            if not env_allowed:
                reasons.append(f"{config.HOLDOUT_ENV_VAR} != '1'")
            if not plan_committed:
                reasons.append("PLAN.md is not committed in git")
            raise PermissionError(
                f"Holdout split is locked. Requirements not met: {', '.join(reasons)}"
            )

        caller_name = (
            inspect.stack()[1].function if len(inspect.stack()) > 1 else "unknown"
        )
        _log_holdout_access(caller_name=caller_name)

    parquet_path = _ensure_split_cached(name, raw_dir=raw_dir, cache_dir=cache_dir)
    return pd.read_parquet(parquet_path)


def split_counts(
    raw_dir: Path | str | None = None,
    cache_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Return only the number of plays and games per split.

    Does not return or expose epa, success, or any outcome column.
    Safe to call without unlocking holdout.
    """
    rows: list[dict[str, Any]] = []

    print("\nSplit Counts Summary:")
    header = f"{'Split':<10} | {'Plays':<10} | {'Games':<8}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    holdout_per_season: dict[int, dict[str, int]] = {}

    for split_name in ("early", "tune", "holdout"):
        parquet_path = _ensure_split_cached(
            split_name, raw_dir=raw_dir, cache_dir=cache_dir
        )
        meta_df = pd.read_parquet(parquet_path, columns=["game_id", "season"])
        n_plays = len(meta_df)
        n_games = int(meta_df["game_id"].nunique())
        rows.append({"split": split_name, "plays": n_plays, "games": n_games})
        print(f"{split_name:<10} | {n_plays:<10,d} | {n_games:<8,d}")

        if split_name == "holdout":
            for y in sorted(meta_df["season"].unique()):
                sub = meta_df[meta_df["season"] == y]
                holdout_per_season[y] = {
                    "plays": len(sub),
                    "games": int(sub["game_id"].nunique()),
                }

    print("-" * len(header))

    if holdout_per_season:
        print("Holdout per-season breakdown:")
        for y, counts in holdout_per_season.items():
            print(f"  {y}: {counts['plays']:,d} plays, {counts['games']} games")

    return pd.DataFrame(rows)


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
