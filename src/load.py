"""Data loading and schema validation module.

AGENTS.md is binding:
- Holdout seasons (2024-2025) must never have their rows/data/EPA read.
- Read headers only for 2024-2025.
- Pathlib is used throughout to avoid hardcoded OS-specific path separators.
"""

from pathlib import Path
from typing import Any

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
