"""Tests for src.load module."""

import gzip
from pathlib import Path

import pandas as pd
import pytest

from src import config
from src.load import assert_schema, find_season_file, inventory, read_season


def _create_mock_file(
    file_path: Path,
    columns: list[str],
    rows: list[list[str]] | None = None,
    is_gzip: bool = False,
) -> None:
    """Helper to write a mock CSV or CSV.GZ file."""
    if rows is None:
        rows = [["0"] * len(columns)]

    content_lines = [",".join(columns)]
    for row in rows:
        content_lines.append(",".join(row))
    content = "\n".join(content_lines) + "\n"

    if is_gzip:
        with gzip.open(file_path, "wt", encoding="utf-8") as f:
            f.write(content)
    else:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)


def _generate_372_valid_columns() -> list[str]:
    """Generate exactly 372 columns with all config.USECOLS and no 'no_play'."""
    cols = list(config.USECOLS)
    needed = 372 - len(cols)
    for i in range(needed):
        cols.append(f"extra_dummy_col_{i}")
    return cols


@pytest.fixture
def fake_raw_dir(tmp_path: Path) -> Path:
    """Fixture providing raw data directory with fake csv and csv.gz files."""
    valid_cols = _generate_372_valid_columns()

    # 2016 as csv only
    _create_mock_file(
        tmp_path / "play_by_play_2016.csv",
        columns=valid_cols,
        is_gzip=False,
    )

    # 2017 as csv.gz only
    _create_mock_file(
        tmp_path / "play_by_play_2017.csv.gz",
        columns=valid_cols,
        is_gzip=True,
    )

    # 2018 as BOTH csv and csv.gz (to test preference)
    _create_mock_file(
        tmp_path / "play_by_play_2018.csv",
        columns=valid_cols,
        is_gzip=False,
    )
    _create_mock_file(
        tmp_path / "play_by_play_2018.csv.gz",
        columns=valid_cols,
        is_gzip=True,
    )

    return tmp_path


def test_find_season_file_formats_and_preference(fake_raw_dir: Path):
    """Verify find_season_file accepts both formats and prefers .csv.gz."""
    # Only csv exists
    file_2016 = find_season_file(2016, raw_dir=fake_raw_dir)
    assert file_2016.name == "play_by_play_2016.csv"

    # Only csv.gz exists
    file_2017 = find_season_file(2017, raw_dir=fake_raw_dir)
    assert file_2017.name == "play_by_play_2017.csv.gz"

    # Both exist: MUST prefer .csv.gz
    file_2018 = find_season_file(2018, raw_dir=fake_raw_dir)
    assert file_2018.name == "play_by_play_2018.csv.gz"


def test_find_season_file_missing_season_error(fake_raw_dir: Path):
    """Verify missing season raises FileNotFoundError naming year and folder."""
    missing_year = 1999
    with pytest.raises(FileNotFoundError) as exc_info:
        find_season_file(missing_year, raw_dir=fake_raw_dir)

    error_text = str(exc_info.value)
    assert str(missing_year) in error_text
    assert str(fake_raw_dir) in error_text


def test_assert_schema_valid_file(fake_raw_dir: Path):
    """Verify assert_schema succeeds on a valid 372-column file."""
    file_path = fake_raw_dir / "play_by_play_2016.csv"
    assert_schema(file_path)

    file_gz_path = fake_raw_dir / "play_by_play_2017.csv.gz"
    assert_schema(file_gz_path)


def test_assert_schema_fires_on_missing_usecols(tmp_path: Path):
    """Verify assert_schema raises AssertionError listing missing USECOLS columns."""
    valid_cols = _generate_372_valid_columns()
    # Remove 'epa' and 'qb_dropback' and replace with dummy columns to keep count 372
    bad_cols = [c for c in valid_cols if c not in ("epa", "qb_dropback")]
    bad_cols.extend(["filler_col_a", "filler_col_b"])
    assert len(bad_cols) == 372

    bad_file = tmp_path / "play_by_play_missing_cols.csv"
    _create_mock_file(bad_file, columns=bad_cols, is_gzip=False)

    with pytest.raises(AssertionError) as exc_info:
        assert_schema(bad_file)

    error_msg = str(exc_info.value)
    assert "Missing required USECOLS" in error_msg
    assert "epa" in error_msg
    assert "qb_dropback" in error_msg


def test_assert_schema_fires_on_no_play_present(tmp_path: Path):
    """Verify assert_schema raises AssertionError when 'no_play' is present."""
    cols = _generate_372_valid_columns()
    cols[100] = "no_play"  # Inject 'no_play'
    assert len(cols) == 372

    bad_file = tmp_path / "play_by_play_with_no_play.csv"
    _create_mock_file(bad_file, columns=cols, is_gzip=False)

    with pytest.raises(AssertionError) as exc_info:
        assert_schema(bad_file)

    error_msg = str(exc_info.value)
    assert "no_play" in error_msg


def test_assert_schema_fires_on_wrong_column_count(tmp_path: Path):
    """Verify assert_schema raises AssertionError when column count != 372."""
    cols = list(config.USECOLS)  # Only 28 columns instead of 372
    bad_file = tmp_path / "play_by_play_few_cols.csv"
    _create_mock_file(bad_file, columns=cols, is_gzip=False)

    with pytest.raises(AssertionError) as exc_info:
        assert_schema(bad_file)

    error_msg = str(exc_info.value)
    assert "Expected exactly 372 columns" in error_msg


def test_read_season_safety_holdout(fake_raw_dir: Path):
    """Verify read_season strictly blocks reading holdout seasons (2024-2025)."""
    with pytest.raises(PermissionError) as exc_2024:
        read_season(2024, raw_dir=fake_raw_dir)
    assert "Holdout protection" in str(exc_2024.value)

    with pytest.raises(PermissionError) as exc_2025:
        read_season(2025, raw_dir=fake_raw_dir)
    assert "Holdout protection" in str(exc_2025.value)


def test_read_season_loads_only_usecols(fake_raw_dir: Path):
    """Verify read_season loads exactly config.USECOLS for early seasons."""
    df = read_season(2016, raw_dir=fake_raw_dir)
    assert isinstance(df, pd.DataFrame)
    assert sorted(df.columns.tolist()) == sorted(config.USECOLS)
    assert len(df.columns) == 28


def test_inventory_returns_records(fake_raw_dir: Path):
    """Verify inventory reports all seasons and detects present/missing files."""
    records = inventory(raw_dir=fake_raw_dir)
    assert len(records) == 10
    found_2016 = [r for r in records if r["season"] == 2016][0]
    assert found_2016["status"] == "FOUND"
    assert found_2016["format"] == "csv"

    found_2017 = [r for r in records if r["season"] == 2017][0]
    assert found_2017["status"] == "FOUND"
    assert found_2017["format"] == "csv.gz"

    found_uncreated = [r for r in records if r["season"] == 2020][0]
    assert found_uncreated["status"] == "MISSING"
