"""Tests for holdout lock and cached split data access."""

from pathlib import Path

import pandas as pd
import pytest

from src import config
from src.load import find_season_file, get_split, split_counts


def _has_raw_data() -> bool:
    """Check if raw data is available in data/raw/."""
    try:
        find_season_file(2016)
        return True
    except FileNotFoundError:
        return False


def test_holdout_raises_by_default(monkeypatch: pytest.MonkeyPatch):
    """Verify get_split('holdout') raises PermissionError when lock is default."""
    monkeypatch.delenv(config.HOLDOUT_ENV_VAR, raising=False)
    with pytest.raises(PermissionError) as exc_info:
        get_split("holdout")
    assert "Holdout split is locked" in str(exc_info.value)


def test_holdout_raises_when_env_set_but_plan_not_committed(
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify holdout raises when env var is 1 but PLAN.md is not committed in git."""
    monkeypatch.setenv(config.HOLDOUT_ENV_VAR, "1")
    monkeypatch.setattr("src.load._is_plan_committed", lambda: False)

    with pytest.raises(PermissionError) as exc_info:
        get_split("holdout")
    assert "PLAN.md is not committed in git" in str(exc_info.value)


def test_holdout_works_with_both_conditions_and_logs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Verify holdout returns data when both conditions are met, and logs access."""
    if not _has_raw_data():
        pytest.skip("Raw play-by-play data not present in data/raw/")
    monkeypatch.setenv(config.HOLDOUT_ENV_VAR, "1")
    monkeypatch.setattr("src.load._is_plan_committed", lambda: True)

    fake_log = tmp_path / "HOLDOUT_ACCESS.log"
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(config, "HOLDOUT_ACCESS_LOG", fake_log)

    df_holdout = get_split("holdout")
    assert isinstance(df_holdout, pd.DataFrame)
    assert len(df_holdout) == 48469

    # Verify log entry
    assert fake_log.exists()
    log_content = fake_log.read_text(encoding="utf-8")
    assert "caller=test_holdout_works_with_both_conditions_and_logs" in log_content
    assert "commit=" in log_content


def test_split_counts_has_no_outcome_columns():
    """Verify split_counts output has no outcome columns (epa, success, etc.)."""
    if not _has_raw_data():
        pytest.skip("Raw play-by-play data not present in data/raw/")
    counts_df = split_counts()
    assert isinstance(counts_df, pd.DataFrame)

    forbidden_cols = ["epa", "success", "wp", "xpass", "call_pass", "qb_dropback"]
    for col in forbidden_cols:
        assert col not in counts_df.columns, f"Outcome column '{col}' exposed!"

    assert set(counts_df.columns) == {"split", "plays", "games"}


def test_early_and_tune_work_without_env_var(monkeypatch: pytest.MonkeyPatch):
    """Verify early and tune splits work without the holdout env var."""
    if not _has_raw_data():
        pytest.skip("Raw play-by-play data not present in data/raw/")
    monkeypatch.delenv(config.HOLDOUT_ENV_VAR, raising=False)

    df_early = get_split("early")
    assert isinstance(df_early, pd.DataFrame)
    assert len(df_early) == 142946

    df_tune = get_split("tune")
    assert isinstance(df_tune, pd.DataFrame)
    assert len(df_tune) == 50122


def test_invalid_split_name_raises_value_error():
    """Verify get_split raises ValueError on unrecognized split names."""
    with pytest.raises(ValueError) as exc_info:
        get_split("postseason")
    assert "Invalid split name" in str(exc_info.value)


def test_split_counts_exact_real_numbers():
    """Verify real data holdout and early/tune numbers in split_counts()."""
    if not _has_raw_data():
        pytest.skip("Raw play-by-play data not present in data/raw/")
    df = split_counts()
    rows = df.set_index("split").to_dict(orient="index")

    assert rows["early"]["plays"] == 142946
    assert rows["early"]["games"] == 1552

    assert rows["tune"]["plays"] == 50122
    assert rows["tune"]["games"] == 543

    assert rows["holdout"]["plays"] == 48469
    assert rows["holdout"]["games"] == 544
