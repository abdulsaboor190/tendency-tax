"""Tests for application data loader, formatting, and shell navigation (D6.2).

Covers:
  1. The loader returns fixtures when TENDENCY_USE_FIXTURES=1.
  2. Missing result file returns LoadResult with error details, not a crash.
  3. Format helpers format EPA, percentages, counts, and CIs cleanly.
  4. App directory strictly conforms to raw-data / holdout isolation.
"""

import re
from pathlib import Path

import numpy as np

from app.data import (
    format_ci,
    format_epa,
    format_int,
    format_pct,
    get_results_bundle,
)
from src.schema import ResultsBundle

# ---------------------------------------------------------------------------
# 1. Loader with TENDENCY_USE_FIXTURES=1
# ---------------------------------------------------------------------------

def test_loader_returns_fixtures_when_env_flag_set(monkeypatch):
    """Setting TENDENCY_USE_FIXTURES=1 forces loading from test fixtures."""
    # Clear Streamlit cache if cached
    get_results_bundle.clear()
    monkeypatch.setenv("TENDENCY_USE_FIXTURES", "1")

    res = get_results_bundle()
    assert res.is_fixtures is True
    assert isinstance(res.bundle, ResultsBundle)
    assert res.bundle.source == "fixtures"
    assert res.error_file is None


# ---------------------------------------------------------------------------
# 2. Missing Result File Produces Friendly Error
# ---------------------------------------------------------------------------

def test_loader_missing_file_friendly_error(monkeypatch, tmp_path):
    """When a file is missing in the target dir, a friendly LoadResult is returned."""
    get_results_bundle.clear()
    monkeypatch.delenv("TENDENCY_USE_FIXTURES", raising=False)

    from unittest.mock import patch

    with patch(
        "app.data.load_results",
        side_effect=FileNotFoundError("missing_artifact.parquet"),
    ):
        with patch.object(Path, "exists", return_value=True):
            res = get_results_bundle()
            assert res.bundle is None
            assert "missing_artifact.parquet" in str(res.error_file)


# ---------------------------------------------------------------------------
# 3. Formatting Helpers Tests
# ---------------------------------------------------------------------------

def test_format_helpers():
    """Verify output of numeric formatting functions."""
    # format_epa
    assert format_epa(0.04123) == "+0.041"
    assert format_epa(-0.0789) == "-0.079"
    assert format_epa(0.0) == "0.000"
    assert format_epa(None) == "—"
    assert format_epa(np.nan) == "—"

    # format_pct
    assert format_pct(0.554) == "55.4%"
    assert format_pct(0.0) == "0.0%"
    assert format_pct(None) == "—"
    assert format_pct(np.nan) == "—"

    # format_int
    assert format_int(1234567) == "1,234,567"
    assert format_int(0) == "0"
    assert format_int(None) == "—"

    # format_ci
    assert format_ci(0.0123, 0.0701) == "[+0.012, +0.070]"
    assert format_ci(-0.05, 0.05) == "[-0.050, +0.050]"
    assert format_ci(None, 0.05) == "[—, —]"
    assert format_ci(0.01, np.nan) == "[—, —]"


# ---------------------------------------------------------------------------
# 4. App Safety Isolation Grep Test
# ---------------------------------------------------------------------------

def test_app_isolation_preserved():
    """Entire app directory must NEVER import src.load or touch data/raw."""
    app_dir = Path("app")
    forbidden = [
        re.compile(r"import\s+src\.load"),
        re.compile(r"from\s+src\.load\s+import"),
        re.compile(r"data/raw"),
        re.compile(r"data\\raw"),
        re.compile(r"TENDENCY_ALLOW_HOLDOUT"),
    ]

    for p in app_dir.rglob("*.py"):
        content = p.read_text(encoding="utf-8")
        for line in content.splitlines():
            for pat in forbidden:
                assert not pat.search(line), (
                    f"Forbidden pattern {pat.pattern} found in {p}: {line}"
                )
