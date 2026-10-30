"""Cached data loader and format helpers for Tendency Tax (D6.2).

Rules:
  - The app reads results files ONLY through src/schema.load_results.
  - No model fitting, no raw data access.
  - Environment variable TENDENCY_USE_FIXTURES=1 forces fixture data.
  - Missing or invalid files return a friendly error container naming the
    missing file and the make command to generate it.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

from src.schema import ResultsBundle, load_results


@dataclass
class LoadResult:
    """Container for loaded results or friendly error details."""

    bundle: ResultsBundle | None
    is_fixtures: bool
    error_file: str | None = None
    make_command: str | None = None
    error_message: str | None = None


FILE_MAKE_COMMANDS: dict[str, str] = {
    "MANIFEST.json": "make dev-run",
    "primary.json": "make confirm",
    "dev_primary.json": "make dev-run",
    "cells.parquet": "make secondary",
    "robustness.parquet": "make secondary",
    "teams.parquet": "make secondary",
}


@st.cache_data
def get_results_bundle() -> LoadResult:
    """Load results with caching, fallback logic, and error handling.

    Logic:
      1. If TENDENCY_USE_FIXTURES == "1", load from tests/fixtures/results.
      2. Otherwise, check if results/MANIFEST.json and results/primary.json exist.
         If so, load real results via load_results('real').
      3. If real results are missing or fail to load, fall back to fixtures.
      4. If fixtures are also missing or raise FileNotFoundError, return error.
    """
    force_fixtures = os.environ.get("TENDENCY_USE_FIXTURES") == "1"

    if not force_fixtures:
        manifest_path = Path("results/MANIFEST.json")
        primary_path = Path("results/primary.json")

        if manifest_path.exists() and primary_path.exists():
            try:
                bundle = load_results(source="real")
                return LoadResult(bundle=bundle, is_fixtures=False)
            except FileNotFoundError as fnf_err:
                # Identify missing file name from error
                missing_name = Path(str(fnf_err)).name
                cmd = FILE_MAKE_COMMANDS.get(
                    missing_name, "make dev-run / make secondary"
                )
                return LoadResult(
                    bundle=None,
                    is_fixtures=False,
                    error_file=missing_name,
                    make_command=cmd,
                    error_message=f"Required result file '{missing_name}' not found.",
                )
            except Exception:
                # If loading real results fails unexpectedly, fall back to fixtures
                pass

    # Load from fixtures
    try:
        bundle = load_results(source="fixtures")
        return LoadResult(bundle=bundle, is_fixtures=True)
    except FileNotFoundError as fnf_err:
        missing_name = Path(str(fnf_err)).name
        return LoadResult(
            bundle=None,
            is_fixtures=True,
            error_file=missing_name,
            make_command="make test",
            error_message=(
                f"Fixture file '{missing_name}' missing in tests/fixtures/results/."
            ),
        )


# ---------------------------------------------------------------------------
# Formatting Helpers
# ---------------------------------------------------------------------------

def format_epa(val: float | None, decimals: int = 3) -> str:
    """Format an EPA number with +/- sign and fixed decimals."""
    if val is None or val != val:  # NaN check
        return "—"
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.{decimals}f}"


def format_pct(val: float | None, decimals: int = 1) -> str:
    """Format a probability or share [0, 1] as a percentage string."""
    if val is None or val != val:
        return "—"
    return f"{val * 100:.{decimals}f}%"


def format_int(val: int | float | None) -> str:
    """Format an integer with thousands separator."""
    if val is None or val != val:
        return "—"
    return f"{int(val):,}"


def format_ci(lo: float | None, hi: float | None, decimals: int = 3) -> str:
    """Format a 95% confidence interval as '[lo, hi]' with signs."""
    if lo is None or hi is None or lo != lo or hi != hi:
        return "[—, —]"
    sign_lo = "+" if lo > 0 else ""
    sign_hi = "+" if hi > 0 else ""
    return f"[{sign_lo}{lo:.{decimals}f}, {sign_hi}{hi:.{decimals}f}]"


# ---------------------------------------------------------------------------
# Query Params Helpers
# ---------------------------------------------------------------------------

def get_query_param(key: str, default: Any = None) -> Any:
    """Read a parameter from Streamlit query params."""
    if hasattr(st, "query_params"):
        return st.query_params.get(key, default)
    return default


def set_query_param(key: str, value: Any) -> None:
    """Set a parameter in Streamlit query params."""
    if hasattr(st, "query_params"):
        st.query_params[key] = str(value)
