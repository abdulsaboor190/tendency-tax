"""Tests for Exploratory Checks (Deliverable D7.5).

Covers:
  1. Sign-agreement helper (compute_down_sign_agreement) on hand-made DataFrames.
  2. Success-rate variant exclusion from the primary EPA interval plot.
  3. Page renders cleanly on fixture data (TENDENCY_USE_FIXTURES=1).
  4. Page renders cleanly on real results bundle.
  5. Absence of banned causal/absolute words in exploratory.py and test_panel.py.
  6. Noisy season thresholding logic.
"""

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.data import get_results_bundle, load_results
from app.views.exploratory import (
    compute_down_sign_agreement,
    filter_by_down_rows,
    filter_by_season_rows,
    filter_robustness_variants,
)


def _make_st_mock():
    """Return a MagicMock capturing streamlit calls."""
    m = MagicMock()
    exp = MagicMock()
    exp.__enter__ = MagicMock(return_value=exp)
    exp.__exit__ = MagicMock(return_value=False)
    m.expander.return_value = exp

    col = MagicMock()
    col.__enter__ = MagicMock(return_value=col)
    col.__exit__ = MagicMock(return_value=False)
    m.columns.return_value = (col, col, col)

    return m


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear st.cache_data between test invocations."""
    get_results_bundle.clear()
    yield
    get_results_bundle.clear()


# ---------------------------------------------------------------------------
# 1. Sign-agreement helper tests
# ---------------------------------------------------------------------------

class TestComputeDownSignAgreement:
    """Verify sign-agreement and CI zero-inclusion calculations."""

    def test_all_positive_none_include_zero(self):
        df = pd.DataFrame([
            {"split": "early", "delta": 0.10, "ci_lo": 0.02, "ci_hi": 0.18},
            {"split": "tune", "delta": 0.12, "ci_lo": 0.01, "ci_hi": 0.20},
            {"split": "holdout", "delta": 0.15, "ci_lo": 0.05, "ci_hi": 0.25},
        ])
        res = compute_down_sign_agreement(df)
        assert res["sign_agrees"] is True
        assert res["sign_direction"] == "positive"
        assert res["all_include_zero"] is False
        assert res["n_include_zero"] == 0
        assert "Sign agrees across all three splits (positive)" in res["line"]
        assert "no 95% confidence interval includes zero" in res["line"]

    def test_all_negative_all_include_zero(self):
        df = pd.DataFrame([
            {"split": "early", "delta": -0.05, "ci_lo": -0.15, "ci_hi": 0.05},
            {"split": "tune", "delta": -0.03, "ci_lo": -0.10, "ci_hi": 0.04},
            {"split": "holdout", "delta": -0.08, "ci_lo": -0.20, "ci_hi": 0.02},
        ])
        res = compute_down_sign_agreement(df)
        assert res["sign_agrees"] is True
        assert res["sign_direction"] == "negative"
        assert res["all_include_zero"] is True
        assert res["n_include_zero"] == 3
        assert "Sign agrees across all three splits (negative)" in res["line"]
        assert "every 95% confidence interval includes zero" in res["line"]

    def test_mixed_signs_some_include_zero(self):
        df = pd.DataFrame([
            {"split": "early", "delta": 0.05, "ci_lo": 0.01, "ci_hi": 0.10},
            {"split": "tune", "delta": -0.02, "ci_lo": -0.08, "ci_hi": 0.04},
            {"split": "holdout", "delta": 0.04, "ci_lo": -0.02, "ci_hi": 0.10},
        ])
        res = compute_down_sign_agreement(df)
        assert res["sign_agrees"] is False
        assert res["sign_direction"] == "mixed"
        assert res["all_include_zero"] is False
        assert res["n_include_zero"] == 2
        assert "Sign does not agree across splits" in res["line"]
        assert "2 of 3 confidence intervals include zero" in res["line"]

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        res = compute_down_sign_agreement(df)
        assert res["sign_agrees"] is False
        assert "No data" in res["line"]


# ---------------------------------------------------------------------------
# 2. Success-rate variant excluded from EPA plot
# ---------------------------------------------------------------------------

def test_success_rate_excluded_from_epa_plots():
    """Verify outcome='success' rows are not included in filter_robustness_variants."""
    df = pd.DataFrame([
        {
            "variant_id": "primary",
            "variant_label": "Primary Spec B",
            "split": "holdout",
            "outcome": "epa",
            "delta": 0.018,
            "ci_lo": -0.05,
            "ci_hi": 0.09,
            "n_plays": 50000,
        },
        {
            "variant_id": "outcome_success",
            "variant_label": "Success Rate",
            "split": "holdout",
            "outcome": "success",
            "delta": 0.006,
            "ci_lo": -0.02,
            "ci_hi": 0.03,
            "n_plays": 50000,
        },
    ])
    rows = filter_robustness_variants(df, split="holdout")
    # Only primary should be present, outcome_success excluded because outcome != 'epa'
    assert len(rows) == 1
    assert "Primary Spec B" in rows[0]["display_label"]
    assert not any("Success" in r["display_label"] for r in rows)


# ---------------------------------------------------------------------------
# 3. Noisy season thresholding
# ---------------------------------------------------------------------------

def test_noisy_season_tagging():
    """Verify seasons with CI width > threshold * |delta| receive (noisy) tag."""
    df = pd.DataFrame([
        {
            # |delta| = 0.05, width = 0.20 -> width > 1.0 * 0.05 -> noisy
            "variant_id": "season_2020",
            "split": "early",
            "outcome": "epa",
            "delta": 0.05,
            "ci_lo": -0.05,
            "ci_hi": 0.15,
            "n_plays": 20000,
        },
        {
            # |delta| = 0.30, width = 0.10 -> width < 1.0 * 0.30 -> not noisy
            "variant_id": "season_2021",
            "split": "early",
            "outcome": "epa",
            "delta": 0.30,
            "ci_lo": 0.25,
            "ci_hi": 0.35,
            "n_plays": 20000,
        },
    ])
    rows = filter_by_season_rows(df)
    assert len(rows) == 2
    assert "(noisy)" in rows[0]["display_label"]
    assert "(noisy)" not in rows[1]["display_label"]


# ---------------------------------------------------------------------------
# 4. Exploratory label on all plotted rows
# ---------------------------------------------------------------------------

def test_exploratory_label_on_all_rows():
    """Every plotted row label must carry the string '[exploratory]'."""
    bundle = load_results(source="fixtures")
    rob_df = bundle.robustness
    assert rob_df is not None

    var_rows = filter_robustness_variants(rob_df, split="holdout")
    down_rows, _ = filter_by_down_rows(rob_df)
    season_rows = filter_by_season_rows(rob_df)

    for r in var_rows + down_rows + season_rows:
        assert "[exploratory]" in r["display_label"]


# ---------------------------------------------------------------------------
# 5. Render on fixtures and real results
# ---------------------------------------------------------------------------

def test_render_exploratory_on_fixtures(monkeypatch):
    """Verify test_panel render with exploratory checks succeeds on fixtures."""
    monkeypatch.setenv("TENDENCY_USE_FIXTURES", "1")

    with (
        patch("app.views.test_panel.st", _make_st_mock()),
        patch("app.views.test_panel.inject_css"),
        patch("app.views.test_panel.render_page_header"),
        patch("app.views.test_panel.render_page_footer"),
        patch("app.views.test_panel.verdict_banner"),
        patch("app.views.test_panel.metric_card"),
        patch("app.views.test_panel.caveat_callout"),
    ):
        from app.views.test_panel import render
        render()


def test_render_exploratory_on_real_results():
    """Verify test_panel render with exploratory checks succeeds on real results."""
    try:
        load_results(source="real")
    except Exception:
        pytest.skip("Real results not present in environment")

    with (
        patch("app.views.test_panel.st", _make_st_mock()),
        patch("app.views.test_panel.inject_css"),
        patch("app.views.test_panel.render_page_header"),
        patch("app.views.test_panel.render_page_footer"),
        patch("app.views.test_panel.verdict_banner"),
        patch("app.views.test_panel.metric_card"),
        patch("app.views.test_panel.caveat_callout"),
    ):
        from app.views.test_panel import render
        render()


# ---------------------------------------------------------------------------
# 6. Absence of banned causal words
# ---------------------------------------------------------------------------

_BANNED_PATTERNS = [
    re.compile(r"\bcauses?\b", re.IGNORECASE),
    re.compile(r"\bproves?\b", re.IGNORECASE),
    re.compile(r"\boptimal\b", re.IGNORECASE),
    re.compile(r"\bcausal\b", re.IGNORECASE),
]

@pytest.mark.parametrize(
    "filepath", ["app/views/exploratory.py", "app/views/test_panel.py"]
)
def test_no_banned_words_in_views(filepath):
    """Ensure no causal/absolute wording appears in exploratory.py or test_panel.py."""
    source = Path(filepath).read_text(encoding="utf-8")
    for pat in _BANNED_PATTERNS:
        matches = pat.findall(source)
        assert not matches, (
            f"Banned word pattern '{pat.pattern}' found in {filepath}: {matches}"
        )
