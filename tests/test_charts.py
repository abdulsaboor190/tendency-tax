"""Tests for the Central Decile Chart (Deliverable D7.2).

Covers:
  1. Dropped-bin logic on a hand-made DataFrame (thresholds for n_pass and n_run).
  2. Chart builder runs successfully on fixture data (early, holdout, tune).
  3. Chart builder runs successfully on real data results.
  4. Absence of banned causal words in charts.py and test_panel.py.
  5. Correct traces and layout properties in the Plotly figure (legend, zero line).
"""

import re
from pathlib import Path

import pandas as pd
import pytest

from app.data import load_results
from app.ui.charts import (
    NULL_SHIFT_SD_THRESHOLD,
    build_decile_gap_chart,
    build_placebo_histogram,
    check_placebo_shift,
    drop_low_play_deciles,
)

# ---------------------------------------------------------------------------
# 1. Dropped-bin logic on hand-made DataFrame
# ---------------------------------------------------------------------------

class TestDropLowPlayDeciles:
    """Test the filtering of deciles with low play counts (< min_plays)."""

    def test_drops_bins_below_min_plays(self):
        df = pd.DataFrame([
            {"split": "early", "decile": 1, "n_pass": 300, "n_run": 500},
            {"split": "early", "decile": 2, "n_pass": 150, "n_run": 500},
            {"split": "early", "decile": 3, "n_pass": 400, "n_run": 80},
            {"split": "tune", "decile": 4, "n_pass": 50, "n_run": 60},
            {"split": "holdout", "decile": 5, "n_pass": 200, "n_run": 200},
        ])

        filtered, dropped = drop_low_play_deciles(df, min_plays=200)

        # Retained: deciles 1 and 5
        assert len(filtered) == 2
        assert filtered["decile"].tolist() == [1, 5]

        # Dropped: deciles 2, 3, 4
        assert len(dropped) == 3
        d2 = dropped[0]
        assert d2["decile"] == 2 and "n_pass" in d2["reason"]
        d3 = dropped[1]
        assert d3["decile"] == 3 and "n_run" in d3["reason"]
        d4 = dropped[2]
        assert d4["decile"] == 4
        assert "n_pass" in d4["reason"] and "n_run" in d4["reason"]

    def test_empty_dataframe(self):
        empty_df = pd.DataFrame()
        filtered, dropped = drop_low_play_deciles(empty_df, min_plays=200)
        assert filtered.empty
        assert dropped == []


# ---------------------------------------------------------------------------
# 2. Chart builder on fixture data
# ---------------------------------------------------------------------------

def test_chart_builds_from_fixtures():
    """Verify that build_decile_gap_chart works seamlessly on fixture bundle."""
    bundle = load_results(source="fixtures")
    assert bundle.decile_gap_dev is not None
    assert bundle.decile_gap_holdout is not None

    early_df = bundle.decile_gap_dev[bundle.decile_gap_dev["split"] == "early"]
    tune_df = bundle.decile_gap_dev[bundle.decile_gap_dev["split"] == "tune"]
    holdout_df = bundle.decile_gap_holdout

    # Without tune
    fig, dropped = build_decile_gap_chart(
        early_df=early_df,
        holdout_df=holdout_df,
        tune_df=tune_df,
        include_tune=False,
        min_plays=200,
    )

    # Check dropped deciles (holdout decile 10 has n_run=110 < 200)
    assert any(d["split"] == "holdout" and d["decile"] == 10 for d in dropped)
    # Figure has traces for early line and holdout line (error bars directly on lines)
    assert len(fig.data) == 2

    # With tune
    fig_tune, dropped_tune = build_decile_gap_chart(
        early_df=early_df,
        holdout_df=holdout_df,
        tune_df=tune_df,
        include_tune=True,
        min_plays=200,
    )
    # Tune decile 10 has n_run=161 < 200
    assert any(d["split"] == "tune" and d["decile"] == 10 for d in dropped_tune)
    assert len(fig_tune.data) == 3


# ---------------------------------------------------------------------------
# 3. Chart builder on real data results
# ---------------------------------------------------------------------------

def test_chart_builds_from_real_results():
    """Verify that build_decile_gap_chart works on real results files if available."""
    try:
        bundle = load_results(source="real")
    except Exception:
        pytest.skip("Real results not present in environment")

    assert bundle.decile_gap_dev is not None
    early_df = bundle.decile_gap_dev[bundle.decile_gap_dev["split"] == "early"]
    holdout_df = bundle.decile_gap_holdout

    fig, dropped = build_decile_gap_chart(
        early_df=early_df,
        holdout_df=holdout_df,
        min_plays=200,
    )
    assert fig is not None
    # Real holdout decile 10 also has n_run=110 < 200
    assert any(d["split"] == "holdout" and d["decile"] == 10 for d in dropped)


# ---------------------------------------------------------------------------
# 4. Absence of banned causal words
# ---------------------------------------------------------------------------

_BANNED_PATTERNS = [
    re.compile(r"\bcauses?\b", re.IGNORECASE),
    re.compile(r"\bproves?\b", re.IGNORECASE),
    re.compile(r"\boptimal\b", re.IGNORECASE),
    re.compile(r"\bcausal\b", re.IGNORECASE),
]

@pytest.mark.parametrize("filepath", ["app/ui/charts.py", "app/views/test_panel.py"])
def test_no_banned_words_in_code(filepath):
    """Ensure no causal or absolute-claim vocabulary exists in chart/view code."""
    source = Path(filepath).read_text(encoding="utf-8")
    for pat in _BANNED_PATTERNS:
        matches = pat.findall(source)
        assert not matches, (
            f"Banned word pattern '{pat.pattern}' found in {filepath}: {matches}"
        )


# ---------------------------------------------------------------------------
# 5. Chart Layout and Style Checks
# ---------------------------------------------------------------------------

def test_chart_layout_specs():
    """Verify legend orientation, zero line, and range styling."""
    df = pd.DataFrame([
        {
            "split": "early",
            "decile": 1,
            "phat_min": 0.1,
            "phat_max": 0.3,
            "n_pass": 500,
            "n_run": 500,
            "epa_gap_pass_minus_run": 0.05,
            "ci_95_low": 0.01,
            "ci_95_high": 0.09,
        }
    ])
    fig, _ = build_decile_gap_chart(early_df=df, min_plays=200)

    # Legend is horizontal and placed below
    assert fig.layout.legend.orientation == "h"
    assert fig.layout.legend.y < 0

    # Horizontal zero reference line exists in layout shapes
    shapes = fig.layout.shapes
    assert any(s.type == "line" and s.y0 == 0.0 and s.y1 == 0.0 for s in shapes)


# ---------------------------------------------------------------------------
# 6. Placebo Shift Warning Tests (Deliverable D7.3)
# ---------------------------------------------------------------------------

class TestPlaceboShiftWarning:
    """Verify warning triggers when |median| > 0.5 * SD, and not when centered."""

    def test_centered_null_triggers_no_warning(self):
        """Centered null (median = 0.01, SD = 0.10, threshold = 0.05) -> no warning."""
        centered_deltas = [
            -0.08, -0.05, -0.02, 0.00, 0.01, 0.02, 0.04, 0.07, 0.09,
        ]
        stats = check_placebo_shift(
            centered_deltas, threshold_factor=NULL_SHIFT_SD_THRESHOLD
        )
        assert stats["is_shifted"] is False

    def test_shifted_null_triggers_warning(self):
        """Shifted null (median = -0.04, SD = 0.02, threshold = 0.01) -> warning."""
        shifted_deltas = [
            -0.07, -0.06, -0.05, -0.04, -0.04, -0.03, -0.02, -0.01,
        ]
        stats = check_placebo_shift(
            shifted_deltas, threshold_factor=NULL_SHIFT_SD_THRESHOLD
        )
        assert stats["is_shifted"] is True

    def test_placebo_histogram_builds_on_fixtures(self):
        """Verify placebo histogram constructs properly from fixture bundle."""
        bundle = load_results(source="fixtures")
        assert bundle.placebo_holdout is not None

        fig, stats = build_placebo_histogram(
            placebo_df=bundle.placebo_holdout,
            observed_delta=0.01878,
            split_name="Holdout",
        )
        assert fig is not None
        assert "median" in stats
        # Fixture placebo holdout has median ≈ -0.038, SD ≈ 0.027 -> is_shifted == True
        assert stats["is_shifted"] is True
        # Verify vertical lines for median and observed delta exist in layout shapes
        shapes = fig.layout.shapes
        assert any(s.type == "line" and s.x0 == stats["median"] for s in shapes)
        assert any(s.type == "line" and s.x0 == 0.01878 for s in shapes)

    def test_placebo_histogram_builds_on_real_results(self):
        """Verify placebo histogram constructs properly on real results if present."""
        try:
            bundle = load_results(source="real")
        except Exception:
            pytest.skip("Real results not present in environment")

        if bundle.placebo_holdout is None:
            pytest.skip("placebo_holdout not present in real results")

        fig, stats = build_placebo_histogram(
            placebo_df=bundle.placebo_holdout,
            observed_delta=0.01878,
        )
        assert fig is not None
        assert stats["is_shifted"] is True


# ---------------------------------------------------------------------------
# 6. Absence of raw HTML entities in chart text
# ---------------------------------------------------------------------------

def test_no_raw_html_entities_in_chart_code():
    """Verify no literal HTML entities (&minus;, &ndash;, etc.) appear in charts.py."""
    chart_py = Path("app/ui/charts.py").read_text(encoding="utf-8")
    # Matches patterns like &minus;, &ndash;, &#770;, etc.
    entity_pattern = re.compile(r"&[a-zA-Z0-9#]+;")
    matches = entity_pattern.findall(chart_py)
    assert not matches, f"Raw HTML entities found in app/ui/charts.py: {matches}"


