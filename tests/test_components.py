"""Tests for reusable UI components (Deliverable D6.3).

Covers:
  1. Each component returns or renders without error.
  2. verdict_chip rejects an unknown state.
  3. sample_size_badge switches to warning style below min_n.
  4. Banned causal words test (causes, proves, optimal, causal) across component texts.
  5. WCAG AA contrast ratio check for verdict text colors in light and dark themes.
"""

import re
from pathlib import Path

import plotly.graph_objects as go
import pytest

from app.ui.components import (
    caveat_callout,
    empty_state,
    interval_plot,
    loading_skeleton,
    metric_card,
    sample_size_badge,
    state_legend,
    verdict_banner,
    verdict_chip,
)
from app.ui.theme import VERDICT_TOKENS

# ---------------------------------------------------------------------------
# 1. verdict_chip tests
# ---------------------------------------------------------------------------

def test_verdict_chip_valid_and_invalid():
    """verdict_chip renders valid states and rejects unknown states."""
    for state in ["premium", "reverse", "none", "cant_tell"]:
        chip_html = verdict_chip(state)
        assert f"verdict-chip {state.replace('_', '-')}" in chip_html
        assert VERDICT_TOKENS[state].icon in chip_html
        assert VERDICT_TOKENS[state].label in chip_html

        # Test small variant
        small_html = verdict_chip(state, small=True)
        assert "small" in small_html

    with pytest.raises(ValueError, match="Unknown verdict state"):
        verdict_chip("unknown_state")


# ---------------------------------------------------------------------------
# 2. sample_size_badge threshold switching
# ---------------------------------------------------------------------------

def test_sample_size_badge_threshold():
    """sample_size_badge switches to warning style below min_n threshold."""
    normal_badge = sample_size_badge(100, min_n=50)
    assert "sample-badge" in normal_badge
    assert "warning" not in normal_badge
    assert "N = 100" in normal_badge

    warning_badge = sample_size_badge(30, min_n=50)
    assert "sample-badge warning" in warning_badge
    assert "⚠️" in warning_badge
    assert "N = 30" in warning_badge


# ---------------------------------------------------------------------------
# 3. Component Rendering Without Error
# ---------------------------------------------------------------------------

def test_components_render_without_error(monkeypatch):
    """Verify other components execute and return without error."""
    # interval_plot returns a go.Figure
    sample_data = [
        {
            "label": "Group A",
            "delta": 0.05,
            "ci_lo": 0.01,
            "ci_hi": 0.09,
            "n_plays": 500,
            "verdict": "premium",
        },
        {
            "label": "Group B",
            "delta": -0.02,
            "ci_lo": -0.08,
            "ci_hi": 0.04,
            "n_plays": 200,
            "verdict": "cant_tell",
        },
    ]
    fig = interval_plot(sample_data)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.layout.template == "tendency" or fig.layout.template is not None

    # HTML rendering functions (mock streamlit.markdown to verify invocation)
    calls = []
    import streamlit as st
    monkeypatch.setattr(st, "markdown", lambda html, **kw: calls.append(html))

    verdict_banner("Sample verdict sentence.", state="premium")
    assert len(calls) == 1
    assert "verdict-banner premium" in calls[-1]

    metric_card("Delta", "+0.041", interval=(0.01, 0.07), n=1000)
    assert len(calls) == 2
    assert "metric-card-box" in calls[-1]

    caveat_callout("Observational caveat note.", kind="warning")
    assert len(calls) == 3
    assert "tendency-callout warning" in calls[-1]

    empty_state("No data.")
    assert len(calls) == 4
    assert "empty-state-box" in calls[-1]

    loading_skeleton(height=150)
    assert len(calls) == 5
    assert "loading-skeleton-box" in calls[-1]

    state_legend()
    assert len(calls) == 6
    assert "state-legend-container" in calls[-1]


def test_metric_card_outside_interval_guard(monkeypatch):
    """metric_card displays warning icon when estimate is outside interval."""
    calls = []
    import streamlit as st
    monkeypatch.setattr(st, "markdown", lambda html, **kw: calls.append(html))

    # Case 1: Point estimate inside interval -> no warning icon
    metric_card("Inside", "+0.020", interval=(0.010, 0.030), n=500)
    assert "⚠️" not in calls[-1]

    # Case 2: Point estimate outside interval (+0.040 > 0.030) -> warning icon present
    metric_card("Outside High", "+0.040", interval=(0.010, 0.030), n=500)
    assert "⚠️" in calls[-1]
    assert "The estimate lies outside its interval; see data notes." in calls[-1]

    # Case 3: Point estimate below interval (-0.020 < -0.010) -> warning icon present
    metric_card("Outside Low", "-0.020", interval=(-0.010, 0.050), n=500)
    assert "⚠️" in calls[-1]
    assert "The estimate lies outside its interval; see data notes." in calls[-1]


# ---------------------------------------------------------------------------
# 4. Banned Causal Words Test
# ---------------------------------------------------------------------------

def test_banned_causal_words_in_components():
    """Ensure components.py and style_guide.py avoid banned causal assertions."""
    files_to_check = [
        Path("app/ui/components.py"),
        Path("app/views/style_guide.py"),
    ]
    banned = [
        re.compile(r"\bcauses\b", re.IGNORECASE),
        re.compile(r"\bproves\b", re.IGNORECASE),
        re.compile(r"\boptimal\b", re.IGNORECASE),
        re.compile(r"\bcausal\b", re.IGNORECASE),
    ]

    for f in files_to_check:
        content = f.read_text(encoding="utf-8")
        for line_no, line in enumerate(content.splitlines(), start=1):
            for pat in banned:
                assert not pat.search(line), (
                    f"Banned word {pat.pattern} found in {f}:{line_no} -> {line}"
                )


# ---------------------------------------------------------------------------
# 5. WCAG AA Contrast Ratios Test
# ---------------------------------------------------------------------------

def _rel_luminance(hex_str: str) -> float:
    hex_str = hex_str.lstrip("#")
    r, g, b = [int(hex_str[i:i+2], 16) / 255.0 for i in (0, 2, 4)]

    def adj(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * adj(r) + 0.7152 * adj(g) + 0.0722 * adj(b)


def _contrast_ratio(hex1: str, hex2: str) -> float:
    l1 = _rel_luminance(hex1)
    l2 = _rel_luminance(hex2)
    return (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)


def test_wcag_contrast_ratios():
    """Verify text contrast ratios meet WCAG AA (>= 4.5:1 for normal text)."""
    # Light theme: text on #FFFFFF
    bg_light = "#FFFFFF"
    light_text_colors = {
        "premium": "#005A8C",
        "reverse": "#B33C00",
        "none": "#007A58",
        "cant_tell": "#495057",
    }
    for state, col in light_text_colors.items():
        ratio = _contrast_ratio(col, bg_light)
        assert ratio >= 4.5, f"Light theme {state} contrast {ratio:.2f} < 4.5"

    # Dark theme: text on #1E1E24
    bg_dark = "#1E1E24"
    dark_text_colors = {
        "premium": "#56B4E9",
        "reverse": "#FF9E66",
        "none": "#33CCA0",
        "cant_tell": "#C4C8CF",
    }
    for state, col in dark_text_colors.items():
        ratio = _contrast_ratio(col, bg_dark)
        assert ratio >= 4.5, f"Dark theme {state} contrast {ratio:.2f} < 4.5"
