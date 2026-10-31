"""Tests for the Test Panel view (Deliverable D7.1).

Covers:
  1. Verdict state mapping for all three cases.
  2. Page renders cleanly on fixture data (TENDENCY_USE_FIXTURES=1).
  3. Stored sentence is preserved exactly in the banner call (no recomputation).
  4. No banned causal / absolute wording in test_panel.py source.
  5. Page renders gracefully when no bundle is available (empty_state path).
"""

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.views.test_panel import map_verdict_state

# ---------------------------------------------------------------------------
# 1. Verdict state mapping — all cases
# ---------------------------------------------------------------------------

class TestMapVerdictState:
    """Verify the three-branch verdict → design-system mapping."""

    def test_supported_positive_returns_premium(self):
        assert map_verdict_state("supported", "positive") == "premium"

    def test_supported_negative_returns_reverse(self):
        assert map_verdict_state("supported", "negative") == "reverse"

    def test_not_supported_returns_cant_tell(self):
        assert (
            map_verdict_state("not supported / inconclusive", "positive")
            == "cant_tell"
        )

    def test_supported_zero_returns_cant_tell(self):
        # "zero" direction with "supported" falls through to cant_tell
        assert map_verdict_state("supported", "zero") == "cant_tell"

    def test_inconclusive_negative_returns_cant_tell(self):
        assert (
            map_verdict_state("not supported / inconclusive", "negative")
            == "cant_tell"
        )


# ---------------------------------------------------------------------------
# 2. Page renders on fixture data
# ---------------------------------------------------------------------------

def _make_st_mock():
    """Return a MagicMock that accepts arbitrary Streamlit calls."""
    m = MagicMock()
    # columns(n) must return n context-manager-compatible mocks
    def _fake_columns(n_or_spec, *args, **kwargs):
        count = n_or_spec if isinstance(n_or_spec, int) else len(n_or_spec)
        cols = []
        for _ in range(count):
            c = MagicMock()
            c.__enter__ = MagicMock(return_value=c)
            c.__exit__ = MagicMock(return_value=False)
            cols.append(c)
        return cols

    m.columns.side_effect = _fake_columns
    # tabs() must return 4 context-manager-compatible tabs
    tab = MagicMock()
    tab.__enter__ = MagicMock(return_value=tab)
    tab.__exit__ = MagicMock(return_value=False)
    m.tabs.return_value = (tab, tab, tab, tab)
    return m


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear st.cache_data between tests."""
    from app.data import get_results_bundle
    get_results_bundle.clear()
    yield
    get_results_bundle.clear()


def test_page_renders_on_fixtures(monkeypatch):
    """render() must complete without error when TENDENCY_USE_FIXTURES=1."""
    monkeypatch.setenv("TENDENCY_USE_FIXTURES", "1")

    with (
        patch("app.views.test_panel.st", _make_st_mock()),
        patch("app.views.test_panel.inject_css"),
        patch("app.views.test_panel.render_page_header"),
        patch("app.views.test_panel.render_page_footer"),
        patch("app.views.test_panel.verdict_banner") as mock_banner,
        patch("app.views.test_panel.metric_card"),
        patch("app.views.test_panel.caveat_callout"),
        patch("app.views.test_panel.empty_state") as mock_empty,
    ):
        from app.views.test_panel import render
        render()

    # empty_state must NOT have been called (real data is present via fixtures)
    mock_empty.assert_not_called()
    # verdict_banner must have been called exactly once
    mock_banner.assert_called_once()


# ---------------------------------------------------------------------------
# 3. Stored sentence is passed to verdict_banner unchanged
# ---------------------------------------------------------------------------

def test_displayed_sentence_equals_stored(monkeypatch):
    """The sentence passed to verdict_banner must equal primary.verdict.sentence."""
    monkeypatch.setenv("TENDENCY_USE_FIXTURES", "1")

    with (
        patch("app.views.test_panel.st", _make_st_mock()),
        patch("app.views.test_panel.inject_css"),
        patch("app.views.test_panel.render_page_header"),
        patch("app.views.test_panel.render_page_footer"),
        patch("app.views.test_panel.verdict_banner") as mock_banner,
        patch("app.views.test_panel.metric_card"),
        patch("app.views.test_panel.caveat_callout"),
        patch("app.views.test_panel.empty_state"),
    ):
        from app.views.test_panel import render
        render()

    # Retrieve the actual sentence from the fixture bundle
    from app.data import get_results_bundle
    get_results_bundle.clear()
    monkeypatch.setenv("TENDENCY_USE_FIXTURES", "1")
    res = get_results_bundle()
    stored_sentence = res.bundle.primary.verdict.sentence

    called_sentence = mock_banner.call_args[0][0]
    assert called_sentence == stored_sentence, (
        f"Banner sentence differs from stored sentence.\n"
        f"  Displayed : {called_sentence!r}\n"
        f"  Stored    : {stored_sentence!r}"
    )


# ---------------------------------------------------------------------------
# 4. No banned causal / absolute wording in test_panel.py source
# ---------------------------------------------------------------------------

_BANNED_PATTERNS = [
    re.compile(r"\bcauses?\b", re.IGNORECASE),
    re.compile(r"\bproves?\b", re.IGNORECASE),
    re.compile(r"\boptimal\b", re.IGNORECASE),
    re.compile(r"\bcausal\b", re.IGNORECASE),
]

def test_no_banned_words_in_source():
    """test_panel.py must not contain causal or absolute-claim language."""
    source = Path("app/views/test_panel.py").read_text(encoding="utf-8")
    for pat in _BANNED_PATTERNS:
        matches = pat.findall(source)
        assert not matches, (
            f"Banned word pattern '{pat.pattern}' found in test_panel.py: {matches}"
        )


# ---------------------------------------------------------------------------
# 5. Empty-state path when no bundle
# ---------------------------------------------------------------------------

def test_empty_state_rendered_when_no_bundle(monkeypatch):
    """When no results bundle is available, empty_state is called."""
    from app.data import LoadResult

    fake_load = LoadResult(bundle=None, is_fixtures=True)

    with (
        patch("app.views.test_panel.get_results_bundle", return_value=fake_load),
        patch("app.views.test_panel.st", _make_st_mock()),
        patch("app.views.test_panel.inject_css"),
        patch("app.views.test_panel.render_page_header"),
        patch("app.views.test_panel.render_page_footer"),
        patch("app.views.test_panel.verdict_banner") as mock_banner,
        patch("app.views.test_panel.empty_state") as mock_empty,
        patch("app.views.test_panel.metric_card"),
        patch("app.views.test_panel.caveat_callout"),
    ):
        from app.views.test_panel import render
        render()

    mock_empty.assert_called_once()
    mock_banner.assert_not_called()

