"""Tests for the Method Drawer and About the Method view (Deliverable D7.4).

Covers:
  1. Method view renders successfully on fixture data (TENDENCY_USE_FIXTURES=1).
  2. Method view renders successfully on real data results.
  3. Numbers displayed equal those in primary.json (n_plays, delta, CIs, etc.).
  4. Absence of banned causal/absolute words in about_method.py and test_panel.py.
  5. Empty state handling when no results bundle is present.
"""

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.data import format_ci, format_epa, format_int, get_results_bundle, load_results


def _make_st_mock():
    """Return a MagicMock capturing streamlit calls."""
    m = MagicMock()
    # Expander context manager support
    exp = MagicMock()
    exp.__enter__ = MagicMock(return_value=exp)
    exp.__exit__ = MagicMock(return_value=False)
    m.expander.return_value = exp

    # Columns support
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
# 1. Render on fixtures
# ---------------------------------------------------------------------------

def test_about_method_renders_on_fixtures(monkeypatch):
    """Verify render() executes without error on fixture bundle."""
    monkeypatch.setenv("TENDENCY_USE_FIXTURES", "1")

    with (
        patch("app.views.about_method.st", _make_st_mock()),
        patch("app.views.about_method.inject_css"),
        patch("app.views.about_method.render_page_header"),
        patch("app.views.about_method.render_page_footer"),
        patch("app.views.about_method.empty_state") as mock_empty,
    ):
        from app.views.about_method import render
        render()

    mock_empty.assert_not_called()


# ---------------------------------------------------------------------------
# 2. Render on real results
# ---------------------------------------------------------------------------

def test_about_method_renders_on_real_results():
    """Verify render() executes without error on real results if present."""
    try:
        load_results(source="real")
    except Exception:
        pytest.skip("Real results not present in environment")

    with (
        patch("app.views.about_method.st", _make_st_mock()),
        patch("app.views.about_method.inject_css"),
        patch("app.views.about_method.render_page_header"),
        patch("app.views.about_method.render_page_footer"),
        patch("app.views.about_method.empty_state") as mock_empty,
    ):
        from app.views.about_method import render
        render()

    mock_empty.assert_not_called()


# ---------------------------------------------------------------------------
# 3. Numbers match primary.json exactly
# ---------------------------------------------------------------------------

def test_numbers_match_primary_json(monkeypatch):
    """Numbers displayed in method text must match primary.json values."""
    monkeypatch.setenv("TENDENCY_USE_FIXTURES", "1")
    bundle = load_results(source="fixtures")
    prim = bundle.primary

    st_mock = _make_st_mock()
    written_texts: list[str] = []

    def mock_write(*args, **kwargs):
        for a in args:
            written_texts.append(str(a))

    def mock_markdown(*args, **kwargs):
        for a in args:
            written_texts.append(str(a))

    st_mock.write.side_effect = mock_write
    st_mock.markdown.side_effect = mock_markdown

    with (
        patch("app.views.about_method.st", st_mock),
        patch("app.views.about_method.inject_css"),
        patch("app.views.about_method.render_page_header"),
        patch("app.views.about_method.render_page_footer"),
    ):
        from app.views.about_method import render
        render()

    all_content = " ".join(written_texts)

    # Check play counts
    assert format_int(prim.early.n_plays) in all_content
    assert format_int(prim.tune.n_plays) in all_content
    assert format_int(prim.holdout.n_plays) in all_content
    assert format_int(prim.holdout.n_games) in all_content

    # Check estimates and CIs
    sb = prim.holdout.spec_b
    assert format_epa(sb.delta) in all_content
    assert format_ci(sb.delta_ci_95[0], sb.delta_ci_95[1]) in all_content
    assert format_epa(prim.early.spec_b.delta) in all_content

    # Check commit and timestamp
    assert prim.plan_commit[:8] in all_content
    assert prim.timestamp in all_content


# ---------------------------------------------------------------------------
# 4. Absence of banned causal words
# ---------------------------------------------------------------------------

_BANNED_PATTERNS = [
    re.compile(r"\bcauses?\b", re.IGNORECASE),
    re.compile(r"\bproves?\b", re.IGNORECASE),
    re.compile(r"\boptimal\b", re.IGNORECASE),
    re.compile(r"\bcausal\b", re.IGNORECASE),
]

@pytest.mark.parametrize(
    "filepath", ["app/views/about_method.py", "app/views/test_panel.py"]
)
def test_no_banned_words_in_method_views(filepath):
    """Ensure no causal/absolute wording appears in about_method.py or test_panel.py."""
    source = Path(filepath).read_text(encoding="utf-8")
    for pat in _BANNED_PATTERNS:
        matches = pat.findall(source)
        assert not matches, (
            f"Banned word pattern '{pat.pattern}' found in {filepath}: {matches}"
        )


# ---------------------------------------------------------------------------
# 5. Empty state when no bundle
# ---------------------------------------------------------------------------

def test_empty_state_when_bundle_missing():
    """Verify empty_state is rendered when primary result is None."""
    from app.data import LoadResult
    empty_res = LoadResult(bundle=None, is_fixtures=True)

    with (
        patch("app.views.about_method.get_results_bundle", return_value=empty_res),
        patch("app.views.about_method.st", _make_st_mock()),
        patch("app.views.about_method.inject_css"),
        patch("app.views.about_method.render_page_header"),
        patch("app.views.about_method.render_page_footer"),
        patch("app.views.about_method.empty_state") as mock_empty,
    ):
        from app.views.about_method import render
        render()

    mock_empty.assert_called_once()
