"""Tests for app/views/situation_explorer.py (Deliverable D8.1).

Tests:
  1. parse_explorer_state returns defaults when no query params are set.
  2. parse_explorer_state returns valid params passed via query params.
  3. parse_explorer_state falls back to defaults on invalid params without error.
  4. cell_id lookup finds records for all 27 combinations × 4 splits in fixtures.
  5. Page renders without error on fixture data (TENDENCY_USE_FIXTURES=1).
  6. Page renders without error on real results if present.
  7. No banned causal words in the view source.
"""

from __future__ import annotations

import ast
import re
from unittest.mock import MagicMock, patch

import pytest

from app.data import get_results_bundle, load_results
from app.views.situation_explorer import (
    DEFAULT_DIST,
    DEFAULT_DOWN,
    DEFAULT_SPLIT,
    DEFAULT_ZONE,
    VALID_DISTS,
    VALID_DOWNS,
    VALID_SPLITS,
    VALID_ZONES,
    get_cell_id,
    lookup_cell_record,
    parse_explorer_state,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_st_mock() -> MagicMock:
    """Return a MagicMock that behaves like the streamlit module."""
    m = MagicMock()
    # pills / segmented_control return first option by default
    m.pills.return_value = None
    m.segmented_control.return_value = None

    # columns() must return exactly as many MagicMocks as the int arg requests
    def _columns(n, *args, **kwargs):
        if isinstance(n, int):
            return [MagicMock() for _ in range(n)]
        # list/tuple form: one entry per spec element
        return [MagicMock() for _ in n]

    m.columns.side_effect = _columns
    m.tabs.return_value = [MagicMock(), MagicMock()]
    m.expander.return_value.__enter__ = lambda s: s
    m.expander.return_value.__exit__ = MagicMock(return_value=False)
    return m


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear st.cache_data between test invocations."""
    get_results_bundle.clear()
    yield
    get_results_bundle.clear()


# ---------------------------------------------------------------------------
# 1. Default state
# ---------------------------------------------------------------------------

def test_parse_explorer_state_defaults():
    """With no query params, parse_explorer_state returns all defaults."""
    with patch(
        "app.views.situation_explorer.get_query_param", return_value=None
    ):
        state = parse_explorer_state()

    assert state["down"] == DEFAULT_DOWN == 1
    assert state["dist"] == DEFAULT_DIST == "short"
    assert state["zone"] == DEFAULT_ZONE == "midfield"
    assert state["split"] == DEFAULT_SPLIT == "tune"


# ---------------------------------------------------------------------------
# 2. Valid params round-trip
# ---------------------------------------------------------------------------

def test_parse_explorer_state_valid_params():
    """Valid query params are parsed and returned correctly."""
    param_map = {
        "down": "2",
        "dist": "medium",
        "zone": "red_zone",
        "split": "holdout",
    }

    def fake_get(key, default=None):
        return param_map.get(key, default)

    with patch(
        "app.views.situation_explorer.get_query_param", side_effect=fake_get
    ):
        state = parse_explorer_state()

    assert state["down"] == 2
    assert state["dist"] == "medium"
    assert state["zone"] == "red_zone"
    assert state["split"] == "holdout"


# ---------------------------------------------------------------------------
# 3. Invalid params fall back to defaults
# ---------------------------------------------------------------------------

def test_parse_explorer_state_invalid_falls_back():
    """Invalid query param values fall back to defaults without raising."""
    param_map = {
        "down": "99",
        "dist": "xxxxx",
        "zone": "invalid_zone",
        "split": "badval",
    }

    def fake_get(key, default=None):
        return param_map.get(key, default)

    with patch(
        "app.views.situation_explorer.get_query_param", side_effect=fake_get
    ):
        state = parse_explorer_state()

    assert state["down"] == DEFAULT_DOWN
    assert state["dist"] == DEFAULT_DIST
    assert state["zone"] == DEFAULT_ZONE
    assert state["split"] == DEFAULT_SPLIT


def test_parse_explorer_state_non_numeric_down_falls_back():
    """A non-numeric 'down' param falls back to default without raising."""
    def fake_get(key, default=None):
        return "abc" if key == "down" else None

    with patch(
        "app.views.situation_explorer.get_query_param", side_effect=fake_get
    ):
        state = parse_explorer_state()

    assert state["down"] == DEFAULT_DOWN


# ---------------------------------------------------------------------------
# 4. Cell-id lookup — all 27 combinations × 4 splits
# ---------------------------------------------------------------------------

def test_cell_id_format():
    """get_cell_id formats identifiers correctly."""
    assert get_cell_id(1, "short", "midfield") == "1_short_midfield"
    assert get_cell_id(3, "long", "red_zone") == "3_long_red_zone"
    assert get_cell_id(2, "medium", "backed_up") == "2_medium_backed_up"


def test_cell_id_lookup_all_27_combinations():
    """lookup_cell_record finds a record for every valid combination in fixtures."""
    bundle = load_results(source="fixtures")
    cells_df = bundle.cells
    assert cells_df is not None, "Fixture bundle must include cells DataFrame"

    missing = []
    for split in VALID_SPLITS:
        for down in VALID_DOWNS:
            for dist in VALID_DISTS:
                for zone in VALID_ZONES:
                    rec = lookup_cell_record(
                        cells_df, split=split, down=down, dist=dist, zone=zone
                    )
                    if rec is None:
                        missing.append(f"{split}/{down}_{dist}_{zone}")

    assert not missing, f"Missing cell records: {missing}"


# ---------------------------------------------------------------------------
# 5. Render on fixture data
# ---------------------------------------------------------------------------

def test_situation_explorer_renders_on_fixtures(monkeypatch):
    """render() executes without error on fixture bundle; empty_state not called."""
    monkeypatch.setenv("TENDENCY_USE_FIXTURES", "1")

    with (
        patch("app.views.situation_explorer.st", _make_st_mock()),
        patch("app.views.situation_explorer.inject_css"),
        patch("app.views.situation_explorer.render_page_header"),
        patch("app.views.situation_explorer.render_page_footer"),
        patch("app.views.situation_explorer.get_query_param", return_value=None),
        patch("app.views.situation_explorer.set_query_param"),
        patch("app.views.situation_explorer.empty_state") as mock_empty,
    ):
        from app.views.situation_explorer import render
        render()

    mock_empty.assert_not_called()


# ---------------------------------------------------------------------------
# 6. Render on real results (optional, skipped if not present)
# ---------------------------------------------------------------------------

def test_situation_explorer_renders_on_real_results():
    """render() executes without error on real results when available."""
    try:
        load_results(source="real")
    except Exception:
        pytest.skip("Real results not present in environment")

    with (
        patch("app.views.situation_explorer.st", _make_st_mock()),
        patch("app.views.situation_explorer.inject_css"),
        patch("app.views.situation_explorer.render_page_header"),
        patch("app.views.situation_explorer.render_page_footer"),
        patch("app.views.situation_explorer.get_query_param", return_value=None),
        patch("app.views.situation_explorer.set_query_param"),
    ):
        from app.views.situation_explorer import render
        render()


# ---------------------------------------------------------------------------
# 7. Banned causal words absent
# ---------------------------------------------------------------------------

BANNED_WORDS = re.compile(
    r"\b(causes?|proves?|optimal|causal)\b", re.IGNORECASE
)

def test_no_banned_words_in_situation_explorer():
    """No causal/definitive language in the view source (AGENTS.md §9)."""
    import pathlib
    src = pathlib.Path("app/views/situation_explorer.py").read_text(encoding="utf-8")
    # Strip comments and docstrings for the check
    try:
        tree = ast.parse(src)
        # Collect all string literals (docstrings / inline strings)
        strings = [
            node.s
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.s, str)
        ]
        text_to_check = "\n".join(strings)
    except SyntaxError:
        text_to_check = src  # fall back to raw source

    matches = BANNED_WORDS.findall(text_to_check)
    assert not matches, f"Banned causal words found: {matches}"
