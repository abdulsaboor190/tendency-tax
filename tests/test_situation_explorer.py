"""Tests for app/views/situation_explorer.py (Deliverables D8.1 and D8.2).

Tests:
  1. parse_explorer_state returns defaults when no query params are set.
  2. parse_explorer_state returns valid params passed via query params.
  3. parse_explorer_state falls back to defaults on invalid params without error.
  4. cell_id lookup finds records for all 27 combinations × 4 splits in fixtures.
  5. Page renders without error on fixture data (TENDENCY_USE_FIXTURES=1).
  6. Page renders without error on real results if present.
  7. All 27 cells render without error across splits on real results (including
     every non-testable cell).
  8. All 27 cells render without error on fixture data.
  9. For 'cant_tell' verdict, never show a bare point estimate (always interval).
  10. Untestable / below_min_plays cells do not display a point estimate.
  11. Cross-split split rows structure for interval_plot is valid.
  12. No banned causal words in the view source.
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
    build_cell_split_rows,
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
    m.pills.return_value = None
    m.segmented_control.return_value = None

    def _columns(n, *args, **kwargs):
        if isinstance(n, int):
            return [MagicMock() for _ in range(n)]
        return [MagicMock() for _ in n]

    m.columns.side_effect = _columns
    m.tabs.return_value = [MagicMock(), MagicMock()]
    m.expander.return_value.__enter__ = lambda s: s
    m.expander.return_value.__exit__ = MagicMock(return_value=False)
    m.button.return_value = False
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
        patch(
            "app.views.situation_explorer.get_query_param", return_value=None
        ),
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
        patch(
            "app.views.situation_explorer.get_query_param", return_value=None
        ),
        patch("app.views.situation_explorer.set_query_param"),
    ):
        from app.views.situation_explorer import render

        render()


# ---------------------------------------------------------------------------
# 7. Render all 27 cells on real results (including untestable cells)
# ---------------------------------------------------------------------------


def test_all_27_cells_render_on_real_results():
    """Every one of the 27 cells in every split renders without raising on real data."""
    try:
        bundle = load_results(source="real")
    except Exception:
        pytest.skip("Real results not present in environment")

    cells_df = bundle.cells
    assert cells_df is not None

    from app.views.situation_explorer import render

    for down in VALID_DOWNS:
        for dist in VALID_DISTS:
            for zone in VALID_ZONES:
                for split in VALID_SPLITS:
                    state_map = {
                        "down": str(down),
                        "dist": dist,
                        "zone": zone,
                        "split": split,
                    }

                    def fake_get(key, default=None, _sm=state_map):
                        return _sm.get(key, default)

                    with (
                        patch(
                            "app.views.situation_explorer.st", _make_st_mock()
                        ),
                        patch("app.views.situation_explorer.inject_css"),
                        patch(
                            "app.views.situation_explorer.render_page_header"
                        ),
                        patch(
                            "app.views.situation_explorer.render_page_footer"
                        ),
                        patch(
                            "app.views.situation_explorer.get_query_param",
                            side_effect=fake_get,
                        ),
                        patch("app.views.situation_explorer.set_query_param"),
                        patch(
                            "app.views.situation_explorer.empty_state"
                        ) as mock_empty,
                    ):
                        render()
                        mock_empty.assert_not_called()


def test_all_27_cells_render_on_fixtures(monkeypatch):
    """Every one of the 27 cells in every split renders cleanly on fixtures."""
    monkeypatch.setenv("TENDENCY_USE_FIXTURES", "1")
    from app.views.situation_explorer import render

    for down in VALID_DOWNS:
        for dist in VALID_DISTS:
            for zone in VALID_ZONES:
                state_map = {
                    "down": str(down),
                    "dist": dist,
                    "zone": zone,
                    "split": "tune",
                }

                def fake_get(key, default=None, _sm=state_map):
                    return _sm.get(key, default)

                with (
                    patch("app.views.situation_explorer.st", _make_st_mock()),
                    patch("app.views.situation_explorer.inject_css"),
                    patch("app.views.situation_explorer.render_page_header"),
                    patch("app.views.situation_explorer.render_page_footer"),
                    patch(
                        "app.views.situation_explorer.get_query_param",
                        side_effect=fake_get,
                    ),
                    patch("app.views.situation_explorer.set_query_param"),
                    patch(
                        "app.views.situation_explorer.empty_state"
                    ) as mock_empty,
                ):
                    render()
                    mock_empty.assert_not_called()


# ---------------------------------------------------------------------------
# 8. No bare point estimate for cant_tell or untestable
# ---------------------------------------------------------------------------


def test_no_bare_point_estimate_for_cant_tell():
    """For 'cant_tell', metric_card displays an interval, not a bare float."""
    from app.data import format_ci, format_epa

    # Sample cell with cant_tell
    try:
        bundle = load_results(source="real")
    except Exception:
        bundle = load_results(source="fixtures")

    cells_df = bundle.cells
    cant_tell_rows = cells_df[
        (cells_df["verdict"] == "cant_tell") & cells_df["testable"]
    ]
    if cant_tell_rows.empty:
        pytest.skip("No testable cant_tell row found")

    rec = cant_tell_rows.iloc[0].to_dict()
    down = rec["down"]
    dist = rec["dist_bucket"]
    zone = rec["zone"]
    split = rec["split"]
    delta_val = rec["delta"]
    ci_lo = rec["ci_lo"]
    ci_hi = rec["ci_hi"]

    bare_point = format_epa(delta_val)
    expected_interval_str = format_ci(ci_lo, ci_hi)

    state_map = {
        "down": str(down),
        "dist": dist,
        "zone": zone,
        "split": split,
    }

    mock_st = _make_st_mock()
    with (
        patch("app.views.situation_explorer.st", mock_st),
        patch("app.views.situation_explorer.inject_css"),
        patch("app.views.situation_explorer.render_page_header"),
        patch("app.views.situation_explorer.render_page_footer"),
        patch(
            "app.views.situation_explorer.get_query_param",
            side_effect=lambda k, d=None: state_map.get(k, d),
        ),
        patch("app.views.situation_explorer.set_query_param"),
        patch("app.views.situation_explorer.metric_card") as mock_metric_card,
    ):
        from app.views.situation_explorer import render

        render()

    # Verify metric_card was called with interval string as value, not bare_point
    mock_metric_card.assert_called_once()
    call_kwargs = mock_metric_card.call_args.kwargs
    displayed_value = call_kwargs.get("value")
    assert displayed_value == expected_interval_str
    assert displayed_value != bare_point


def test_untestable_cell_shows_friendly_explanation_no_point_estimate():
    """Non-testable cell displays caveat_callout and 'Not Testable'."""
    try:
        bundle = load_results(source="real")
    except Exception:
        pytest.skip("Real results needed to test untestable cells")

    cells_df = bundle.cells
    untestable_rows = cells_df[~cells_df["testable"]]
    assert not untestable_rows.empty, "Real cells.parquet must have untestable cells"

    rec = untestable_rows.iloc[0].to_dict()
    state_map = {
        "down": str(rec["down"]),
        "dist": rec["dist_bucket"],
        "zone": rec["zone"],
        "split": rec["split"],
    }

    mock_st = _make_st_mock()
    with (
        patch("app.views.situation_explorer.st", mock_st),
        patch("app.views.situation_explorer.inject_css"),
        patch("app.views.situation_explorer.render_page_header"),
        patch("app.views.situation_explorer.render_page_footer"),
        patch(
            "app.views.situation_explorer.get_query_param",
            side_effect=lambda k, d=None: state_map.get(k, d),
        ),
        patch("app.views.situation_explorer.set_query_param"),
        patch("app.views.situation_explorer.metric_card") as mock_metric_card,
        patch("app.views.situation_explorer.caveat_callout") as mock_callout,
    ):
        from app.views.situation_explorer import render

        render()

    # Friendly caveat callout displayed
    mock_callout.assert_called()
    callout_text = mock_callout.call_args[0][0]
    assert "cannot be evaluated" in callout_text

    # Metric card shows 'Not Testable'
    mock_metric_card.assert_called_once()
    val = mock_metric_card.call_args.kwargs.get("value")
    assert val == "Not Testable"


# ---------------------------------------------------------------------------
# 9. build_cell_split_rows helper
# ---------------------------------------------------------------------------


def test_build_cell_split_rows():
    """build_cell_split_rows returns four rows matching splits in standard order."""
    bundle = load_results(source="fixtures")
    cells_df = bundle.cells
    rows = build_cell_split_rows(cells_df, "1_short_midfield")
    assert len(rows) == 4
    splits = [r["split"] for r in rows]
    assert splits == ["early", "tune", "holdout", "all"]
    for r in rows:
        assert "delta" in r
        assert "ci_lo" in r
        assert "ci_hi" in r
        assert "n_plays" in r
        assert "verdict" in r


# ---------------------------------------------------------------------------
# 10. Deliverable D8.3 Tests: Summary counts, 27-cell grid, fallback table
# ---------------------------------------------------------------------------


def test_compute_cells_summary_counts_match_real_cells():
    """Summary counts of testable cells and verdict categories match cells.parquet."""
    from app.ui.charts import compute_cells_summary

    try:
        bundle = load_results(source="real")
    except Exception:
        bundle = load_results(source="fixtures")

    cells_df = bundle.cells
    for split in VALID_SPLITS:
        sub = cells_df[cells_df["split"] == split]
        summary = compute_cells_summary(sub)

        assert summary["total"] == len(sub) == 27
        assert summary["n_testable"] == int(sub["testable"].sum())

        counts = sub["verdict"].value_counts().to_dict()
        assert summary["n_premium"] == int(counts.get("premium", 0))
        assert summary["n_reverse"] == int(counts.get("reverse", 0))
        assert summary["n_none"] == int(counts.get("none", 0))
        assert summary["n_cant_tell"] == int(counts.get("cant_tell", 0))

        # Check line format
        line = summary["summary_line"]
        assert f"{summary['n_testable']} of 27 cells testable" in line
        assert "premium" in line
        assert "can't tell" in line


def test_cell_grid_chart_has_27_cells():
    """cell_grid_chart produces 3 traces with 9 bars each = 27 cells."""
    from app.ui.charts import cell_grid_chart

    bundle = load_results(source="fixtures")
    cells_df = bundle.cells
    sub = cells_df[cells_df["split"] == "tune"]

    fig = cell_grid_chart(sub)
    # 3 field zone traces
    assert len(fig.data) == 3
    total_cells = sum(len(trace.x) for trace in fig.data)
    assert total_cells == 27

    # Check customdata format
    for trace in fig.data:
        assert len(trace.customdata) == 9
        for row in trace.customdata:
            assert len(row) >= 4
            down_val, dist_val, zone_val, cid = row[:4]
            assert down_val in VALID_DOWNS
            assert dist_val in VALID_DISTS
            assert zone_val in VALID_ZONES
            assert cid == f"{down_val}_{dist_val}_{zone_val}"


def test_fallback_table_has_27_rows():
    """build_fallback_table produces a 27-row DataFrame for accessibility."""
    from app.views.situation_explorer import build_fallback_table

    try:
        bundle = load_results(source="real")
    except Exception:
        bundle = load_results(source="fixtures")

    cells_df = bundle.cells
    for split in VALID_SPLITS:
        table_df = build_fallback_table(cells_df, split)
        assert len(table_df) == 27
        assert "Cell ID" in table_df.columns
        assert "Down" in table_df.columns
        assert "Distance" in table_df.columns
        assert "Field Zone" in table_df.columns
        assert "State" in table_df.columns
        assert "Delta (EPA)" in table_df.columns
        assert "95% CI" in table_df.columns
        assert "Plays" in table_df.columns
        assert "Reason" in table_df.columns


# ---------------------------------------------------------------------------
# 11. Banned causal words absent
# ---------------------------------------------------------------------------


BANNED_WORDS = re.compile(
    r"\b(causes?|proves?|optimal|causal)\b", re.IGNORECASE
)


def test_no_banned_words_in_situation_explorer():
    """No causal/definitive language in the view source (AGENTS.md §9)."""
    import pathlib

    src = pathlib.Path("app/views/situation_explorer.py").read_text(
        encoding="utf-8"
    )
    try:
        tree = ast.parse(src)
        strings = [
            node.s
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.s, str)
        ]
        text_to_check = "\n".join(strings)
    except SyntaxError:
        text_to_check = src

    matches = BANNED_WORDS.findall(text_to_check)
    assert not matches, f"Banned causal words found: {matches}"

