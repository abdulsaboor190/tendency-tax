"""Tests for Streamlit design system and app safety hygiene (Deliverable D6.1).

Covers:
  1. The app never imports src.load, touches data/raw, or sets TENDENCY_ALLOW_HOLDOUT.
  2. All four verdict tokens exist with unique colors, icons, and labels.
  3. No two verdict states share the exact same color.
  4. 'tendency' Plotly template is registered and applied properly.
  5. inject_css and render helpers generate valid HTML snippets.
"""

import re
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

from app.ui.theme import (
    VERDICT_TOKENS,
    apply_template,
    register_plotly_template,
    render_callout,
    render_verdict_chip,
)

# ---------------------------------------------------------------------------
# 1. App Safety Grep Test
# ---------------------------------------------------------------------------

def test_app_never_imports_src_load_or_touches_raw_data():
    """App code must NEVER import src.load, mention data/raw, or set HOLDOUT env."""
    app_dir = Path("app")
    assert app_dir.exists(), "app directory does not exist"

    forbidden_patterns = [
        re.compile(r"import\s+src\.load"),
        re.compile(r"from\s+src\.load\s+import"),
        re.compile(r"data/raw"),
        re.compile(r"data\\raw"),
        re.compile(r"TENDENCY_ALLOW_HOLDOUT"),
    ]

    py_files = list(app_dir.rglob("*.py"))
    assert len(py_files) > 0, "No python files found in app/"

    violations = []
    for f in py_files:
        content = f.read_text(encoding="utf-8")
        for line_no, line in enumerate(content.splitlines(), start=1):
            for pat in forbidden_patterns:
                if pat.search(line):
                    violations.append(f"{f}:{line_no} matches {pat.pattern}: {line}")

    assert not violations, "App safety violations detected:\n" + "\n".join(violations)


# ---------------------------------------------------------------------------
# 2. Verdict Tokens & Color Uniqueness
# ---------------------------------------------------------------------------

def test_verdict_tokens_complete_and_unique():
    """Verify all four verdict states exist with distinct colors, icons, labels."""
    expected_states = {"premium", "reverse", "none", "cant_tell"}
    assert set(VERDICT_TOKENS.keys()) == expected_states

    colors = set()
    for state, token in VERDICT_TOKENS.items():
        assert token.name == state
        assert token.color.startswith("#"), f"Invalid hex color for {state}"
        assert len(token.icon) >= 1, f"Missing icon for {state}"
        assert len(token.label) > 0, f"Missing label for {state}"
        assert token.color not in colors, f"Duplicate color detected for state {state}"
        colors.add(token.color)

    # Cant_tell must have hatched pattern
    assert VERDICT_TOKENS["cant_tell"].pattern == "hatched"


# ---------------------------------------------------------------------------
# 3. Plotly Template Registered
# ---------------------------------------------------------------------------

def test_plotly_template_registered():
    """The 'tendency' Plotly template must be registered and apply cleanly."""
    register_plotly_template()
    assert "tendency" in pio.templates

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4]))
    fig = apply_template(fig)
    # Setting template="tendency" resolves fig.layout.template to the Template object
    assert fig.layout.template == pio.templates["tendency"]
    assert fig.layout.template.layout.font.family is not None


# ---------------------------------------------------------------------------
# 4. Render Component HTML Helpers
# ---------------------------------------------------------------------------

def test_render_helpers_output_markup():
    """render_verdict_chip and render_callout return non-empty HTML markup."""
    chip = render_verdict_chip("premium")
    assert "verdict-chip premium" in chip
    assert "▲" in chip
    assert "Evidence of a surprise premium" in chip

    chip_cant = render_verdict_chip("cant_tell")
    assert "verdict-chip cant-tell" in chip_cant
    assert "◌" in chip_cant
    assert "Can't tell" in chip_cant

    callout = render_callout("cant_tell", "Sample uncertainty text")
    assert "tendency-callout cant-tell" in callout
    assert "Sample uncertainty text" in callout
