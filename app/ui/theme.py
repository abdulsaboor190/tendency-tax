"""Design system tokens and Plotly theme for Tendency Tax (Deliverable D6.1).

Design Rules:
  - Verdict first, evidence beneath.
  - "Can't tell" is a first-class neutral state, never an error.
  - Never use color alone: every state has an icon plus a text label.
  - Always show uncertainty and sample size n.
  - Plain, association-only language.
  - Works in light and dark, and on phone width.
"""

from dataclasses import dataclass
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ---------------------------------------------------------------------------
# 1. Okabe-Ito Color Tokens & Verdict States
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VerdictToken:
    name: str
    color: str
    tint: str
    icon: str
    label: str
    pattern: str = "solid"  # 'solid' or 'hatched'


VERDICT_TOKENS: dict[str, VerdictToken] = {
    "premium": VerdictToken(
        name="premium",
        color="#0072B2",  # Okabe-Ito Blue
        tint="rgba(0, 114, 178, 0.12)",
        icon="▲",
        label="Evidence of a surprise premium",
    ),
    "reverse": VerdictToken(
        name="reverse",
        color="#D55E00",  # Okabe-Ito Vermillion
        tint="rgba(213, 94, 0, 0.12)",
        icon="▼",
        label="Evidence that predictability pays",
    ),
    "none": VerdictToken(
        name="none",
        color="#009E73",  # Okabe-Ito Bluish Green
        tint="rgba(0, 158, 115, 0.12)",
        icon="＝",
        label="Evidence of no meaningful gap",
    ),
    "cant_tell": VerdictToken(
        name="cant_tell",
        color="#8A8F98",  # Neutral Grey
        tint="rgba(138, 143, 152, 0.14)",
        icon="◌",
        label="Can't tell",
        pattern="hatched",
    ),
}

# Series Tokens
SERIES_COLORS: dict[str, str] = {
    "early": "#56B4E9",    # Sky Blue
    "tune": "#E69F00",     # Orange
    "holdout": "#0072B2",  # Blue
    "all": "#333333",      # Charcoal / Neutral
}

# General UI Tokens
UI_TOKENS: dict[str, str] = {
    "surface": "#FFFFFF",
    "surface_subtle": "#F8F9FA",
    "text": "#212529",
    "text_muted": "#6C757D",
    "border": "#E9ECEF",
    "border_subtle": "#DEE2E6",
}

FONT_FAMILY: str = (
    "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
    "Helvetica, Arial, sans-serif"
)


# ---------------------------------------------------------------------------
# 2. CSS Injection
# ---------------------------------------------------------------------------

def inject_css() -> None:
    """Inject custom design system CSS into the active Streamlit app."""
    css_path = Path(__file__).parent / "style.css"
    if css_path.exists():
        css_text = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css_text}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 3. HTML Component Helpers
# ---------------------------------------------------------------------------

def render_verdict_chip(state: str, custom_text: str | None = None) -> str:
    """Return HTML string for a colored verdict chip with icon + label."""
    token = VERDICT_TOKENS.get(state, VERDICT_TOKENS["cant_tell"])
    label_text = custom_text if custom_text else token.label
    css_class = token.name.replace("_", "-")
    return (
        f'<span class="verdict-chip {css_class}">'
        f'<span class="verdict-icon">{token.icon}</span>'
        f"<span>{label_text}</span>"
        f"</span>"
    )


def render_callout(
    state: str,
    text: str,
    title: str | None = None,
) -> str:
    """Return HTML string for an uncertainty-first callout box."""
    token = VERDICT_TOKENS.get(state, VERDICT_TOKENS["cant_tell"])
    css_class = token.name.replace("_", "-")
    title_html = (
        f'<div style="font-weight: 600; margin-bottom: 4px;">'
        f"{token.icon} {title}</div>"
        if title
        else f'<div style="font-weight: 600; margin-bottom: 4px;">'
        f"{token.icon} {token.label}</div>"
    )
    return (
        f'<div class="tendency-callout {css_class}">'
        f"{title_html}"
        f"<div>{text}</div>"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# 4. Plotly Template ("tendency")
# ---------------------------------------------------------------------------

def register_plotly_template() -> None:
    """Register the 'tendency' Plotly template with accessible colors and font."""
    template = go.layout.Template()

    # Layout styling
    template.layout = go.Layout(
        font=dict(
            family=FONT_FAMILY,
            size=13,
            color="#212529",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=[
            VERDICT_TOKENS["premium"].color,
            VERDICT_TOKENS["reverse"].color,
            VERDICT_TOKENS["none"].color,
            VERDICT_TOKENS["cant_tell"].color,
            SERIES_COLORS["early"],
            SERIES_COLORS["tune"],
            SERIES_COLORS["holdout"],
        ],
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            font_size=12,
            font_family=FONT_FAMILY,
            bordercolor="#E9ECEF",
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(0, 0, 0, 0.08)",
            gridwidth=1,
            zeroline=True,
            zerolinecolor="rgba(0, 0, 0, 0.25)",
            zerolinewidth=1.5,
            linecolor="#DEE2E6",
            ticks="outside",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(0, 0, 0, 0.08)",
            gridwidth=1,
            zeroline=True,
            zerolinecolor="rgba(0, 0, 0, 0.25)",
            zerolinewidth=1.5,
            linecolor="#DEE2E6",
            ticks="outside",
        ),
        margin=dict(l=40, r=20, t=40, b=40),
    )

    pio.templates["tendency"] = template


def apply_template(fig: go.Figure) -> go.Figure:
    """Apply the 'tendency' design system template to a Plotly figure."""
    if "tendency" not in pio.templates:
        register_plotly_template()
    fig.update_layout(template="tendency")
    return fig


# Automatically register template on import
register_plotly_template()
