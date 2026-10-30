"""Chart components and builders for Tendency Tax (Deliverable D7.2).

Provides interactive Plotly visualizations conforming to the design system:
  - decile_gap_chart: Pass-minus-run EPA gap by decile of predicted pass probability.
  - drop_low_play_deciles: Filters out deciles with fewer than min_plays calls.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go

from app.data import format_ci, format_int
from app.ui.theme import SERIES_COLORS, apply_template


def drop_low_play_deciles(
    df: pd.DataFrame,
    min_plays: int = 200,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Filter out rows where either n_pass or n_run is below min_plays.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns 'split', 'decile', 'n_pass', 'n_run'.
    min_plays : int
        Minimum play threshold for each play call type (pass and run).

    Returns
    -------
    tuple[pd.DataFrame, list[dict[str, Any]]]
        Filtered DataFrame, and a list of records for the dropped deciles:
        [{'split': str, 'decile': int, 'n_pass': int, 'n_run': int, 'reason': str}, ...]
    """
    if df.empty:
        return df.copy(), []

    dropped_records: list[dict[str, Any]] = []
    keep_indices: list[Any] = []

    for idx, row in df.iterrows():
        n_pass = int(row.get("n_pass", 0))
        n_run = int(row.get("n_run", 0))
        split = str(row.get("split", "unknown"))
        decile = int(row.get("decile", 0))

        if n_pass < min_plays or n_run < min_plays:
            reasons = []
            if n_pass < min_plays:
                reasons.append(f"n_pass={format_int(n_pass)} < {min_plays}")
            if n_run < min_plays:
                reasons.append(f"n_run={format_int(n_run)} < {min_plays}")
            reason_str = ", ".join(reasons)

            dropped_records.append({
                "split": split,
                "decile": decile,
                "n_pass": n_pass,
                "n_run": n_run,
                "reason": reason_str,
            })
        else:
            keep_indices.append(idx)

    filtered_df = df.loc[keep_indices].copy()
    return filtered_df, dropped_records


def build_decile_gap_chart(
    early_df: pd.DataFrame,
    holdout_df: pd.DataFrame | None = None,
    tune_df: pd.DataFrame | None = None,
    include_tune: bool = False,
    min_plays: int = 200,
) -> tuple[go.Figure, list[dict[str, Any]]]:
    """Build the central decile gap chart with bootstrap uncertainty bands.

    Plots the pass-minus-run EPA gap by decile of predicted pass probability (phat)
    with the bootstrap bands for early and holdout (and optionally tune).

    Parameters
    ----------
    early_df : pd.DataFrame
        Decile gap DataFrame for the early split (2016–2021).
    holdout_df : pd.DataFrame | None
        Decile gap DataFrame for the holdout split (2024–2025).
    tune_df : pd.DataFrame | None
        Decile gap DataFrame for the tune split (2022–2023).
    include_tune : bool
        Whether to display the tune split series.
    min_plays : int
        Minimum plays per call type (default 200).

    Returns
    -------
    tuple[go.Figure, list[dict[str, Any]]]
        The configured Plotly Figure and list of all dropped decile records.
    """
    all_dropped: list[dict[str, Any]] = []
    fig = go.Figure()

    # Series definitions in order of rendering
    series_configs: list[tuple[str, str, pd.DataFrame | None]] = [
        ("early", "Early (2016–2021)", early_df),
    ]
    if include_tune and tune_df is not None:
        series_configs.append(("tune", "Tune (2022–2023)", tune_df))
    if holdout_df is not None:
        series_configs.append(("holdout", "Holdout (2024–2025)", holdout_df))

    # Helper for converting hex color to rgba with opacity
    def _hex_to_rgba(hex_color: str, alpha: float) -> str:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r}, {g}, {b}, {alpha})"

    for split_key, series_label, raw_df in series_configs:
        if raw_df is None or raw_df.empty:
            continue

        clean_df, dropped = drop_low_play_deciles(raw_df, min_plays=min_plays)
        all_dropped.extend(dropped)

        if clean_df.empty:
            continue

        clean_df = clean_df.sort_values("decile")

        x_vals = clean_df["decile"].tolist()
        y_vals = clean_df["epa_gap_pass_minus_run"].tolist()
        ci_lo_vals = clean_df["ci_95_low"].tolist()
        ci_hi_vals = clean_df["ci_95_high"].tolist()

        color = SERIES_COLORS.get(split_key, "#8A8F98")
        fill_color = _hex_to_rgba(color, 0.16)

        # Custom hover data: decile range, gap, band, n pass, n run
        customdata = []
        for _, row in clean_df.iterrows():
            d_min = float(row.get("phat_min", 0.0))
            d_max = float(row.get("phat_max", 0.0))
            gap = float(row.get("epa_gap_pass_minus_run", 0.0))
            lo = float(row.get("ci_95_low", 0.0))
            hi = float(row.get("ci_95_high", 0.0))
            np_ = int(row.get("n_pass", 0))
            nr = int(row.get("n_run", 0))

            phat_range = f"[{d_min:.3f}, {d_max:.3f}]"
            ci_str = format_ci(lo, hi)
            gap_sign = "+" if gap > 0 else ""
            gap_str = f"{gap_sign}{gap:.3f}"

            customdata.append([
                phat_range,
                gap_str,
                ci_str,
                format_int(np_),
                format_int(nr),
            ])

        # 1. Shaded CI band: Upper bound + Lower bound closed polygon
        # Trace for upper bound
        fig.add_trace(
            go.Scatter(
                x=x_vals + x_vals[::-1],
                y=ci_hi_vals + ci_lo_vals[::-1],
                fill="toself",
                fillcolor=fill_color,
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
                name=f"{series_label} 95% CI",
            )
        )

        # 2. Main point-and-line trace
        hovertemplate = (
            f"<b>{series_label}</b><br>"
            "Decile %{x} (p&#770; range: %{customdata[0]})<br>"
            "Pass &minus; Run EPA Gap: <b>%{customdata[1]}</b><br>"
            "95% Bootstrap Band: %{customdata[2]}<br>"
            "Pass Plays: %{customdata[3]} | Run Plays: %{customdata[4]}"
            "<extra></extra>"
        )

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines+markers",
                name=series_label,
                line=dict(color=color, width=2.5),
                marker=dict(size=8, color=color),
                customdata=customdata,
                hovertemplate=hovertemplate,
            )
        )

    # Dashed zero reference line
    fig.add_hline(
        y=0.0,
        line_dash="dash",
        line_color="rgba(138, 143, 152, 0.7)",
        line_width=1.5,
    )

    # Layout configuration
    fig.update_layout(
        title="Pass &minus; Run EPA Gap by Predicted Pass Probability Decile",
        xaxis=dict(
            title="Decile of Predicted Pass Probability (p&#770;)",
            tickmode="linear",
            tick0=1,
            dtick=1,
            range=[0.5, 10.5],
        ),
        yaxis=dict(
            title="Pass &minus; Run EPA Gap (EPA/play)",
            zeroline=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
        ),
        hovermode="closest",
        margin=dict(l=10, r=20, t=50, b=80),
        height=450,
    )

    return apply_template(fig), all_dropped
