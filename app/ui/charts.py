"""Chart components and builders for Tendency Tax (Deliverables D7.2, D7.3).

Provides interactive Plotly visualizations conforming to the design system:
  - decile_gap_chart: Pass-minus-run EPA gap by decile of predicted pass probability.
  - drop_low_play_deciles: Filters out deciles with fewer than min_plays calls.
  - placebo_histogram: Histogram of shuffled-data null estimates vs observed delta.
  - check_placebo_shift: Detects whether the placebo null is shifted away from zero.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go

from app.data import format_ci, format_epa, format_int
from app.ui.theme import SERIES_COLORS, apply_template

# Threshold for flagging an uncentered placebo null:
# If |median| > NULL_SHIFT_SD_THRESHOLD * SD, a caveat warning is triggered.
NULL_SHIFT_SD_THRESHOLD: float = 0.5


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


def check_placebo_shift(
    null_deltas: pd.Series | list[float],
    threshold_factor: float = NULL_SHIFT_SD_THRESHOLD,
) -> dict[str, Any]:
    """Assess whether the placebo null distribution is substantially shifted from zero.

    Parameters
    ----------
    null_deltas : pd.Series or list[float]
        Series of null delta values from permutation test.
    threshold_factor : float
        Multiple of standard deviation to use as the shift threshold.

    Returns
    -------
    dict[str, Any]
        Dictionary with:
          - median: float
          - sd: float
          - threshold: float (threshold_factor * sd)
          - is_shifted: bool (|median| > threshold)
          - pct_2_5: float
          - pct_97_5: float
    """
    s = pd.Series(null_deltas).dropna()
    if s.empty:
        return {
            "median": 0.0,
            "sd": 0.0,
            "threshold": 0.0,
            "is_shifted": False,
            "pct_2_5": 0.0,
            "pct_97_5": 0.0,
        }

    median_val = float(s.median())
    sd_val = float(s.std(ddof=1)) if len(s) > 1 else 0.0
    threshold = threshold_factor * sd_val
    is_shifted = abs(median_val) > threshold
    pct_2_5 = float(s.quantile(0.025))
    pct_97_5 = float(s.quantile(0.975))

    return {
        "median": median_val,
        "sd": sd_val,
        "threshold": threshold,
        "is_shifted": is_shifted,
        "pct_2_5": pct_2_5,
        "pct_97_5": pct_97_5,
    }


def build_placebo_histogram(
    placebo_df: pd.DataFrame,
    observed_delta: float,
    split_name: str = "Holdout (2024–2025)",
) -> tuple[go.Figure, dict[str, Any]]:
    """Build the placebo permutation null histogram with shaded range and markers.

    Features:
      - 500 shuffled-data estimates of delta.
      - 2.5–97.5% range shaded.
      - Observed delta as a solid vertical line.
      - Null median as a dashed vertical line.
      - Zero reference line.
      - Uses registered 'tendency' Plotly template.

    Parameters
    ----------
    placebo_df : pd.DataFrame
        DataFrame with 'null_delta' column.
    observed_delta : float
        The observed point estimate of delta on the corresponding split.
    split_name : str
        Label for the split being evaluated.

    Returns
    -------
    tuple[go.Figure, dict[str, Any]]
        The Plotly Figure and the shift check summary dictionary.
    """
    if placebo_df.empty or "null_delta" not in placebo_df.columns:
        fig = go.Figure()
        return apply_template(fig), check_placebo_shift([])

    null_deltas = placebo_df["null_delta"].dropna()
    stats = check_placebo_shift(null_deltas)

    median_val = stats["median"]
    p025 = stats["pct_2_5"]
    p975 = stats["pct_97_5"]

    fig = go.Figure()

    # 1. Shaded 2.5% - 97.5% null region
    fig.add_vrect(
        x0=p025,
        x1=p975,
        fillcolor="rgba(138, 143, 152, 0.16)",
        line_width=0,
        layer="below",
    )

    # 2. Histogram of null replications
    fig.add_trace(
        go.Histogram(
            x=null_deltas,
            nbinsx=35,
            name="Shuffled-data null (500 reps)",
            marker_color="#8A8F98",
            opacity=0.75,
            hovertemplate=(
                "Null Δ bin: %{x:.3f}<br>"
                "Count: %{y}"
                "<extra></extra>"
            ),
        )
    )

    # 3. Reference line at zero
    fig.add_vline(
        x=0.0,
        line_dash="dot",
        line_color="rgba(138, 143, 152, 0.7)",
        line_width=1.5,
    )

    # 4. Null median vertical line
    fig.add_vline(
        x=median_val,
        line_dash="dash",
        line_color="#E69F00",  # Orange accent
        line_width=2.0,
        annotation_text=f"Null Median: {format_epa(median_val)}",
        annotation_position="top left",
        annotation_font=dict(size=12, color="#E69F00"),
    )

    # 5. Observed delta vertical line
    fig.add_vline(
        x=observed_delta,
        line_dash="solid",
        line_color="#0072B2",  # Okabe-Ito Blue
        line_width=2.5,
        annotation_text=f"Observed Δ: {format_epa(observed_delta)}",
        annotation_position="top right",
        annotation_font=dict(size=12, color="#0072B2"),
    )

    fig.update_layout(
        title=f"Placebo Distribution: Permuted Situation vs. Call ({split_name})",
        xaxis=dict(
            title="Estimated Delta (EPA / play)",
            zeroline=False,
        ),
        yaxis=dict(
            title="Replication Count",
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=10, r=20, t=50, b=80),
        height=400,
    )

    return apply_template(fig), stats


# ---------------------------------------------------------------------------
# Deliverable D8.3: 27-Cell Overview Grid
# ---------------------------------------------------------------------------

def compute_cells_summary(split_df: pd.DataFrame) -> dict[str, Any]:
    """Compute summary counts of testable cells and verdicts for a split.

    Parameters
    ----------
    split_df : pd.DataFrame
        DataFrame containing cells for a single split.

    Returns
    -------
    dict[str, Any]
        Dictionary with total, n_testable, n_premium, n_reverse, n_none, n_cant_tell,
        and a plain-language summary line.
    """
    total = len(split_df)
    n_testable = int(split_df["testable"].sum()) if "testable" in split_df else 0

    if "verdict" in split_df:
        verdict_series = split_df["verdict"]
    else:
        verdict_series = pd.Series(dtype=str)
    counts = verdict_series.value_counts().to_dict()

    n_premium = int(counts.get("premium", 0))
    n_reverse = int(counts.get("reverse", 0))
    n_none = int(counts.get("none", 0))
    n_cant_tell = int(counts.get("cant_tell", 0))

    summary_line = (
        f"{n_testable} of {total} cells testable • "
        f"{n_premium} premium, {n_reverse} reverse, "
        f"{n_none} none, {n_cant_tell} can't tell."
    )

    return {
        "total": total,
        "n_testable": n_testable,
        "n_premium": n_premium,
        "n_reverse": n_reverse,
        "n_none": n_none,
        "n_cant_tell": n_cant_tell,
        "summary_line": summary_line,
    }


def cell_grid_chart(
    split_df: pd.DataFrame,
    selected_cell_id: str | None = None,
) -> go.Figure:
    """Render a 9x3 interactive grid of all 27 situational cells.

    Rows: 9 combinations of down x distance bucket.
    Columns: 3 field zones (Red Zone, Midfield, Backed Up).

    Features:
      - Colored by verdict token color.
      - "cant_tell" cells display design-system hatched fill (pattern shape '/').
      - Text inside cell displays verdict icon, delta, and play count N.
      - Never relies on color alone (icon + label + numbers).
      - Selection support: customdata contains (down, dist, zone, cell_id).

    Parameters
    ----------
    split_df : pd.DataFrame
        Cells data for the selected split (27 rows).
    selected_cell_id : str | None
        Optionally highlight the currently active cell.
    """
    from app.ui.theme import VERDICT_TOKENS

    downs = [1, 2, 3]
    dists = ["short", "medium", "long"]
    dist_short_labels = {
        "short": "Short (1–3)",
        "medium": "Med (4–7)",
        "long": "Long (8+)",
    }
    zones = ["red_zone", "midfield", "backed_up"]
    zone_titles = {
        "red_zone": "Red Zone (1–20)",
        "midfield": "Midfield (21–79)",
        "backed_up": "Backed Up (80–99)",
    }

    # 9 rows ordered down 1..3 and dist short..long
    row_keys = [(d, dist) for d in downs for dist in dists]
    row_labels = [f"Down {d} • {dist_short_labels[dist]}" for d, dist in row_keys]

    fig = go.Figure()

    # Build one Bar trace per zone column so they stack horizontally into 3 columns
    for z in zones:
        bar_colors = []
        pattern_shapes = []
        pattern_fgcolors = []
        bar_texts = []
        customdata = []
        bar_line_colors = []
        bar_line_widths = []

        for d, dist in row_keys:
            c_id = f"{d}_{dist}_{z}"
            sub = split_df[split_df["cell_id"] == c_id]

            if not sub.empty:
                rec = sub.iloc[0].to_dict()
                verdict = rec.get("verdict", "cant_tell")
                testable = rec.get("testable", True)
                delta_val = rec.get("delta")
                n_plays = rec.get("n_plays", 0)
                reason = rec.get("reason", "")
            else:
                verdict = "cant_tell"
                testable = False
                delta_val = None
                n_plays = 0
                reason = "No data"

            token = VERDICT_TOKENS.get(verdict, VERDICT_TOKENS["cant_tell"])
            icon = token.icon
            color = token.color

            # Hatched fill for cant_tell
            if verdict == "cant_tell":
                pattern_shapes.append("/")
                pattern_fgcolors.append("rgba(80, 80, 80, 0.45)")
            else:
                pattern_shapes.append("")
                pattern_fgcolors.append("rgba(0, 0, 0, 0)")

            # Text inside the cell (icon + delta + plays)
            is_nan_val = isinstance(delta_val, float) and delta_val != delta_val
            if not testable:
                delta_text = "Untestable"
            elif delta_val is None or is_nan_val:
                delta_text = "—"
            else:
                sign = "+" if delta_val > 0 else ""
                delta_text = f"Δ {sign}{delta_val:.2f}"

            n_str = f"N={format_int(n_plays)}"
            cell_text = f"<b>{icon} {token.label[:4]}</b><br>{delta_text}<br>{n_str}"

            # Highlight selected cell with thicker high-contrast border
            if selected_cell_id and c_id == selected_cell_id:
                line_color = "#212529"
                line_width = 3
            else:
                line_color = "#FFFFFF"
                line_width = 1

            bar_colors.append(color)
            bar_texts.append(cell_text)
            bar_line_colors.append(line_color)
            bar_line_widths.append(line_width)

            customdata.append(
                [d, dist, z, c_id, delta_text, n_str, token.label, reason]
            )

        fig.add_trace(
            go.Bar(
                name=zone_titles[z],
                y=row_labels,
                x=[1] * len(row_labels),
                orientation="h",
                marker=dict(
                    color=bar_colors,
                    pattern=dict(
                        shape=pattern_shapes,
                        fgcolor=pattern_fgcolors,
                    ),
                    line=dict(color=bar_line_colors, width=bar_line_widths),
                ),
                text=bar_texts,
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(size=11, color="#FFFFFF"),
                customdata=customdata,
                hovertemplate=(
                    "<b>Cell: %{customdata[3]}</b><br>"
                    "State: %{customdata[6]}<br>"
                    "Estimate: %{customdata[4]}<br>"
                    "Sample: %{customdata[5]}<br>"
                    "Reason: %{customdata[7]}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        barmode="stack",
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=12),
            title="",
        ),
        xaxis=dict(
            visible=False,
            showgrid=False,
            zeroline=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=10, r=10, t=60, b=10),
        height=540,
    )

    return apply_template(fig)



