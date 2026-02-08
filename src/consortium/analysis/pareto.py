"""Pareto frontier analysis — quality vs. cost trade-off."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger()


def compute_pareto_frontier(
    df: pd.DataFrame,
    quality_col: str = "mean_score",
    cost_col: str = "mean_cost",
) -> pd.DataFrame:
    """Identify Pareto-optimal rows (maximize quality, minimize cost).

    Args:
        df: DataFrame with at least quality_col and cost_col.
        quality_col: Column to maximize.
        cost_col: Column to minimize.

    Returns:
        Subset of df that lies on the Pareto frontier.
    """
    if df.empty:
        return df

    # Sort by cost ascending
    sorted_df = df.sort_values(cost_col).reset_index(drop=True)

    pareto_mask = np.zeros(len(sorted_df), dtype=bool)
    max_quality = -np.inf

    for i in range(len(sorted_df)):
        q = sorted_df.loc[i, quality_col]
        if q > max_quality:
            pareto_mask[i] = True
            max_quality = q

    return sorted_df[pareto_mask].copy()


def plot_pareto(
    df: pd.DataFrame,
    output_dir: Path,
    quality_col: str = "mean_score",
    cost_col: str = "mean_cost",
    label_col: str = "variant_id",
) -> None:
    """Create Pareto frontier plot: interactive Plotly HTML + static PNG.

    Args:
        df: DataFrame with quality_col, cost_col, and label_col.
        output_dir: Directory to write pareto_frontier.html and .png.
        quality_col: Column for Y axis (quality to maximize).
        cost_col: Column for X axis (cost to minimize).
        label_col: Column for point labels.
    """
    import plotly.graph_objects as go

    if df.empty:
        logger.warning("pareto.empty_data")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    frontier = compute_pareto_frontier(df, quality_col, cost_col)

    fig = go.Figure()

    # All points
    fig.add_trace(
        go.Scatter(
            x=df[cost_col],
            y=df[quality_col],
            mode="markers+text",
            text=df[label_col],
            textposition="top center",
            marker=dict(size=12, color="steelblue", opacity=0.7),
            name="All Variants",
        )
    )

    # Pareto frontier line
    frontier_sorted = frontier.sort_values(cost_col)
    fig.add_trace(
        go.Scatter(
            x=frontier_sorted[cost_col],
            y=frontier_sorted[quality_col],
            mode="lines+markers",
            marker=dict(size=14, color="red", symbol="star"),
            line=dict(color="red", dash="dash"),
            text=frontier_sorted[label_col],
            name="Pareto Frontier",
        )
    )

    fig.update_layout(
        title="Quality vs. Cost — Pareto Frontier",
        xaxis_title="Mean Cost per Run ($)",
        yaxis_title="Mean Overall Score",
        template="plotly_white",
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
        font=dict(size=14),
    )

    # Save interactive HTML
    html_path = output_dir / "pareto_frontier.html"
    fig.write_html(str(html_path))
    logger.info("pareto.saved_html", path=str(html_path))

    # Save static PNG (requires kaleido)
    try:
        png_path = output_dir / "pareto_frontier.png"
        fig.write_image(str(png_path), width=1200, height=800, scale=2)
        logger.info("pareto.saved_png", path=str(png_path))
    except Exception:
        logger.warning("pareto.png_failed", hint="Install kaleido for PNG export")
