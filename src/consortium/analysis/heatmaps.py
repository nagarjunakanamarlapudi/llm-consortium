"""Heatmap visualizations for experiment analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
import structlog

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

logger = structlog.get_logger()


def _save_fig(fig: plt.Figure, path: Path) -> None:
    """Save matplotlib figure and close."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("heatmap.saved", path=str(path))


def variant_dimension_heatmap(
    scores_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Heatmap: variant (rows) x rubric dimension (columns).

    Expects scores_df to have 'variant_id' and 'dimension_medians' (JSON string).
    """
    import json

    if scores_df.empty:
        return

    # Explode dimension_medians JSON into separate rows
    records: list[dict] = []
    for _, row in scores_df.iterrows():
        try:
            dims = (
                json.loads(row["dimension_medians"])
                if isinstance(row["dimension_medians"], str)
                else row["dimension_medians"]
            )
            if isinstance(dims, dict):
                for dim_name, score in dims.items():
                    records.append(
                        {
                            "variant_id": row["variant_id"],
                            "dimension": dim_name,
                            "score": float(score),
                        }
                    )
        except (json.JSONDecodeError, TypeError):
            continue

    if not records:
        logger.warning("heatmap.no_dimension_data")
        return

    dim_df = pd.DataFrame(records)
    pivot = dim_df.pivot_table(
        index="variant_id",
        columns="dimension",
        values="score",
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 1.5), max(6, len(pivot) * 0.8)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
        vmin=1,
        vmax=10,
    )
    ax.set_title("Mean Score by Variant x Rubric Dimension")
    ax.set_ylabel("Variant")
    ax.set_xlabel("Dimension")

    _save_fig(fig, output_path)


def complexity_variant_heatmap(
    scores_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Heatmap: task complexity (rows) x variant (columns).

    Requires 'complexity' column in scores_df.
    """
    if scores_df.empty or "complexity" not in scores_df.columns:
        logger.warning("heatmap.no_complexity_data")
        return

    pivot = scores_df.pivot_table(
        index="complexity",
        columns="variant_id",
        values="overall_median",
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 1.5), max(4, len(pivot) * 1.2)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Mean Score by Complexity x Variant")
    ax.set_ylabel("Complexity")
    ax.set_xlabel("Variant")

    _save_fig(fig, output_path)


def task_variant_heatmap(
    scores_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Heatmap: task (rows) x variant (columns)."""
    if scores_df.empty:
        return

    pivot = scores_df.pivot_table(
        index="task_id",
        columns="variant_id",
        values="overall_median",
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 1.5), max(6, len(pivot) * 0.8)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Mean Score by Task x Variant")
    ax.set_ylabel("Task")
    ax.set_xlabel("Variant")

    _save_fig(fig, output_path)


def coherence_variant_heatmap(
    coherence_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Heatmap: variant (rows) x task (columns) showing coherence rate.

    Color scale: green (1.0 = fully consistent) → red (0.0 = all contradictions).
    """
    if coherence_df.empty:
        return

    pivot = coherence_df.pivot_table(
        index="variant_id",
        columns="task_id",
        values="coherence_rate",
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 1.5), max(6, len(pivot) * 0.8)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        linewidths=0.5,
        ax=ax,
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_title("Mean Coherence Rate by Variant x Task")
    ax.set_ylabel("Variant")
    ax.set_xlabel("Task")

    _save_fig(fig, output_path)
