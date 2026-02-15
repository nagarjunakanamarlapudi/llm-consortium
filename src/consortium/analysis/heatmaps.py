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
        cmap="RdYlGn",
        linewidths=0.5,
        ax=ax,
        vmin=1,
        vmax=5,
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
        cmap="RdYlGn",
        linewidths=0.5,
        ax=ax,
        vmin=1,
        vmax=5,
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
        vmin=1,
        vmax=5,
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


# ── Box Plots (§8.2 Stage 1) ──────────────────────────────────────────────


def variant_boxplot(
    scores_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Box plot of quality scores per variant, sorted by median.

    Shows distribution of overall_median across all tasks and repetitions.
    Implements thesis §8.2 Stage 1.
    """
    if scores_df.empty:
        return

    # Sort variants by median score
    medians = scores_df.groupby("variant_id")["overall_median"].median().sort_values(ascending=False)
    order = list(medians.index)

    fig, ax = plt.subplots(figsize=(max(10, len(order) * 0.8), 6))
    sns.boxplot(
        data=scores_df,
        x="variant_id",
        y="overall_median",
        order=order,
        palette="Set2",
        ax=ax,
    )
    sns.stripplot(
        data=scores_df,
        x="variant_id",
        y="overall_median",
        order=order,
        color="0.3",
        size=3,
        alpha=0.5,
        ax=ax,
    )
    ax.set_title("Quality Score Distribution by Variant")
    ax.set_xlabel("Variant")
    ax.set_ylabel("Overall Median Score (1–5)")
    ax.set_ylim(0.5, 5.5)
    plt.xticks(rotation=45, ha="right")

    _save_fig(fig, output_path)


def variant_boxplot_by_complexity(
    scores_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Box plots of quality scores per variant, faceted by task complexity.

    Produces a multi-panel figure with one subplot per complexity level.
    Implements thesis §8.2 Stage 1.
    """
    if scores_df.empty or "complexity" not in scores_df.columns:
        logger.warning("boxplot.no_complexity_data")
        return

    complexities = sorted(scores_df["complexity"].unique())
    n_panels = len(complexities)
    if n_panels == 0:
        return

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(max(6, n_panels * 5), 6),
        sharey=True,
    )
    if n_panels == 1:
        axes = [axes]

    for ax, complexity in zip(axes, complexities):
        subset = scores_df[scores_df["complexity"] == complexity]
        medians = subset.groupby("variant_id")["overall_median"].median().sort_values(ascending=False)
        order = list(medians.index)

        sns.boxplot(
            data=subset,
            x="variant_id",
            y="overall_median",
            order=order,
            palette="Set2",
            ax=ax,
        )
        ax.set_title(f"{complexity.title()} Tasks")
        ax.set_xlabel("Variant")
        ax.set_ylabel("Score" if ax == axes[0] else "")
        ax.set_ylim(0.5, 5.5)
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle("Quality Score Distribution by Variant and Complexity", fontsize=14, y=1.02)
    fig.tight_layout()

    _save_fig(fig, output_path)


def cost_quality_scatter(
    scores_df: pd.DataFrame,
    costs_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Scatter plot: mean quality vs mean cost per variant.

    Each point is a variant; includes error bars for IQR.
    """
    if scores_df.empty or costs_df.empty:
        return

    merged = scores_df.merge(
        costs_df[["run_id", "total_cost_usd"]],
        on="run_id",
        how="inner",
    )

    stats = merged.groupby("variant_id").agg(
        mean_quality=("overall_median", "mean"),
        mean_cost=("total_cost_usd", "mean"),
        q25=("overall_median", lambda x: x.quantile(0.25)),
        q75=("overall_median", lambda x: x.quantile(0.75)),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.errorbar(
        stats["mean_cost"],
        stats["mean_quality"],
        yerr=[
            stats["mean_quality"] - stats["q25"],
            stats["q75"] - stats["mean_quality"],
        ],
        fmt="o",
        capsize=4,
        markersize=8,
        color="steelblue",
    )
    for _, row in stats.iterrows():
        ax.annotate(
            row["variant_id"],
            (row["mean_cost"], row["mean_quality"]),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=9,
        )

    ax.set_xlabel("Mean Cost (USD)")
    ax.set_ylabel("Mean Quality Score (1–5)")
    ax.set_title("Cost vs Quality by Variant")
    ax.grid(True, alpha=0.3)

    _save_fig(fig, output_path)
