"""Statistical analysis for LLM Consortium experiment results.

Implements: Friedman + Nemenyi, Wilcoxon signed-rank, Mann-Whitney U,
coefficient of variation, and token efficiency analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import structlog
from scipy import stats as scipy_stats

logger = structlog.get_logger()

# Minimum number of observations for reliable nonparametric tests.
_MIN_OBS = 5


@dataclass
class RankingResult:
    """Result of a variant ranking analysis."""

    variant_ids: list[str]
    mean_ranks: list[float]
    mean_scores: list[float]
    statistic: float
    p_value: float
    test_name: str
    significant: bool
    posthoc: dict[str, float] = field(default_factory=dict)


@dataclass
class PairwiseResult:
    """Result of a pairwise comparison."""

    variant_a: str
    variant_b: str
    statistic: float
    p_value: float
    p_adjusted: float
    significant: bool
    effect_size: float  # rank-biserial r or similar


@dataclass
class AxisEffect:
    """Mann-Whitney test result for the 2x2x2 axis analysis."""

    axis_name: str  # authority, roles, dynamics
    level_a: str
    level_b: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: float
    mean_a: float
    mean_b: float


@dataclass
class VarianceResult:
    """Coefficient of variation analysis per variant."""

    variant_id: str
    mean_score: float
    std_score: float
    cv: float  # coefficient of variation
    n: int


@dataclass
class EfficiencyResult:
    """Quality per token/cost analysis."""

    variant_id: str
    mean_score: float
    mean_cost: float
    mean_tokens: int
    quality_per_dollar: float  # score / cost
    quality_per_1k_tokens: float  # score / (tokens/1000)


# ── Ranking ──────────────────────────────────────────────────────────────────


def rank_variants_overall(scores_df: pd.DataFrame) -> RankingResult | None:
    """Rank variants using Friedman test + Nemenyi post-hoc.

    Each (task, repetition) is treated as a "block".
    """
    if scores_df.empty:
        return None

    pivot = scores_df.pivot_table(
        index=["task_id", "repetition"],
        columns="variant_id",
        values="overall_median",
        aggfunc="first",
    ).dropna()

    variants = list(pivot.columns)
    if len(variants) < 2:
        return None

    if len(pivot) < _MIN_OBS:
        logger.warning(
            "statistics.insufficient_blocks",
            blocks=len(pivot),
            minimum=_MIN_OBS,
        )

    # Friedman test (requires >= 3 variants); fall back to Wilcoxon for 2
    groups = [pivot[v].values for v in variants]
    if len(variants) >= 3:
        stat, p_val = scipy_stats.friedmanchisquare(*groups)
        test_name = "Friedman"
    else:
        try:
            stat, p_val = scipy_stats.wilcoxon(
                groups[0],
                groups[1],
                alternative="two-sided",
            )
        except ValueError:
            stat, p_val = 0.0, 1.0
        test_name = "Wilcoxon"

    # Compute mean ranks
    ranks_matrix = pivot.rank(axis=1, ascending=False)
    mean_ranks = [ranks_matrix[v].mean() for v in variants]
    mean_scores = [pivot[v].mean() for v in variants]

    # Sort by mean rank (1 = best)
    order = sorted(range(len(variants)), key=lambda i: mean_ranks[i])
    variants = [variants[i] for i in order]
    mean_ranks = [mean_ranks[i] for i in order]
    mean_scores = [mean_scores[i] for i in order]

    # Post-hoc Nemenyi (approximate using pairwise Wilcoxon with Bonferroni)
    posthoc: dict[str, float] = {}
    if p_val < 0.05 and len(variants) >= 3:
        n_comparisons = len(variants) * (len(variants) - 1) // 2
        for i in range(len(variants)):
            for j in range(i + 1, len(variants)):
                try:
                    _, pw = scipy_stats.wilcoxon(
                        pivot[variants[i]].values,
                        pivot[variants[j]].values,
                        alternative="two-sided",
                    )
                    pw_adj = min(pw * n_comparisons, 1.0)
                    posthoc[f"{variants[i]}_vs_{variants[j]}"] = pw_adj
                except ValueError:
                    posthoc[f"{variants[i]}_vs_{variants[j]}"] = 1.0

    return RankingResult(
        variant_ids=variants,
        mean_ranks=mean_ranks,
        mean_scores=mean_scores,
        statistic=stat,
        p_value=p_val,
        test_name=test_name,
        significant=p_val < 0.05,
        posthoc=posthoc,
    )


def rank_variants_by_complexity(scores_df: pd.DataFrame) -> dict[str, RankingResult | None]:
    """Run Friedman ranking stratified by task complexity.

    Requires a 'complexity' column. Groups tasks by complexity level
    and runs rank_variants_overall on each.
    """
    if "complexity" not in scores_df.columns:
        logger.warning("statistics.no_complexity_column")
        return {}

    results: dict[str, RankingResult | None] = {}
    for complexity, group in scores_df.groupby("complexity"):
        results[str(complexity)] = rank_variants_overall(group)

    return results


# ── Pairwise Comparisons ────────────────────────────────────────────────────


def compare_to_baseline(scores_df: pd.DataFrame, baseline: str = "v1") -> list[PairwiseResult]:
    """Compare all variants to baseline using Wilcoxon + Bonferroni (m=7)."""
    if scores_df.empty:
        return []

    pivot = scores_df.pivot_table(
        index=["task_id", "repetition"],
        columns="variant_id",
        values="overall_median",
        aggfunc="first",
    ).dropna()

    variants = [v for v in pivot.columns if v != baseline]
    if baseline not in pivot.columns or not variants:
        return []

    n_comparisons = len(variants)  # m=7 for v2-v8
    results: list[PairwiseResult] = []

    for v in sorted(variants):
        a = pivot[baseline].values
        b = pivot[v].values

        try:
            stat, p_val = scipy_stats.wilcoxon(a, b, alternative="two-sided")
        except ValueError:
            stat, p_val = 0.0, 1.0

        p_adj = min(p_val * n_comparisons, 1.0)

        # Effect size: rank-biserial correlation
        n = len(a)
        effect_size = 1 - (2 * stat) / (n * (n + 1)) if n > 0 else 0.0

        results.append(
            PairwiseResult(
                variant_a=baseline,
                variant_b=v,
                statistic=stat,
                p_value=p_val,
                p_adjusted=p_adj,
                significant=p_adj < 0.05,
                effect_size=effect_size,
            )
        )

    return results


def pairwise_comparisons(scores_df: pd.DataFrame) -> list[PairwiseResult]:
    """All-pairs Wilcoxon signed-rank + Bonferroni (m=28 for 8 variants)."""
    if scores_df.empty:
        return []

    pivot = scores_df.pivot_table(
        index=["task_id", "repetition"],
        columns="variant_id",
        values="overall_median",
        aggfunc="first",
    ).dropna()

    variants = sorted(pivot.columns)
    n_comparisons = len(variants) * (len(variants) - 1) // 2
    results: list[PairwiseResult] = []

    for i in range(len(variants)):
        for j in range(i + 1, len(variants)):
            a = pivot[variants[i]].values
            b = pivot[variants[j]].values

            try:
                stat, p_val = scipy_stats.wilcoxon(a, b, alternative="two-sided")
            except ValueError:
                stat, p_val = 0.0, 1.0

            p_adj = min(p_val * n_comparisons, 1.0)
            n = len(a)
            effect_size = 1 - (2 * stat) / (n * (n + 1)) if n > 0 else 0.0

            results.append(
                PairwiseResult(
                    variant_a=variants[i],
                    variant_b=variants[j],
                    statistic=stat,
                    p_value=p_val,
                    p_adjusted=p_adj,
                    significant=p_adj < 0.05,
                    effect_size=effect_size,
                )
            )

    return results


# ── Axis Analysis (2x2x2) ───────────────────────────────────────────────────


# Map variants to their taxonomy positions (from thesis Table 1)
VARIANT_TAXONOMY: dict[str, dict[str, str]] = {
    "v1": {"authority": "flat", "roles": "homogeneous", "dynamics": "sequential"},
    "v2": {"authority": "flat", "roles": "homogeneous", "dynamics": "iterative"},
    "v3": {"authority": "flat", "roles": "heterogeneous", "dynamics": "sequential"},
    "v4": {"authority": "flat", "roles": "heterogeneous", "dynamics": "iterative"},
    "v5": {"authority": "hierarchical", "roles": "heterogeneous", "dynamics": "sequential"},
    "v6": {"authority": "hierarchical", "roles": "heterogeneous", "dynamics": "iterative"},
    "v7": {"authority": "hierarchical", "roles": "homogeneous", "dynamics": "iterative"},
    "v8": {"authority": "hierarchical", "roles": "heterogeneous", "dynamics": "iterative"},
}


def analyze_axis_effect(
    scores_df: pd.DataFrame,
    axis: str,
) -> AxisEffect | None:
    """Mann-Whitney U test for a single axis of the 2x2x2 taxonomy.

    axis: 'authority', 'roles', or 'dynamics'
    """
    if scores_df.empty or axis not in ("authority", "roles", "dynamics"):
        return None

    # Classify each variant's scores by axis level
    level_a_scores: list[float] = []
    level_b_scores: list[float] = []

    levels = set()
    for v, tax in VARIANT_TAXONOMY.items():
        levels.add(tax[axis])

    if len(levels) != 2:
        return None

    level_list = sorted(levels)

    for _, row in scores_df.iterrows():
        vid = row["variant_id"]
        if vid not in VARIANT_TAXONOMY:
            continue
        level = VARIANT_TAXONOMY[vid][axis]
        if level == level_list[0]:
            level_a_scores.append(row["overall_median"])
        else:
            level_b_scores.append(row["overall_median"])

    if len(level_a_scores) < _MIN_OBS or len(level_b_scores) < _MIN_OBS:
        logger.warning(
            "statistics.axis_insufficient_data",
            axis=axis,
            n_a=len(level_a_scores),
            n_b=len(level_b_scores),
        )

    if not level_a_scores or not level_b_scores:
        return None

    a = np.array(level_a_scores)
    b = np.array(level_b_scores)

    stat, p_val = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
    # Effect size: rank-biserial r
    n_a, n_b = len(a), len(b)
    effect_size = 1 - (2 * stat) / (n_a * n_b) if n_a * n_b > 0 else 0.0

    return AxisEffect(
        axis_name=axis,
        level_a=level_list[0],
        level_b=level_list[1],
        statistic=stat,
        p_value=p_val,
        significant=p_val < 0.05,
        effect_size=effect_size,
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
    )


# ── Variance Analysis ────────────────────────────────────────────────────────


def variance_analysis(scores_df: pd.DataFrame) -> list[VarianceResult]:
    """Compute coefficient of variation per variant."""
    if scores_df.empty:
        return []

    results: list[VarianceResult] = []
    for vid, group in scores_df.groupby("variant_id"):
        scores = group["overall_median"].values
        mean = float(np.mean(scores))
        std = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
        cv = std / mean if mean > 0 else 0.0

        results.append(
            VarianceResult(
                variant_id=str(vid),
                mean_score=mean,
                std_score=std,
                cv=cv,
                n=len(scores),
            )
        )

    return sorted(results, key=lambda r: r.cv)


# ── Token Efficiency ─────────────────────────────────────────────────────────


def token_efficiency(
    scores_df: pd.DataFrame,
    costs_df: pd.DataFrame,
) -> list[EfficiencyResult]:
    """Compute quality per dollar and quality per 1K tokens."""
    if scores_df.empty or costs_df.empty:
        return []

    # Merge scores with costs by run_id
    merged = scores_df.merge(
        costs_df[["run_id", "total_cost_usd", "total_input_tokens", "total_output_tokens"]],
        on="run_id",
        how="inner",
    )

    results: list[EfficiencyResult] = []
    for vid, group in merged.groupby("variant_id"):
        mean_score = float(group["overall_median"].mean())
        mean_cost = float(group["total_cost_usd"].mean())
        total_tokens = group["total_input_tokens"].fillna(0) + group["total_output_tokens"].fillna(
            0
        )
        mean_tokens = int(total_tokens.mean())

        qpd = mean_score / mean_cost if mean_cost > 0 else 0.0
        qpt = mean_score / (mean_tokens / 1000) if mean_tokens > 0 else 0.0

        results.append(
            EfficiencyResult(
                variant_id=str(vid),
                mean_score=mean_score,
                mean_cost=mean_cost,
                mean_tokens=mean_tokens,
                quality_per_dollar=qpd,
                quality_per_1k_tokens=qpt,
            )
        )

    return sorted(results, key=lambda r: r.quality_per_dollar, reverse=True)


# ── Coherence Analysis ───────────────────────────────────────────────────────


@dataclass
class CoherenceCorrelation:
    """Spearman correlation between quality scores and coherence rates."""

    spearman_r: float
    p_value: float
    n_designs: int
    significant: bool


def rank_variants_by_coherence(coherence_df: pd.DataFrame) -> RankingResult | None:
    """Rank variants by coherence rate using Friedman test + Nemenyi post-hoc.

    Same approach as rank_variants_overall() but uses coherence_rate
    instead of overall_median. Each (task, repetition) is a block.
    """
    if coherence_df.empty:
        return None

    pivot = coherence_df.pivot_table(
        index=["task_id", "repetition"],
        columns="variant_id",
        values="coherence_rate",
        aggfunc="first",
    ).dropna()

    variants = list(pivot.columns)
    if len(variants) < 2:
        return None

    if len(pivot) < _MIN_OBS:
        logger.warning(
            "statistics.coherence_insufficient_blocks",
            blocks=len(pivot),
            minimum=_MIN_OBS,
        )

    groups = [pivot[v].values for v in variants]
    if len(variants) >= 3:
        stat, p_val = scipy_stats.friedmanchisquare(*groups)
        test_name = "Friedman (coherence)"
    else:
        try:
            stat, p_val = scipy_stats.wilcoxon(
                groups[0],
                groups[1],
                alternative="two-sided",
            )
        except ValueError:
            stat, p_val = 0.0, 1.0
        test_name = "Wilcoxon (coherence)"

    ranks_matrix = pivot.rank(axis=1, ascending=False)
    mean_ranks = [ranks_matrix[v].mean() for v in variants]
    mean_scores = [pivot[v].mean() for v in variants]

    order = sorted(range(len(variants)), key=lambda i: mean_ranks[i])
    variants = [variants[i] for i in order]
    mean_ranks = [mean_ranks[i] for i in order]
    mean_scores = [mean_scores[i] for i in order]

    posthoc: dict[str, float] = {}
    if p_val < 0.05 and len(variants) >= 3:
        n_comparisons = len(variants) * (len(variants) - 1) // 2
        for i in range(len(variants)):
            for j in range(i + 1, len(variants)):
                try:
                    _, pw = scipy_stats.wilcoxon(
                        pivot[variants[i]].values,
                        pivot[variants[j]].values,
                        alternative="two-sided",
                    )
                    pw_adj = min(pw * n_comparisons, 1.0)
                    posthoc[f"{variants[i]}_vs_{variants[j]}"] = pw_adj
                except ValueError:
                    posthoc[f"{variants[i]}_vs_{variants[j]}"] = 1.0

    return RankingResult(
        variant_ids=variants,
        mean_ranks=mean_ranks,
        mean_scores=mean_scores,
        statistic=stat,
        p_value=p_val,
        test_name=test_name,
        significant=p_val < 0.05,
        posthoc=posthoc,
    )


def compare_coherence_to_baseline(
    coherence_df: pd.DataFrame,
    baseline: str = "v1",
) -> list[PairwiseResult]:
    """Compare coherence rates of all variants to baseline.

    Wilcoxon signed-rank + Bonferroni (m=7).
    """
    if coherence_df.empty:
        return []

    pivot = coherence_df.pivot_table(
        index=["task_id", "repetition"],
        columns="variant_id",
        values="coherence_rate",
        aggfunc="first",
    ).dropna()

    variants = [v for v in pivot.columns if v != baseline]
    if baseline not in pivot.columns or not variants:
        return []

    n_comparisons = len(variants)
    results: list[PairwiseResult] = []

    for v in sorted(variants):
        a = pivot[baseline].values
        b = pivot[v].values

        try:
            stat, p_val = scipy_stats.wilcoxon(a, b, alternative="two-sided")
        except ValueError:
            stat, p_val = 0.0, 1.0

        p_adj = min(p_val * n_comparisons, 1.0)
        n = len(a)
        effect_size = 1 - (2 * stat) / (n * (n + 1)) if n > 0 else 0.0

        results.append(
            PairwiseResult(
                variant_a=baseline,
                variant_b=v,
                statistic=stat,
                p_value=p_val,
                p_adjusted=p_adj,
                significant=p_adj < 0.05,
                effect_size=effect_size,
            )
        )

    return results


def coherence_by_complexity(
    coherence_df: pd.DataFrame,
    task_complexity_map: dict[str, str] | None = None,
) -> dict[str, float]:
    """Compute mean coherence rate grouped by task complexity.

    Args:
        coherence_df: DataFrame from load_coherence_dataframe().
        task_complexity_map: Mapping of task_id -> complexity level.
            If None, tries to use a 'complexity' column on the DataFrame.

    Returns:
        Dict mapping complexity level to mean coherence rate.
    """
    if coherence_df.empty:
        return {}

    df = coherence_df.copy()

    if task_complexity_map:
        df["complexity"] = df["task_id"].map(task_complexity_map)
    elif "complexity" not in df.columns:
        logger.warning("statistics.coherence_no_complexity")
        return {}

    result: dict[str, float] = {}
    for complexity, group in df.groupby("complexity"):
        result[str(complexity)] = float(group["coherence_rate"].mean())

    return result


def coherence_quality_correlation(
    scores_df: pd.DataFrame,
    coherence_df: pd.DataFrame,
) -> CoherenceCorrelation | None:
    """Spearman rank correlation between quality and coherence.

    Merges on design_id and computes the correlation between
    overall_median (quality) and coherence_rate.
    """
    if scores_df.empty or coherence_df.empty:
        return None

    merged = scores_df[["design_id", "overall_median"]].merge(
        coherence_df[["design_id", "coherence_rate"]],
        on="design_id",
        how="inner",
    )

    if len(merged) < _MIN_OBS:
        logger.warning(
            "statistics.coherence_correlation_insufficient",
            n=len(merged),
        )
        return None

    r, p_val = scipy_stats.spearmanr(
        merged["overall_median"].values,
        merged["coherence_rate"].values,
    )

    return CoherenceCorrelation(
        spearman_r=float(r),
        p_value=float(p_val),
        n_designs=len(merged),
        significant=p_val < 0.05,
    )
