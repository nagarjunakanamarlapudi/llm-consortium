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


# Map variants to their 2×2×2 taxonomy positions (thesis §2.1–§2.2).
# Axes: Authority (centralized/decentralized), Roles (homogeneous/specialized),
#        Dynamics (cooperative/adversarial).
# v1 sits outside the matrix as the experimental control.
# Sub-variants inherit their parent's taxonomy position.
_PARENT_TAXONOMY: dict[str, dict[str, str]] = {
    "v1":  {"authority": "centralized",  "roles": "homogeneous", "dynamics": "cooperative"},
    "v2":  {"authority": "centralized",  "roles": "homogeneous", "dynamics": "cooperative"},
    "v3":  {"authority": "decentralized", "roles": "homogeneous", "dynamics": "cooperative"},
    "v4":  {"authority": "centralized",  "roles": "homogeneous", "dynamics": "adversarial"},
    "v5":  {"authority": "centralized",  "roles": "specialized", "dynamics": "cooperative"},
    "v6":  {"authority": "decentralized", "roles": "specialized", "dynamics": "cooperative"},
    "v7":  {"authority": "decentralized", "roles": "homogeneous", "dynamics": "cooperative"},
    "v8":  {"authority": "decentralized", "roles": "homogeneous", "dynamics": "adversarial"},
}


def _get_taxonomy(variant_id: str) -> dict[str, str] | None:
    """Resolve taxonomy for a variant, including sub-variants.

    Sub-variants (v1a, v2a-c, v3a-c) inherit their parent's position.
    """
    if variant_id in _PARENT_TAXONOMY:
        return _PARENT_TAXONOMY[variant_id]
    # Strip trailing letters to find parent: v2a → v2, v3c → v3
    import re
    m = re.match(r"^(v\d+)", variant_id)
    if m:
        return _PARENT_TAXONOMY.get(m.group(1))
    return None


# Public alias for backward compatibility
VARIANT_TAXONOMY: dict[str, dict[str, str]] = _PARENT_TAXONOMY


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
    for v, tax in _PARENT_TAXONOMY.items():
        levels.add(tax[axis])

    if len(levels) != 2:
        return None

    level_list = sorted(levels)

    for _, row in scores_df.iterrows():
        vid = row["variant_id"]
        tax = _get_taxonomy(vid)
        if tax is None:
            continue
        level = tax[axis]
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


# ── Blocker Analysis ────────────────────────────────────────────────────────


@dataclass
class BlockerAnalysis:
    """Blocker count analysis per variant."""

    variant_id: str
    mean_blocker_count: float
    total_blockers: int
    blocker_types: dict[str, int]
    n: int


@dataclass
class DimensionDelta:
    """Per-dimension quality delta vs baseline."""

    dimension: str
    baseline_median: float
    variant_median: float
    delta: float
    variant_id: str


@dataclass
class ConvergencePoint:
    """Convergence speed data for iterative variants."""

    variant_id: str
    task_id: str
    round_scores: list[float]
    rounds_to_plateau: int


def analyze_blockers(scores_df: pd.DataFrame) -> list[BlockerAnalysis]:
    """Analyze critical blocker counts per variant.

    Requires 'blocker_count' and optionally 'blockers_json' columns.
    """
    if scores_df.empty or "blocker_count" not in scores_df.columns:
        return []

    results: list[BlockerAnalysis] = []
    for vid, group in scores_df.groupby("variant_id"):
        blocker_counts = group["blocker_count"].fillna(0).values
        mean_count = float(np.mean(blocker_counts))
        total = int(np.sum(blocker_counts))

        blocker_types: dict[str, int] = {}
        if "blockers_json" in group.columns:
            import json

            for bj in group["blockers_json"].dropna():
                try:
                    blockers = json.loads(bj)
                    for b in blockers:
                        b_str = str(b)[:100]
                        blocker_types[b_str] = blocker_types.get(b_str, 0) + 1
                except (json.JSONDecodeError, TypeError):
                    pass

        results.append(
            BlockerAnalysis(
                variant_id=str(vid),
                mean_blocker_count=mean_count,
                total_blockers=total,
                blocker_types=blocker_types,
                n=len(group),
            )
        )

    return sorted(results, key=lambda r: r.mean_blocker_count)


# ── Per-Dimension Delta ────────────────────────────────────────────────────


def per_dimension_delta(
    scores_df: pd.DataFrame,
    baseline: str = "v1",
) -> list[DimensionDelta]:
    """Compute per-dimension quality delta of each variant vs baseline.

    Requires 'dimension_medians' column (JSON string) in scores_df.
    """
    if scores_df.empty or "dimension_medians" not in scores_df.columns:
        return []

    import json

    variant_dim_scores: dict[str, dict[str, list[float]]] = {}
    for _, row in scores_df.iterrows():
        vid = row["variant_id"]
        try:
            dm = row["dimension_medians"]
            dim_medians = json.loads(dm) if isinstance(dm, str) else dm
        except (json.JSONDecodeError, TypeError):
            continue
        variant_dim_scores.setdefault(vid, {})
        for dim_id, score in dim_medians.items():
            variant_dim_scores[vid].setdefault(dim_id, []).append(float(score))

    if baseline not in variant_dim_scores:
        return []

    baseline_dims = {
        dim: float(np.median(scores))
        for dim, scores in variant_dim_scores[baseline].items()
    }
    results: list[DimensionDelta] = []

    for vid, dims in variant_dim_scores.items():
        if vid == baseline:
            continue
        for dim_id, scores in dims.items():
            variant_median = float(np.median(scores))
            baseline_median = baseline_dims.get(dim_id, 0.0)
            results.append(
                DimensionDelta(
                    dimension=dim_id,
                    baseline_median=baseline_median,
                    variant_median=variant_median,
                    delta=variant_median - baseline_median,
                    variant_id=vid,
                )
            )

    return sorted(results, key=lambda r: abs(r.delta), reverse=True)


# ── Cohen's d ──────────────────────────────────────────────────────────────


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups.

    Uses pooled standard deviation: d = (mean_a - mean_b) / s_pooled
    """
    n_a, n_b = len(group_a), len(group_b)
    if n_a < 2 or n_b < 2:
        return 0.0

    mean_a, mean_b = float(np.mean(group_a)), float(np.mean(group_b))
    var_a = float(np.var(group_a, ddof=1))
    var_b = float(np.var(group_b, ddof=1))

    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    pooled_sd = np.sqrt(pooled_var)

    if pooled_sd == 0:
        return 0.0

    return float((mean_a - mean_b) / pooled_sd)


def pairwise_cohens_d(scores_df: pd.DataFrame) -> dict[str, float]:
    """Compute Cohen's d for all variant pairs.

    Returns a dict mapping 'vX_vs_vY' to Cohen's d value.
    """
    if scores_df.empty:
        return {}

    variant_scores: dict[str, np.ndarray] = {}
    for vid, group in scores_df.groupby("variant_id"):
        variant_scores[str(vid)] = group["overall_median"].values

    results: dict[str, float] = {}
    variants = sorted(variant_scores.keys())
    for i in range(len(variants)):
        for j in range(i + 1, len(variants)):
            d = cohens_d(variant_scores[variants[i]], variant_scores[variants[j]])
            results[f"{variants[i]}_vs_{variants[j]}"] = d

    return results


# ── Convergence Speed ──────────────────────────────────────────────────────


def analyze_convergence_speed(db: object) -> list[ConvergencePoint]:
    """Analyze convergence speed of iterative variants.

    Queries per-round designs from the database and estimates quality
    trajectory using token count as a proxy for design completeness.
    """
    results: list[ConvergencePoint] = []

    try:
        rows = db.conn.execute(  # type: ignore[union-attr]
            """SELECT r.variant_id, r.task_id, d.round, d.token_count
               FROM designs d
               JOIN runs r ON d.run_id = r.run_id
               WHERE r.status = 'completed'
               ORDER BY r.variant_id, r.task_id, d.round"""
        ).fetchall()
    except Exception:
        return results

    from collections import defaultdict

    groups: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for row in rows:
        key = (row["variant_id"], row["task_id"])
        groups[key].append((row["round"], row["token_count"] or 0))

    for (vid, tid), rounds_data in groups.items():
        if len(rounds_data) < 2:
            continue

        round_scores = [float(tc) for _, tc in sorted(rounds_data)]

        # Find plateau: round where improvement < 5% of total range
        if max(round_scores) == min(round_scores):
            plateau = 0
        else:
            total_range = max(round_scores) - min(round_scores)
            plateau = len(round_scores) - 1
            for i in range(1, len(round_scores)):
                improvement = abs(round_scores[i] - round_scores[i - 1])
                if improvement < 0.05 * total_range:
                    plateau = i
                    break

        results.append(
            ConvergencePoint(
                variant_id=vid,
                task_id=tid,
                round_scores=round_scores,
                rounds_to_plateau=plateau,
            )
        )

    return results


# ── Stratified Pairwise Comparisons (§7.5 Layer 3) ───────────────────────


def pairwise_by_complexity(
    scores_df: pd.DataFrame,
    variant_a: str,
    variant_b: str,
) -> dict[str, PairwiseResult | None]:
    """Wilcoxon signed-rank test between two variants, stratified by complexity.

    Returns one PairwiseResult per complexity level (simple, medium, complex).
    This implements thesis §7.5 Layer 3.
    """
    if scores_df.empty or "complexity" not in scores_df.columns:
        return {}

    results: dict[str, PairwiseResult | None] = {}
    for complexity, group in scores_df.groupby("complexity"):
        pivot = group.pivot_table(
            index=["task_id", "repetition"],
            columns="variant_id",
            values="overall_median",
            aggfunc="first",
        ).dropna()

        if variant_a not in pivot.columns or variant_b not in pivot.columns:
            results[str(complexity)] = None
            continue

        a = pivot[variant_a].values
        b = pivot[variant_b].values

        if len(a) < 3:
            results[str(complexity)] = None
            continue

        try:
            stat, p_val = scipy_stats.wilcoxon(a, b, alternative="two-sided")
        except ValueError:
            stat, p_val = 0.0, 1.0

        n = len(a)
        effect_size = 1 - (2 * stat) / (n * (n + 1)) if n > 0 else 0.0
        d = cohens_d(a, b)

        results[str(complexity)] = PairwiseResult(
            variant_a=variant_a,
            variant_b=variant_b,
            statistic=stat,
            p_value=p_val,
            p_adjusted=p_val,  # no correction within single comparison
            significant=p_val < 0.05,
            effect_size=effect_size,
        )

    return results


def compare_to_baseline_by_complexity(
    scores_df: pd.DataFrame,
    baseline: str = "v1_baseline",
) -> dict[str, list[PairwiseResult]]:
    """Compare all variants to baseline, stratified by complexity level.

    Returns dict mapping complexity → list of PairwiseResult.
    """
    if scores_df.empty or "complexity" not in scores_df.columns:
        return {}

    results: dict[str, list[PairwiseResult]] = {}
    for complexity, group in scores_df.groupby("complexity"):
        results[str(complexity)] = compare_to_baseline(group, baseline=baseline)

    return results


# ── Targeted Thesis Comparisons (§7.4) ────────────────────────────────────


@dataclass
class ThesisComparison:
    """A targeted pairwise comparison from thesis §7.4."""

    comparison_id: str
    variant_a: str
    variant_b: str
    variable_isolated: str
    hypothesis: str
    overall: PairwiseResult | None = None
    by_complexity: dict[str, PairwiseResult | None] = field(default_factory=dict)
    cohens_d_value: float = 0.0


# The 8 targeted comparisons from thesis Table 7.4.
# Updated to use actual experiment variant IDs.
THESIS_COMPARISONS: list[dict[str, str]] = [
    {
        "id": "C1",
        "a": "v1_baseline",
        "b": "v2a_same_model",
        "variable": "Feedback source: rubric scores vs. qualitative critique (same model)",
        "hypothesis": "Qualitative critique provides more actionable revision guidance",
    },
    {
        "id": "C2",
        "a": "v2a_same_model",
        "b": "v2b_cross_model",
        "variable": "Reviewer identity: same-model vs. cross-model",
        "hypothesis": "Different training distribution surfaces different blind spots",
    },
    {
        "id": "C3",
        "a": "v2b_cross_model",
        "b": "v4_adversarial",
        "variable": "Feedback dynamics: cooperative vs. adversarial (same topology)",
        "hypothesis": "Adversarial pressure produces more robust designs",
    },
    {
        "id": "C4",
        "a": "v2b_cross_model",
        "b": "v5_specialist_panel",
        "variable": "Reviewer specialization: general vs. domain-expert",
        "hypothesis": "Specialists catch deeper flaws in their domain",
    },
    {
        "id": "C5",
        "a": "v3b_rubric_merge",
        "b": "v7_consensus",
        "variable": "Synthesis mechanism: central merger vs. peer convergence",
        "hypothesis": "Central synthesis → higher peak; peer convergence → more consistency",
    },
    {
        "id": "C6",
        "a": "v3c_dialectical_merge",
        "b": "v8_structured_debate",
        "variable": "Diversity source: random vs. forced opposition",
        "hypothesis": "Forced disagreement explores solution space more thoroughly",
    },
    {
        "id": "C7",
        "a": "v4_adversarial",
        "b": "v8_structured_debate",
        "variable": "Adversarial structure: critique-only vs. build+critique",
        "hypothesis": "Constructive adversarial > destructive adversarial",
    },
    {
        "id": "C8",
        "a": "v5_specialist_panel",
        "b": "v6_rotating_leader",
        "variable": "Specialization target: reviewer expertise vs. leader expertise",
        "hypothesis": "Specialized leaders > specialized reviewers",
    },
]


def targeted_thesis_comparisons(
    scores_df: pd.DataFrame,
) -> list[ThesisComparison]:
    """Run the 8 targeted pairwise comparisons from thesis §7.4.

    For each comparison, computes:
    - Overall Wilcoxon signed-rank test
    - Per-complexity stratification (§7.5 Layer 3)
    - Cohen's d effect size
    """
    if scores_df.empty:
        return []

    results: list[ThesisComparison] = []
    n_comparisons = len(THESIS_COMPARISONS)

    for spec in THESIS_COMPARISONS:
        tc = ThesisComparison(
            comparison_id=spec["id"],
            variant_a=spec["a"],
            variant_b=spec["b"],
            variable_isolated=spec["variable"],
            hypothesis=spec["hypothesis"],
        )

        # Overall test
        pivot = scores_df.pivot_table(
            index=["task_id", "repetition"],
            columns="variant_id",
            values="overall_median",
            aggfunc="first",
        ).dropna()

        if spec["a"] in pivot.columns and spec["b"] in pivot.columns:
            a = pivot[spec["a"]].values
            b = pivot[spec["b"]].values

            try:
                stat, p_val = scipy_stats.wilcoxon(a, b, alternative="two-sided")
            except ValueError:
                stat, p_val = 0.0, 1.0

            p_adj = min(p_val * n_comparisons, 1.0)
            n = len(a)
            effect_size = 1 - (2 * stat) / (n * (n + 1)) if n > 0 else 0.0

            tc.overall = PairwiseResult(
                variant_a=spec["a"],
                variant_b=spec["b"],
                statistic=stat,
                p_value=p_val,
                p_adjusted=p_adj,
                significant=p_adj < 0.05,
                effect_size=effect_size,
            )
            tc.cohens_d_value = cohens_d(a, b)

        # Per-complexity
        if "complexity" in scores_df.columns:
            tc.by_complexity = pairwise_by_complexity(
                scores_df, spec["a"], spec["b"],
            )

        results.append(tc)

    return results


# ── Descriptive Statistics Table (§8.2 Stage 1) ──────────────────────────


@dataclass
class DescriptiveStats:
    """Descriptive statistics for a variant at a given complexity level."""

    variant_id: str
    complexity: str  # "simple", "medium", "complex", or "all"
    n: int
    mean: float
    std: float
    median: float
    min: float
    max: float
    q25: float
    q75: float
    mean_cost_usd: float = 0.0
    mean_tokens: int = 0


def descriptive_statistics(
    scores_df: pd.DataFrame,
    costs_df: pd.DataFrame | None = None,
) -> list[DescriptiveStats]:
    """Compute descriptive statistics per variant and complexity level.

    Produces rows for each (variant, complexity) combination plus
    an "all" row aggregating across complexity levels.
    Implements thesis §8.2 Stage 1.
    """
    if scores_df.empty:
        return []

    # Merge costs if available
    merged = scores_df.copy()
    if costs_df is not None and not costs_df.empty:
        cost_cols = ["run_id", "total_cost_usd", "total_input_tokens", "total_output_tokens"]
        available = [c for c in cost_cols if c in costs_df.columns]
        if "run_id" in available:
            merged = merged.merge(costs_df[available], on="run_id", how="left")

    results: list[DescriptiveStats] = []

    def _compute(group: pd.DataFrame, variant_id: str, complexity: str) -> DescriptiveStats:
        scores = group["overall_median"].values
        cost = 0.0
        tokens = 0
        if "total_cost_usd" in group.columns:
            cost = float(group["total_cost_usd"].mean()) if not group["total_cost_usd"].isna().all() else 0.0
        if "total_input_tokens" in group.columns and "total_output_tokens" in group.columns:
            total_tok = group["total_input_tokens"].fillna(0) + group["total_output_tokens"].fillna(0)
            tokens = int(total_tok.mean())
        return DescriptiveStats(
            variant_id=variant_id,
            complexity=complexity,
            n=len(scores),
            mean=float(np.mean(scores)),
            std=float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
            median=float(np.median(scores)),
            min=float(np.min(scores)),
            max=float(np.max(scores)),
            q25=float(np.percentile(scores, 25)),
            q75=float(np.percentile(scores, 75)),
            mean_cost_usd=cost,
            mean_tokens=tokens,
        )

    # Per variant, overall
    for vid, group in merged.groupby("variant_id"):
        results.append(_compute(group, str(vid), "all"))

    # Per variant × complexity
    if "complexity" in merged.columns:
        for (vid, complexity), group in merged.groupby(["variant_id", "complexity"]):
            results.append(_compute(group, str(vid), str(complexity)))

    return sorted(results, key=lambda r: (r.complexity, r.variant_id))


def export_descriptive_table(
    stats: list[DescriptiveStats],
    output_dir: str | Path,
) -> None:
    """Export descriptive statistics as CSV.

    Produces: {output_dir}/tables/descriptive_statistics.csv
    """
    from pathlib import Path as _Path

    output_dir = _Path(output_dir)
    csv_path = output_dir / "tables" / "descriptive_statistics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for s in stats:
        rows.append({
            "variant_id": s.variant_id,
            "complexity": s.complexity,
            "n": s.n,
            "mean": round(s.mean, 3),
            "std": round(s.std, 3),
            "median": round(s.median, 3),
            "min": round(s.min, 3),
            "max": round(s.max, 3),
            "q25": round(s.q25, 3),
            "q75": round(s.q75, 3),
            "mean_cost_usd": round(s.mean_cost_usd, 4),
            "mean_tokens": s.mean_tokens,
        })

    if rows:
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        logger.info("descriptive_statistics.exported", path=str(csv_path))


# ── Break-Even Analysis (§8.2 Stage 3) ──────────────────────────────────


@dataclass
class BreakEvenResult:
    """Break-even analysis for a variant vs baseline."""

    variant_id: str
    baseline_id: str
    quality_delta: float  # mean(variant) - mean(baseline)
    cost_delta: float  # mean(variant_cost) - mean(baseline_cost)
    cost_multiplier: float  # variant_cost / baseline_cost
    quality_per_extra_dollar: float  # quality_delta / cost_delta
    justified: bool  # Is the quality improvement worth the cost?


def break_even_analysis(
    scores_df: pd.DataFrame,
    costs_df: pd.DataFrame,
    baseline: str = "v1_baseline",
    min_quality_delta: float = 0.1,
) -> list[BreakEvenResult]:
    """Compute break-even point for each variant vs baseline.

    A variant is "justified" if:
    1. Quality improvement > min_quality_delta, AND
    2. Quality per extra dollar > 0 (positive return)

    Implements thesis §8.2 Stage 3.
    """
    if scores_df.empty or costs_df.empty:
        return []

    merged = scores_df.merge(
        costs_df[["run_id", "total_cost_usd"]],
        on="run_id",
        how="inner",
    )

    variant_stats: dict[str, tuple[float, float]] = {}
    for vid, group in merged.groupby("variant_id"):
        variant_stats[str(vid)] = (
            float(group["overall_median"].mean()),
            float(group["total_cost_usd"].mean()),
        )

    if baseline not in variant_stats:
        return []

    base_quality, base_cost = variant_stats[baseline]
    results: list[BreakEvenResult] = []

    for vid, (quality, cost) in sorted(variant_stats.items()):
        if vid == baseline:
            continue

        q_delta = quality - base_quality
        c_delta = cost - base_cost
        c_mult = cost / base_cost if base_cost > 0 else float("inf")
        qped = q_delta / c_delta if c_delta > 0 else float("inf")
        justified = q_delta > min_quality_delta and c_delta >= 0 and qped > 0

        results.append(BreakEvenResult(
            variant_id=vid,
            baseline_id=baseline,
            quality_delta=q_delta,
            cost_delta=c_delta,
            cost_multiplier=c_mult,
            quality_per_extra_dollar=qped,
            justified=justified,
        ))

    return sorted(results, key=lambda r: r.quality_delta, reverse=True)


# ── Budget Simulation (§8.2 Stage 3) ────────────────────────────────────


@dataclass
class BudgetSimResult:
    """Result of a fixed-budget variant allocation simulation."""

    budget_usd: float
    variant_id: str
    runs_affordable: int
    expected_mean_quality: float
    expected_min_quality: float  # worst-case (q25)


def budget_simulation(
    scores_df: pd.DataFrame,
    costs_df: pd.DataFrame,
    budgets: list[float] | None = None,
) -> list[BudgetSimResult]:
    """Simulate: given a fixed budget, which variant maximizes quality?

    For each budget level, computes how many runs of each variant
    are affordable and the expected quality from those runs.
    Implements thesis §8.2 Stage 3.
    """
    if scores_df.empty or costs_df.empty:
        return []

    if budgets is None:
        budgets = [1.0, 5.0, 10.0, 25.0, 50.0, 100.0]

    merged = scores_df.merge(
        costs_df[["run_id", "total_cost_usd"]],
        on="run_id",
        how="inner",
    )

    results: list[BudgetSimResult] = []
    for vid, group in merged.groupby("variant_id"):
        mean_cost = float(group["total_cost_usd"].mean())
        mean_quality = float(group["overall_median"].mean())
        q25 = float(np.percentile(group["overall_median"].values, 25))

        for budget in budgets:
            runs = int(budget / mean_cost) if mean_cost > 0 else 0
            if runs < 1:
                continue
            results.append(BudgetSimResult(
                budget_usd=budget,
                variant_id=str(vid),
                runs_affordable=runs,
                expected_mean_quality=mean_quality,
                expected_min_quality=q25,
            ))

    return sorted(results, key=lambda r: (r.budget_usd, -r.expected_mean_quality))


# ── Human Validation Subset (§5.4) ──────────────────────────────────────


def select_human_validation_subset(
    scores_df: pd.DataFrame,
    fraction: float = 0.10,
    seed: int = 42,
) -> pd.DataFrame:
    """Select a stratified random subset of designs for human validation.

    Stratified by variant_id to ensure every variant is represented.
    Returns a DataFrame with columns: design_id, variant_id, task_id,
    repetition, overall_median (LLM score for later comparison).
    Implements thesis §5.4.
    """
    if scores_df.empty:
        return pd.DataFrame()

    rng = np.random.RandomState(seed)
    subset_rows = []

    for vid, group in scores_df.groupby("variant_id"):
        n_select = max(1, int(len(group) * fraction))
        indices = rng.choice(len(group), size=min(n_select, len(group)), replace=False)
        subset_rows.append(group.iloc[indices])

    if not subset_rows:
        return pd.DataFrame()

    subset = pd.concat(subset_rows, ignore_index=True)
    cols = ["design_id", "variant_id", "task_id", "repetition", "overall_median"]
    available = [c for c in cols if c in subset.columns]
    return subset[available].sort_values(["variant_id", "task_id"])


def compute_human_llm_correlation(
    human_scores: pd.DataFrame,
    llm_scores: pd.DataFrame,
) -> dict[str, float]:
    """Compute Spearman ρ between human and LLM scores.

    Both DataFrames must have 'design_id' and 'overall_score' columns.
    Returns dict with 'spearman_r', 'p_value', and 'n'.
    Implements thesis §5.4.
    """
    merged = human_scores.merge(
        llm_scores[["design_id", "overall_median"]],
        on="design_id",
        how="inner",
        suffixes=("_human", "_llm"),
    )

    if len(merged) < _MIN_OBS:
        return {"spearman_r": 0.0, "p_value": 1.0, "n": len(merged)}

    r, p = scipy_stats.spearmanr(
        merged["overall_score"].values,
        merged["overall_median"].values,
    )
    return {"spearman_r": float(r), "p_value": float(p), "n": len(merged)}


# ── Evaluator Reliability Summary (§5.4) ────────────────────────────────


@dataclass
class EvaluatorReliability:
    """Aggregate evaluator reliability statistics."""

    n_designs: int
    mean_krippendorff_alpha: float
    median_krippendorff_alpha: float
    pct_alpha_above_07: float  # % of designs with α > 0.7
    mean_disagreement_rate: float  # fraction of designs with any disagreement flag
    intra_rater_std: float  # mean std across evaluator runs per design


def evaluator_reliability_summary(
    scores_df: pd.DataFrame,
    db: object | None = None,
) -> EvaluatorReliability | None:
    """Compute aggregate evaluator reliability metrics.

    Uses Krippendorff's alpha from scores_median and optionally
    queries individual evaluator runs for intra-rater variance.
    Implements thesis §5.4.
    """
    if scores_df.empty:
        return None

    n = len(scores_df)

    # Krippendorff's alpha stats
    alpha_col = "krippendorff_alpha"
    if alpha_col in scores_df.columns:
        alphas = scores_df[alpha_col].dropna().values
        mean_alpha = float(np.mean(alphas)) if len(alphas) > 0 else 0.0
        median_alpha = float(np.median(alphas)) if len(alphas) > 0 else 0.0
        pct_above = float(np.mean(alphas > 0.7) * 100) if len(alphas) > 0 else 0.0
    else:
        mean_alpha = median_alpha = pct_above = 0.0

    # Disagreement rate
    if "disagreement_flags" in scores_df.columns:
        has_disagreement = scores_df["disagreement_flags"].apply(
            lambda x: bool(x) and x != "[]" and x != "{}"
        )
        disagreement_rate = float(has_disagreement.mean())
    else:
        disagreement_rate = 0.0

    # Intra-rater variance from evaluations table
    intra_std = 0.0
    if db is not None:
        try:
            rows = db.conn.execute(
                """SELECT design_id, overall_score
                   FROM evaluations
                   ORDER BY design_id"""
            ).fetchall()
            from collections import defaultdict
            by_design: dict[str, list[float]] = defaultdict(list)
            for row in rows:
                by_design[row["design_id"]].append(float(row["overall_score"]))
            stds = [float(np.std(scores)) for scores in by_design.values() if len(scores) >= 2]
            intra_std = float(np.mean(stds)) if stds else 0.0
        except Exception:
            pass

    return EvaluatorReliability(
        n_designs=n,
        mean_krippendorff_alpha=mean_alpha,
        median_krippendorff_alpha=median_alpha,
        pct_alpha_above_07=pct_above,
        mean_disagreement_rate=disagreement_rate,
        intra_rater_std=intra_std,
    )


# ── Case Study Export (§8.2 Stage 4) ────────────────────────────────────


@dataclass
class CaseStudyDesign:
    """A design identified for manual case study analysis."""

    design_id: str
    variant_id: str
    task_id: str
    repetition: int
    overall_median: float
    rank_label: str  # "top" or "bottom"


def select_case_study_designs(
    scores_df: pd.DataFrame,
    n_per_group: int = 3,
) -> list[CaseStudyDesign]:
    """Select top and bottom designs per (variant, task) for case study.

    Returns the n_per_group highest and lowest scoring designs overall,
    plus the best and worst per variant for manual inspection.
    Implements thesis §8.2 Stage 4.
    """
    if scores_df.empty:
        return []

    results: list[CaseStudyDesign] = []
    seen: set[str] = set()

    # Global top and bottom
    sorted_df = scores_df.sort_values("overall_median", ascending=False)

    for _, row in sorted_df.head(n_per_group).iterrows():
        did = row["design_id"]
        if did not in seen:
            seen.add(did)
            results.append(CaseStudyDesign(
                design_id=did,
                variant_id=row["variant_id"],
                task_id=row["task_id"],
                repetition=int(row.get("repetition", 0)),
                overall_median=float(row["overall_median"]),
                rank_label="top",
            ))

    for _, row in sorted_df.tail(n_per_group).iterrows():
        did = row["design_id"]
        if did not in seen:
            seen.add(did)
            results.append(CaseStudyDesign(
                design_id=did,
                variant_id=row["variant_id"],
                task_id=row["task_id"],
                repetition=int(row.get("repetition", 0)),
                overall_median=float(row["overall_median"]),
                rank_label="bottom",
            ))

    # Per variant: best and worst
    for vid, group in scores_df.groupby("variant_id"):
        sorted_group = group.sort_values("overall_median", ascending=False)
        for _, row in sorted_group.head(1).iterrows():
            did = row["design_id"]
            if did not in seen:
                seen.add(did)
                results.append(CaseStudyDesign(
                    design_id=did,
                    variant_id=row["variant_id"],
                    task_id=row["task_id"],
                    repetition=int(row.get("repetition", 0)),
                    overall_median=float(row["overall_median"]),
                    rank_label="top",
                ))
        for _, row in sorted_group.tail(1).iterrows():
            did = row["design_id"]
            if did not in seen:
                seen.add(did)
                results.append(CaseStudyDesign(
                    design_id=did,
                    variant_id=row["variant_id"],
                    task_id=row["task_id"],
                    repetition=int(row.get("repetition", 0)),
                    overall_median=float(row["overall_median"]),
                    rank_label="bottom",
                ))

    return sorted(results, key=lambda r: -r.overall_median)


# ── Novelty / Diversity Metric (§8.1) ────────────────────────────────────────


@dataclass
class NoveltyDiversityResult:
    """Per-variant diversity of final designs across repetitions."""

    variant_id: str
    task_id: str
    n_designs: int
    mean_pairwise_cosine: float  # mean pairwise cosine similarity (1.0 = identical)
    diversity_score: float  # 1 - mean_pairwise_cosine (higher = more diverse)


def analyze_novelty_diversity(
    designs: list[dict],
    *,
    model_name: str = "all-MiniLM-L6-v2",
) -> list[NoveltyDiversityResult]:
    """Compute architectural diversity across repetitions via embedding similarity.

    For each (variant, task) group with ≥2 designs, computes pairwise cosine
    similarity of section-level embeddings.  Returns one result per group.

    Args:
        designs: list of dicts with keys ``variant_id``, ``task_id``, ``full_text``.
        model_name: sentence-transformers model to use (default lightweight).

    Returns:
        List of :class:`NoveltyDiversityResult`.  Empty if sentence-transformers
        is not installed (graceful degradation).
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
    except ImportError:
        logger.warning(
            "novelty_diversity.skipped",
            reason="sentence-transformers not installed; pip install sentence-transformers",
        )
        return []

    if not designs:
        return []

    # Group designs by (variant_id, task_id)
    from collections import defaultdict

    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for d in designs:
        vid = d.get("variant_id", "")
        tid = d.get("task_id", "")
        text = d.get("full_text", "")
        if vid and tid and text:
            groups[(vid, tid)].append(text)

    # Only analyse groups with ≥2 designs
    groups = {k: v for k, v in groups.items() if len(v) >= 2}
    if not groups:
        return []

    # Collect all texts, encode once
    all_texts: list[str] = []
    group_indices: dict[tuple[str, str], list[int]] = {}
    for key, texts in groups.items():
        start = len(all_texts)
        all_texts.extend(texts)
        group_indices[key] = list(range(start, start + len(texts)))

    logger.info("novelty_diversity.encoding", n_texts=len(all_texts), model=model_name)
    model = SentenceTransformer(model_name)
    embeddings = model.encode(all_texts, show_progress_bar=False, convert_to_numpy=True)

    # Normalise for cosine similarity via dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # guard against zero vectors
    embeddings = embeddings / norms

    results: list[NoveltyDiversityResult] = []
    for (vid, tid), idxs in group_indices.items():
        embs = embeddings[idxs]
        n = len(idxs)
        # Pairwise cosine similarities (upper triangle)
        sim_matrix = embs @ embs.T
        upper_mask = np.triu_indices(n, k=1)
        pairwise_sims = sim_matrix[upper_mask]
        mean_cos = float(np.mean(pairwise_sims))
        results.append(NoveltyDiversityResult(
            variant_id=vid,
            task_id=tid,
            n_designs=n,
            mean_pairwise_cosine=round(mean_cos, 4),
            diversity_score=round(1.0 - mean_cos, 4),
        ))

    return sorted(results, key=lambda r: (r.variant_id, r.task_id))
