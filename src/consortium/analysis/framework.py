"""Decision framework generator and prediction validator (§10.1, §10.2)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
from scipy import stats as scipy_stats

from consortium.analysis.pareto import compute_pareto_frontier
from consortium.analysis.statistics import (
    CoherenceCorrelation,
    coherence_quality_correlation,
    variance_analysis,
)

logger = structlog.get_logger()


# ── Data Models ──────────────────────────────────────────────────────────────


@dataclass
class FrameworkRow:
    complexity: str  # simple | medium | complex
    budget_priority: str  # low | moderate | any
    quality_priority: str  # good | high | maximum | robust
    recommended_variant: str
    expected_quality_iqr: tuple[float, float]  # 25th-75th percentile
    expected_cost_iqr: tuple[float, float]
    confidence: str  # high | medium | low
    evidence: str  # "Based on N runs, Wilcoxon p=X vs baseline"
    expected_coherence_rate: float | None = None
    coherence_warning: str | None = None


@dataclass
class DecisionFramework:
    rows: list[FrameworkRow] = field(default_factory=list)
    pareto_variants: list[str] = field(default_factory=list)
    generation_date: str = ""
    n_runs_analyzed: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class PredictionResult:
    prediction_id: str  # P1..P6
    description: str
    supported: bool
    p_value: float | None = None
    evidence: str = ""


# ── Prediction definitions ───────────────────────────────────────────────────

PREDICTIONS = {
    "P1": "v1_baseline matches consortium quality on simple tasks",
    "P2": "v3 sub-variants achieve highest peak quality on complex tasks",
    "P3": "v7_consensus has lowest variance across repetitions",
    "P4": "v4_adversarial has highest variance across repetitions",
    "P5": "v5_specialist_panel outperforms v2 sub-variants on specialist dimensions",
    "P6": "Rubric awareness matters more than topology",
    "P7": "Consortium variants have higher coherence than single-LLM baseline",
    "P8": "Quality and coherence are positively correlated (r > 0.3)",
}

# Helper to match variant IDs flexibly (v1_baseline, v1a_single_shot, etc.)
_BASELINE_IDS = {"v1_baseline", "v1"}
_V2_IDS = {"v2a_same_model", "v2b_cross_model", "v2c_multi_model", "v2_leader_reviewers", "v2"}
_V3_IDS = {"v3a_naive_merge", "v3b_rubric_merge", "v3c_dialectical_merge", "v3_parallel_merge", "v3"}
_V4_IDS = {"v4_adversarial", "v4"}
_V5_IDS = {"v5_specialist_panel", "v5"}
_V7_IDS = {"v7_consensus", "v7"}

# Coherence rate threshold below which a warning is issued
_COHERENCE_WARNING_THRESHOLD = 0.8


# ── Framework Generation ─────────────────────────────────────────────────────


def generate_decision_framework(
    scores_df: pd.DataFrame,
    costs_df: pd.DataFrame,
    output_dir: Path,
    coherence_df: pd.DataFrame | None = None,
) -> DecisionFramework:
    """Generate the decision framework (thesis §10.2).

    For each complexity level:
      1. Compute mean quality and cost per variant
      2. Identify Pareto-optimal variants
      3. Assign budget and quality priority buckets
      4. Compute IQR from repetition data
      5. Assess confidence via Wilcoxon test
    """
    fw = DecisionFramework(
        generation_date=datetime.now().isoformat(),
        n_runs_analyzed=len(scores_df),
    )

    if scores_df.empty or costs_df.empty:
        fw.warnings.append("Insufficient data to generate framework")
        return fw

    # Merge scores and costs
    merged = scores_df.merge(
        costs_df[["run_id", "total_cost_usd"]],
        on="run_id",
        how="inner",
    )

    # Pre-compute per-variant coherence rates if available
    coherence_by_variant: dict[str, float] = {}
    if coherence_df is not None and not coherence_df.empty:
        for vid, group in coherence_df.groupby("variant_id"):
            coherence_by_variant[str(vid)] = float(group["coherence_rate"].mean())

    complexity_levels = ["simple", "medium", "complex"]
    if "complexity" not in merged.columns:
        # If there's no complexity column, treat as one group
        complexity_levels = ["all"]
        merged["complexity"] = "all"

    for complexity in complexity_levels:
        group = merged[merged["complexity"] == complexity] if complexity != "all" else merged
        if group.empty:
            continue

        # Per-variant stats
        variant_stats = (
            group.groupby("variant_id")
            .agg(
                mean_score=("overall_median", "mean"),
                mean_cost=("total_cost_usd", "mean"),
                q25=("overall_median", lambda x: x.quantile(0.25)),
                q75=("overall_median", lambda x: x.quantile(0.75)),
                cost_q25=("total_cost_usd", lambda x: x.quantile(0.25)),
                cost_q75=("total_cost_usd", lambda x: x.quantile(0.75)),
                cv=(
                    "overall_median",
                    lambda x: x.std() / x.mean() if len(x) > 1 and x.mean() > 0 else np.nan,
                ),
                n=("overall_median", "count"),
            )
            .reset_index()
        )

        if variant_stats.empty:
            continue

        # Fill NaN CVs with 0 (single-sample case: no variance measurable)
        variant_stats["cv"] = variant_stats["cv"].fillna(0.0)

        if variant_stats["n"].max() < 2:
            fw.warnings.append(
                f"Only 1 run per variant for '{complexity}' — IQR and CV are unreliable."
            )

        # Pareto frontier
        pareto = compute_pareto_frontier(variant_stats, "mean_score", "mean_cost")
        pareto_ids = list(pareto["variant_id"])
        fw.pareto_variants = list(set(fw.pareto_variants) | set(pareto_ids))

        cheapest_pareto = pareto.loc[pareto["mean_cost"].idxmin()]
        best_quality = variant_stats.loc[variant_stats["mean_score"].idxmax()]
        lowest_cv = variant_stats.loc[variant_stats["cv"].idxmin()]

        # Budget rows
        budget_entries = [
            ("low", cheapest_pareto),
            ("any", best_quality),
        ]

        # Moderate: best quality where cost < 2x cheapest
        moderate_mask = variant_stats["mean_cost"] < 2 * cheapest_pareto["mean_cost"]
        if moderate_mask.any():
            moderate_df = variant_stats[moderate_mask]
            moderate_best = moderate_df.loc[moderate_df["mean_score"].idxmax()]
            budget_entries.insert(1, ("moderate", moderate_best))

        for budget_priority, v_row in budget_entries:
            vid = v_row["variant_id"]

            # Determine quality_priority label
            score = v_row["mean_score"]
            if vid == lowest_cv["variant_id"]:
                quality_label = "robust"
            elif vid == best_quality["variant_id"]:
                quality_label = "maximum"
            elif score >= 4.0:
                quality_label = "high"
            else:
                quality_label = "good"

            # Confidence via Wilcoxon against next-best
            confidence = "low"
            p_val = 1.0
            try:
                other_variants = variant_stats[variant_stats["variant_id"] != vid]
                if not other_variants.empty:
                    next_best = other_variants.loc[other_variants["mean_score"].idxmax()]
                    v_scores = group[group["variant_id"] == vid]["overall_median"].values
                    nb_scores = group[group["variant_id"] == next_best["variant_id"]][
                        "overall_median"
                    ].values
                    if len(v_scores) >= 5 and len(nb_scores) >= 5:
                        min_len = min(len(v_scores), len(nb_scores))
                        _, p_val = scipy_stats.wilcoxon(v_scores[:min_len], nb_scores[:min_len])
                        if p_val < 0.01:
                            confidence = "high"
                        elif p_val < 0.05:
                            confidence = "medium"
            except (ValueError, ZeroDivisionError):
                pass

            fw.rows.append(
                FrameworkRow(
                    complexity=complexity,
                    budget_priority=budget_priority,
                    quality_priority=quality_label,
                    recommended_variant=vid,
                    expected_quality_iqr=(float(v_row["q25"]), float(v_row["q75"])),
                    expected_cost_iqr=(float(v_row["cost_q25"]), float(v_row["cost_q75"])),
                    confidence=confidence,
                    evidence=f"Based on {int(v_row['n'])} runs, p={p_val:.4f}",
                    expected_coherence_rate=coherence_by_variant.get(vid),
                    coherence_warning=(
                        f"Low coherence ({coherence_by_variant[vid]:.0%})"
                        if vid in coherence_by_variant
                        and coherence_by_variant[vid] < _COHERENCE_WARNING_THRESHOLD
                        else None
                    ),
                )
            )

    # Write outputs
    _export_framework(fw, output_dir)

    return fw


def _export_framework(fw: DecisionFramework, output_dir: Path) -> None:
    """Export framework as CSV and Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = output_dir / "tables" / "decision_framework.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in fw.rows:
        rows.append(
            {
                "complexity": r.complexity,
                "budget_priority": r.budget_priority,
                "quality_priority": r.quality_priority,
                "recommended_variant": r.recommended_variant,
                "quality_iqr_low": r.expected_quality_iqr[0],
                "quality_iqr_high": r.expected_quality_iqr[1],
                "cost_iqr_low": r.expected_cost_iqr[0],
                "cost_iqr_high": r.expected_cost_iqr[1],
                "coherence_rate": r.expected_coherence_rate,
                "coherence_warning": r.coherence_warning or "",
                "confidence": r.confidence,
                "evidence": r.evidence,
            }
        )
    if rows:
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    # Markdown
    md_path = output_dir / "reports" / "decision_framework.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# LLM Consortium Decision Framework",
        "",
        f"Generated: {fw.generation_date}",
        f"Runs analyzed: {fw.n_runs_analyzed}",
        f"Pareto-optimal variants: {', '.join(sorted(fw.pareto_variants))}",
        "",
    ]

    if fw.warnings:
        lines.append("## Warnings")
        for w in fw.warnings:
            lines.append(f"- ⚠ {w}")
        lines.append("")

    lines.append("## Recommendations")
    lines.append("")
    lines.append(
        "| Complexity | Budget | Quality | Variant | Quality IQR | Cost IQR | Coherence | Confidence |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in fw.rows:
        q_iqr = f"{r.expected_quality_iqr[0]:.2f}-{r.expected_quality_iqr[1]:.2f}"
        c_iqr = f"${r.expected_cost_iqr[0]:.2f}-${r.expected_cost_iqr[1]:.2f}"
        coh = f"{r.expected_coherence_rate:.0%}" if r.expected_coherence_rate is not None else "N/A"
        if r.coherence_warning:
            coh = f"{coh} ⚠"
        lines.append(
            f"| {r.complexity} | {r.budget_priority} | {r.quality_priority} "
            f"| **{r.recommended_variant}** | {q_iqr} | {c_iqr} | {coh} | {r.confidence} |"
        )
    lines.append("")

    md_path.write_text("\n".join(lines))
    logger.info("framework.exported", csv=str(csv_path), md=str(md_path))


# ── Prediction Validation ────────────────────────────────────────────────────


def validate_predictions(
    scores_df: pd.DataFrame,
    costs_df: pd.DataFrame,
    coherence_df: pd.DataFrame | None = None,
) -> list[PredictionResult]:
    """Test thesis predictions P1-P8 against experiment data."""
    results: list[PredictionResult] = []

    if scores_df.empty:
        for pid, desc in PREDICTIONS.items():
            results.append(PredictionResult(pid, desc, supported=False, evidence="No data"))
        return results

    # P1: v1 matches consortium quality on simple tasks
    results.append(_validate_p1(scores_df))

    # P2: v3 achieves highest peak quality on complex tasks
    results.append(_validate_p2(scores_df))

    # P3: v7 has lowest variance
    results.append(_validate_p3(scores_df))

    # P4: v4 has highest variance
    results.append(_validate_p4(scores_df))

    # P5: v5 outperforms v2 on specialist dimensions
    results.append(_validate_p5(scores_df))

    # P6: Rubric awareness > topology
    results.append(_validate_p6(scores_df))

    # P7: Consortium has higher coherence than v1
    coh_df = coherence_df if coherence_df is not None else pd.DataFrame()
    results.append(_validate_p7(coh_df))

    # P8: Quality and coherence are positively correlated
    results.append(_validate_p8(scores_df, coh_df))

    return results


def _validate_p1(df: pd.DataFrame) -> PredictionResult:
    """P1: v1_baseline matches consortium quality on simple tasks."""
    simple = (
        df[df.get("complexity", pd.Series(dtype=str)) == "simple"]
        if "complexity" in df.columns
        else pd.DataFrame()
    )

    baseline_mask = simple["variant_id"].isin(_BASELINE_IDS) if not simple.empty else pd.Series(dtype=bool)
    if simple.empty or not baseline_mask.any():
        return PredictionResult(
            "P1", PREDICTIONS["P1"], supported=False, evidence="Insufficient data"
        )

    v1_scores = simple[baseline_mask]["overall_median"].values
    other_scores = simple[~baseline_mask]["overall_median"].values

    if len(v1_scores) < 3 or len(other_scores) < 3:
        return PredictionResult(
            "P1", PREDICTIONS["P1"], supported=False, evidence="Insufficient data"
        )

    stat, p_val = scipy_stats.mannwhitneyu(v1_scores, other_scores, alternative="two-sided")
    supported = p_val > 0.05  # v1 is NOT significantly different → matches
    return PredictionResult(
        "P1",
        PREDICTIONS["P1"],
        supported=supported,
        p_value=p_val,
        evidence=f"Mann-Whitney U={stat:.1f}, p={p_val:.4f} (NS means baseline competitive)",
    )


def _validate_p2(df: pd.DataFrame) -> PredictionResult:
    """P2: v3 sub-variants achieve highest peak quality on complex tasks."""
    complex_df = (
        df[df.get("complexity", pd.Series(dtype=str)) == "complex"]
        if "complexity" in df.columns
        else pd.DataFrame()
    )

    if complex_df.empty:
        return PredictionResult(
            "P2", PREDICTIONS["P2"], supported=False, evidence="No complex tasks"
        )

    means = complex_df.groupby("variant_id")["overall_median"].mean()
    best = means.idxmax()
    # Supported if any v3 sub-variant is top
    supported = best in _V3_IDS
    v3_means = {vid: means[vid] for vid in means.index if vid in _V3_IDS}
    best_v3 = max(v3_means.values()) if v3_means else 0
    return PredictionResult(
        "P2",
        PREDICTIONS["P2"],
        supported=supported,
        evidence=f"Top variant on complex: {best} (mean={means[best]:.3f}), "
        f"best v3 sub-variant mean={best_v3:.3f}",
    )


def _validate_p3(df: pd.DataFrame) -> PredictionResult:
    """P3: v7_consensus has lowest variance across repetitions."""
    va = variance_analysis(df)
    if not va:
        return PredictionResult("P3", PREDICTIONS["P3"], supported=False, evidence="No data")

    lowest_cv = va[0]  # sorted ascending by CV
    supported = lowest_cv.variant_id in _V7_IDS
    v7_cv = next((v.cv for v in va if v.variant_id in _V7_IDS), "N/A")
    return PredictionResult(
        "P3",
        PREDICTIONS["P3"],
        supported=supported,
        evidence=f"Lowest CV: {lowest_cv.variant_id} (CV={lowest_cv.cv:.4f}), "
        f"v7 CV={v7_cv}",
    )


def _validate_p4(df: pd.DataFrame) -> PredictionResult:
    """P4: v4_adversarial has highest variance across repetitions."""
    va = variance_analysis(df)
    if not va:
        return PredictionResult("P4", PREDICTIONS["P4"], supported=False, evidence="No data")

    highest_cv = va[-1]  # sorted ascending
    supported = highest_cv.variant_id in _V4_IDS
    v4_cv = next((v.cv for v in va if v.variant_id in _V4_IDS), "N/A")
    return PredictionResult(
        "P4",
        PREDICTIONS["P4"],
        supported=supported,
        evidence=f"Highest CV: {highest_cv.variant_id} (CV={highest_cv.cv:.4f}), "
        f"v4 CV={v4_cv}",
    )


def _validate_p5(df: pd.DataFrame) -> PredictionResult:
    """P5: v5_specialist_panel outperforms v2 sub-variants on specialist dimensions."""
    v5 = df[df["variant_id"].isin(_V5_IDS)]
    v2 = df[df["variant_id"].isin(_V2_IDS)]

    if v5.empty or v2.empty:
        return PredictionResult(
            "P5", PREDICTIONS["P5"], supported=False, evidence="Missing v5 or v2 data"
        )

    # Parse dimension_medians to find security/scalability
    v5_specialist: list[float] = []
    v2_specialist: list[float] = []

    for _, row in v5.iterrows():
        try:
            dims = (
                json.loads(row["dimension_medians"])
                if isinstance(row["dimension_medians"], str)
                else {}
            )
            for k, v in dims.items():
                if any(kw in k.lower() for kw in ["security", "scalability", "performance"]):
                    v5_specialist.append(float(v))
        except (json.JSONDecodeError, TypeError):
            pass

    for _, row in v2.iterrows():
        try:
            dims = (
                json.loads(row["dimension_medians"])
                if isinstance(row["dimension_medians"], str)
                else {}
            )
            for k, v in dims.items():
                if any(kw in k.lower() for kw in ["security", "scalability", "performance"]):
                    v2_specialist.append(float(v))
        except (json.JSONDecodeError, TypeError):
            pass

    if len(v5_specialist) < 3 or len(v2_specialist) < 3:
        return PredictionResult(
            "P5", PREDICTIONS["P5"], supported=False, evidence="Insufficient dimension data"
        )

    stat, p_val = scipy_stats.mannwhitneyu(v5_specialist, v2_specialist, alternative="greater")
    supported = p_val < 0.05
    return PredictionResult(
        "P5",
        PREDICTIONS["P5"],
        supported=supported,
        p_value=p_val,
        evidence=f"v5 specialist mean={np.mean(v5_specialist):.3f}, "
        f"v2 specialist mean={np.mean(v2_specialist):.3f}, U={stat:.1f}, p={p_val:.4f}",
    )


def _validate_p6(df: pd.DataFrame) -> PredictionResult:
    """P6: Rubric awareness matters more than topology."""
    # Rubric-aware variants: those where the rubric is explicitly used in
    # review/merge prompts (v3 sub-variants, v4, v5, v6)
    rubric_aware_ids = _V3_IDS | _V4_IDS | _V5_IDS | {"v6_rotating_leader", "v6"}
    non_rubric_ids = _BASELINE_IDS | _V2_IDS | _V7_IDS | {"v8_structured_debate", "v8", "v1a_single_shot"}
    rubric_aware = df[df["variant_id"].isin(rubric_aware_ids)]["overall_median"].values
    non_rubric = df[df["variant_id"].isin(non_rubric_ids)]["overall_median"].values

    if len(rubric_aware) < 3 or len(non_rubric) < 3:
        return PredictionResult(
            "P6", PREDICTIONS["P6"], supported=False, evidence="Insufficient data"
        )

    stat, p_val = scipy_stats.mannwhitneyu(rubric_aware, non_rubric, alternative="greater")
    supported = p_val < 0.05
    return PredictionResult(
        "P6",
        PREDICTIONS["P6"],
        supported=supported,
        p_value=p_val,
        evidence=f"Rubric-aware mean={np.mean(rubric_aware):.3f}, "
        f"Non-rubric mean={np.mean(non_rubric):.3f}, U={stat:.1f}, p={p_val:.4f}",
    )


def _validate_p7(coherence_df: pd.DataFrame) -> PredictionResult:
    """P7: Consortium variants have higher coherence than single-LLM baseline."""
    if coherence_df.empty:
        return PredictionResult(
            "P7", PREDICTIONS["P7"], supported=False, evidence="No coherence data"
        )

    single_ids = _BASELINE_IDS | {"v1a_single_shot"}
    v1 = coherence_df[coherence_df["variant_id"].isin(single_ids)]["coherence_rate"].values
    consortium = coherence_df[~coherence_df["variant_id"].isin(single_ids)]["coherence_rate"].values

    if len(v1) < 3 or len(consortium) < 3:
        return PredictionResult(
            "P7",
            PREDICTIONS["P7"],
            supported=False,
            evidence=f"Insufficient data: v1 n={len(v1)}, consortium n={len(consortium)}",
        )

    stat, p_val = scipy_stats.mannwhitneyu(consortium, v1, alternative="greater")
    supported = p_val < 0.05
    return PredictionResult(
        "P7",
        PREDICTIONS["P7"],
        supported=supported,
        p_value=p_val,
        evidence=(
            f"Consortium mean coherence={np.mean(consortium):.3f}, "
            f"v1 mean={np.mean(v1):.3f}, U={stat:.1f}, p={p_val:.4f}"
        ),
    )


def _validate_p8(
    scores_df: pd.DataFrame,
    coherence_df: pd.DataFrame,
) -> PredictionResult:
    """P8: Quality and coherence are positively correlated (r > 0.3)."""
    if scores_df.empty or coherence_df.empty:
        return PredictionResult(
            "P8", PREDICTIONS["P8"], supported=False, evidence="No data"
        )

    corr = coherence_quality_correlation(scores_df, coherence_df)
    if corr is None:
        return PredictionResult(
            "P8",
            PREDICTIONS["P8"],
            supported=False,
            evidence="Insufficient paired data for correlation",
        )

    supported = corr.significant and corr.spearman_r > 0.3
    return PredictionResult(
        "P8",
        PREDICTIONS["P8"],
        supported=supported,
        p_value=corr.p_value,
        evidence=(
            f"Spearman r={corr.spearman_r:.3f}, p={corr.p_value:.4f}, "
            f"n={corr.n_designs} designs"
        ),
    )


def export_prediction_report(
    results: list[PredictionResult],
    output_dir: Path,
) -> None:
    """Export prediction validation as markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "reports" / "prediction_validation.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Thesis Prediction Validation (§10.1)",
        "",
        "| ID | Prediction | Supported? | p-value | Evidence |",
        "|---|---|---|---|---|",
    ]

    for r in results:
        check = "✅" if r.supported else "❌"
        p_str = f"{r.p_value:.4f}" if r.p_value is not None else "N/A"
        lines.append(f"| {r.prediction_id} | {r.description} | {check} | {p_str} | {r.evidence} |")

    lines.append("")
    supported = sum(1 for r in results if r.supported)
    lines.append(f"**{supported}/{len(results)} predictions supported by data.**")

    md_path.write_text("\n".join(lines))
    logger.info("predictions.exported", path=str(md_path))
