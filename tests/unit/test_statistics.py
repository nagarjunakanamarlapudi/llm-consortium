"""Unit tests for statistics module — Friedman, Wilcoxon, Mann-Whitney U."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from consortium.analysis.statistics import (
    AxisEffect,
    CoherenceCorrelation,
    EfficiencyResult,
    PairwiseResult,
    RankingResult,
    VarianceResult,
    analyze_axis_effect,
    coherence_quality_correlation,
    compare_coherence_to_baseline,
    compare_to_baseline,
    pairwise_comparisons,
    rank_variants_by_coherence,
    rank_variants_overall,
    token_efficiency,
    variance_analysis,
)


@pytest.fixture
def scores_df():
    """Synthetic scores DataFrame with 4 variants, 2 tasks, 3 reps."""
    np.random.seed(42)
    records = []
    for vid in ["v1", "v2", "v3", "v4"]:
        for tid in ["t1", "t2"]:
            for rep in range(3):
                base = {"v1": 3.5, "v2": 4.0, "v3": 4.2, "v4": 3.8}[vid]
                score = base + np.random.normal(0, 0.3)
                records.append(
                    {
                        "variant_id": vid,
                        "task_id": tid,
                        "repetition": rep,
                        "design_id": f"d_{vid}_{tid}_{rep}",
                        "run_id": f"r_{vid}_{tid}_{rep}",
                        "overall_median": max(1.0, min(5.0, score)),
                        "dimension_medians": '{"quality": 4.0, "security": 3.5}',
                        "krippendorff_alpha": 0.8,
                        "disagreement_flags": None,
                    }
                )
    return pd.DataFrame(records)


@pytest.fixture
def costs_df():
    """Synthetic costs DataFrame."""
    np.random.seed(42)
    records = []
    for vid in ["v1", "v2", "v3", "v4"]:
        for tid in ["t1", "t2"]:
            for rep in range(3):
                base_cost = {"v1": 0.10, "v2": 0.15, "v3": 0.20, "v4": 0.12}[vid]
                records.append(
                    {
                        "variant_id": vid,
                        "task_id": tid,
                        "repetition": rep,
                        "run_id": f"r_{vid}_{tid}_{rep}",
                        "total_cost_usd": base_cost + np.random.normal(0, 0.02),
                        "total_input_tokens": 1000 + np.random.randint(0, 500),
                        "total_output_tokens": 500 + np.random.randint(0, 200),
                        "duration_seconds": 5.0 + np.random.normal(0, 1),
                    }
                )
    return pd.DataFrame(records)


class TestRanking:
    def test_rank_variants_overall(self, scores_df):
        result = rank_variants_overall(scores_df)
        assert result is not None
        assert isinstance(result, RankingResult)
        assert len(result.variant_ids) == 4
        assert result.test_name == "Friedman"
        assert result.statistic > 0

    def test_rank_empty_df(self):
        result = rank_variants_overall(pd.DataFrame())
        assert result is None

    def test_rank_single_variant(self):
        df = pd.DataFrame(
            [
                {
                    "variant_id": "v1",
                    "task_id": "t1",
                    "repetition": 0,
                    "overall_median": 4.0,
                }
            ]
        )
        result = rank_variants_overall(df)
        assert result is None  # Need >= 2 variants


class TestBaselineComparison:
    def test_compare_to_baseline(self, scores_df):
        results = compare_to_baseline(scores_df, "v1")
        assert len(results) == 3  # v2, v3, v4
        for r in results:
            assert isinstance(r, PairwiseResult)
            assert r.variant_a == "v1"
            assert 0.0 <= r.p_adjusted <= 1.0

    def test_compare_empty_df(self):
        results = compare_to_baseline(pd.DataFrame())
        assert results == []


class TestPairwiseComparisons:
    def test_pairwise(self, scores_df):
        results = pairwise_comparisons(scores_df)
        # 4 variants → 4*3/2 = 6 pairs
        assert len(results) == 6
        for r in results:
            assert isinstance(r, PairwiseResult)

    def test_pairwise_empty(self):
        assert pairwise_comparisons(pd.DataFrame()) == []


class TestAxisAnalysis:
    def test_analyze_axis(self, scores_df):
        result = analyze_axis_effect(scores_df, "authority")
        # Only has v1-v4 and only v1-v4 are in taxonomy
        if result is not None:
            assert isinstance(result, AxisEffect)
            assert result.axis_name == "authority"

    def test_invalid_axis(self, scores_df):
        result = analyze_axis_effect(scores_df, "invalid")
        assert result is None


class TestVarianceAnalysis:
    def test_variance(self, scores_df):
        results = variance_analysis(scores_df)
        assert len(results) == 4
        for r in results:
            assert isinstance(r, VarianceResult)
            assert r.cv >= 0
        # Should be sorted ascending by CV
        cvs = [r.cv for r in results]
        assert cvs == sorted(cvs)


class TestTokenEfficiency:
    def test_efficiency(self, scores_df, costs_df):
        results = token_efficiency(scores_df, costs_df)
        assert len(results) > 0
        for r in results:
            assert isinstance(r, EfficiencyResult)
            assert r.quality_per_dollar > 0

    def test_efficiency_empty(self):
        assert token_efficiency(pd.DataFrame(), pd.DataFrame()) == []


# ── Coherence Tests ──────────────────────────────────────────────────────────


@pytest.fixture
def coherence_df():
    """Synthetic coherence DataFrame with 4 variants, 2 tasks, 3 reps."""
    np.random.seed(42)
    records = []
    for vid in ["v1", "v2", "v3", "v4"]:
        for tid in ["t1", "t2"]:
            for rep in range(3):
                # v1 has lower coherence, v3 has highest
                base_rate = {"v1": 0.70, "v2": 0.85, "v3": 0.95, "v4": 0.80}[vid]
                pairs = 5
                contradictions = max(0, round(pairs * (1 - base_rate) + np.random.normal(0, 0.3)))
                coherence_rate = 1.0 - (contradictions / pairs)
                records.append(
                    {
                        "variant_id": vid,
                        "task_id": tid,
                        "repetition": rep,
                        "design_id": f"d_{vid}_{tid}_{rep}",
                        "run_id": f"r_{vid}_{tid}_{rep}",
                        "pairs_checked": pairs,
                        "contradictions": contradictions,
                        "coherence_rate": coherence_rate,
                    }
                )
    return pd.DataFrame(records)


class TestCoherenceRanking:
    def test_rank_by_coherence(self, coherence_df):
        result = rank_variants_by_coherence(coherence_df)
        assert result is not None
        assert isinstance(result, RankingResult)
        assert len(result.variant_ids) == 4
        assert result.test_name == "Friedman (coherence)"

    def test_rank_coherence_empty(self):
        result = rank_variants_by_coherence(pd.DataFrame())
        assert result is None


class TestCoherenceBaseline:
    def test_compare_coherence_to_baseline(self, coherence_df):
        results = compare_coherence_to_baseline(coherence_df, "v1")
        assert len(results) == 3  # v2, v3, v4
        for r in results:
            assert isinstance(r, PairwiseResult)
            assert r.variant_a == "v1"

    def test_compare_coherence_empty(self):
        results = compare_coherence_to_baseline(pd.DataFrame())
        assert results == []


class TestCoherenceQualityCorrelation:
    def test_correlation(self, scores_df, coherence_df):
        corr = coherence_quality_correlation(scores_df, coherence_df)
        assert corr is not None
        assert isinstance(corr, CoherenceCorrelation)
        assert -1.0 <= corr.spearman_r <= 1.0
        assert corr.n_designs > 0

    def test_correlation_empty(self):
        result = coherence_quality_correlation(pd.DataFrame(), pd.DataFrame())
        assert result is None

    def test_correlation_no_overlap(self, scores_df):
        """No matching design_ids should return None."""
        coh_df = pd.DataFrame(
            [
                {
                    "design_id": "no_match",
                    "coherence_rate": 0.9,
                }
            ]
        )
        result = coherence_quality_correlation(scores_df, coh_df)
        assert result is None
