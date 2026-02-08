"""Unit tests for decision framework and prediction validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from consortium.analysis.framework import (
    DecisionFramework,
    FrameworkRow,
    PredictionResult,
    generate_decision_framework,
    validate_predictions,
)


@pytest.fixture
def scores_df():
    """Synthetic scores with complexity annotations."""
    np.random.seed(42)
    records = []
    complexities = {"t1": "simple", "t2": "medium", "t3": "complex"}
    for vid in ["v1", "v2", "v3", "v4"]:
        for tid in ["t1", "t2", "t3"]:
            for rep in range(3):
                base = {"v1": 3.5, "v2": 4.0, "v3": 4.2, "v4": 3.8}[vid]
                score = base + np.random.normal(0, 0.2)
                records.append(
                    {
                        "variant_id": vid,
                        "task_id": tid,
                        "repetition": rep,
                        "design_id": f"d_{vid}_{tid}_{rep}",
                        "run_id": f"r_{vid}_{tid}_{rep}",
                        "overall_median": max(1.0, min(5.0, score)),
                        "dimension_medians": '{"quality": 4.0, "security": 3.5}',
                        "complexity": complexities[tid],
                    }
                )
    return pd.DataFrame(records)


@pytest.fixture
def costs_df():
    np.random.seed(42)
    records = []
    for vid in ["v1", "v2", "v3", "v4"]:
        for tid in ["t1", "t2", "t3"]:
            for rep in range(3):
                base_cost = {"v1": 0.10, "v2": 0.15, "v3": 0.20, "v4": 0.12}[vid]
                records.append(
                    {
                        "run_id": f"r_{vid}_{tid}_{rep}",
                        "total_cost_usd": base_cost + np.random.normal(0, 0.01),
                    }
                )
    return pd.DataFrame(records)


class TestFramework:
    def test_generate_framework(self, scores_df, costs_df, tmp_path):
        fw = generate_decision_framework(scores_df, costs_df, tmp_path)

        assert isinstance(fw, DecisionFramework)
        assert len(fw.rows) > 0
        assert fw.n_runs_analyzed > 0

    def test_framework_output_files(self, scores_df, costs_df, tmp_path):
        generate_decision_framework(scores_df, costs_df, tmp_path)

        csv_path = tmp_path / "tables" / "decision_framework.csv"
        md_path = tmp_path / "reports" / "decision_framework.md"
        assert csv_path.exists()
        assert md_path.exists()

    def test_empty_data(self, tmp_path):
        fw = generate_decision_framework(pd.DataFrame(), pd.DataFrame(), tmp_path)
        assert len(fw.warnings) > 0

    def test_framework_row_fields(self, scores_df, costs_df, tmp_path):
        fw = generate_decision_framework(scores_df, costs_df, tmp_path)
        for row in fw.rows:
            assert isinstance(row, FrameworkRow)
            assert row.complexity in ("simple", "medium", "complex", "all")
            assert row.budget_priority in ("low", "moderate", "any")
            assert row.confidence in ("high", "medium", "low")


@pytest.fixture
def coherence_df():
    """Synthetic coherence DataFrame."""
    np.random.seed(42)
    records = []
    for vid in ["v1", "v2", "v3", "v4"]:
        for tid in ["t1", "t2", "t3"]:
            for rep in range(3):
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


class TestFrameworkWithCoherence:
    def test_framework_includes_coherence(self, scores_df, costs_df, coherence_df, tmp_path):
        fw = generate_decision_framework(scores_df, costs_df, tmp_path, coherence_df=coherence_df)
        assert isinstance(fw, DecisionFramework)
        # At least some rows should have coherence data
        has_coherence = any(r.expected_coherence_rate is not None for r in fw.rows)
        assert has_coherence

    def test_framework_no_coherence(self, scores_df, costs_df, tmp_path):
        fw = generate_decision_framework(scores_df, costs_df, tmp_path, coherence_df=None)
        for row in fw.rows:
            assert row.expected_coherence_rate is None
            assert row.coherence_warning is None


class TestPredictions:
    def test_validate_predictions_without_coherence(self, scores_df, costs_df):
        results = validate_predictions(scores_df, costs_df)
        assert len(results) == 8  # P1-P8
        for r in results:
            assert isinstance(r, PredictionResult)
            assert r.prediction_id.startswith("P")
        # P7 and P8 should report no coherence data
        p7 = next(r for r in results if r.prediction_id == "P7")
        assert not p7.supported
        assert "No coherence data" in p7.evidence or "Insufficient" in p7.evidence

    def test_validate_predictions_with_coherence(self, scores_df, costs_df, coherence_df):
        results = validate_predictions(scores_df, costs_df, coherence_df=coherence_df)
        assert len(results) == 8  # P1-P8
        p7 = next(r for r in results if r.prediction_id == "P7")
        assert p7.evidence != "No coherence data"  # Should have run the test
        p8 = next(r for r in results if r.prediction_id == "P8")
        assert p8.evidence != "No data"

    def test_empty_data(self):
        results = validate_predictions(pd.DataFrame(), pd.DataFrame())
        assert len(results) == 8
        for r in results:
            assert not r.supported
            assert "No data" in r.evidence
