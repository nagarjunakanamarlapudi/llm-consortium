"""Unit tests for Pareto frontier computation."""

from __future__ import annotations

import pandas as pd
import pytest

from consortium.analysis.pareto import compute_pareto_frontier


class TestPareto:
    def test_basic_pareto(self):
        df = pd.DataFrame(
            {
                "variant_id": ["v1", "v2", "v3", "v4"],
                "mean_score": [3.0, 4.0, 3.5, 4.5],
                "mean_cost": [0.10, 0.20, 0.15, 0.30],
            }
        )
        frontier = compute_pareto_frontier(df)
        # v1(3.0, $0.10) → cheapest
        # v3(3.5, $0.15) → higher quality than v1 at incremental cost
        # v2(4.0, $0.20) → higher quality than v3
        # v4(4.5, $0.30) → highest quality
        # All 4 are Pareto-optimal (increasing quality with increasing cost)
        assert len(frontier) == 4
        assert frontier["variant_id"].tolist() == ["v1", "v3", "v2", "v4"]

    def test_single_point(self):
        df = pd.DataFrame(
            {
                "variant_id": ["v1"],
                "mean_score": [4.0],
                "mean_cost": [0.10],
            }
        )
        frontier = compute_pareto_frontier(df)
        assert len(frontier) == 1

    def test_empty_df(self):
        df = pd.DataFrame()
        frontier = compute_pareto_frontier(df)
        assert frontier.empty

    def test_all_same_score(self):
        df = pd.DataFrame(
            {
                "variant_id": ["v1", "v2", "v3"],
                "mean_score": [4.0, 4.0, 4.0],
                "mean_cost": [0.10, 0.20, 0.30],
            }
        )
        frontier = compute_pareto_frontier(df)
        # Only the cheapest should be Pareto-optimal since all have same quality
        assert len(frontier) == 1
        assert frontier.iloc[0]["variant_id"] == "v1"

    def test_dominated_points(self):
        df = pd.DataFrame(
            {
                "variant_id": ["v1", "v2"],
                "mean_score": [4.0, 3.0],
                "mean_cost": [0.10, 0.20],
            }
        )
        frontier = compute_pareto_frontier(df)
        # v2 is dominated by v1 (lower quality AND higher cost)
        assert len(frontier) == 1
        assert frontier.iloc[0]["variant_id"] == "v1"
