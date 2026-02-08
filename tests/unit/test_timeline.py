"""Unit tests for the timeline builder and data models."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from consortium.observability.timeline import (
    RunTimeline,
    SwimLane,
    TimelineBuilder,
    TraceEvent,
    VariantSummary,
)


@pytest.fixture
def mock_db(tmp_path):
    """Create a mock database with traces and runs tables."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    conn.execute("""
        CREATE TABLE runs (
            run_id TEXT PRIMARY KEY,
            variant_id TEXT, task_id TEXT, repetition INTEGER,
            status TEXT, duration_seconds REAL,
            total_cost_usd REAL, total_input_tokens INTEGER, total_output_tokens INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE traces (
            trace_id TEXT PRIMARY KEY, run_id TEXT, agent_id TEXT, agent_role TEXT,
            step TEXT, round INTEGER, started_at TEXT, ended_at TEXT,
            input_tokens INTEGER, output_tokens INTEGER,
            cached_input_tokens INTEGER DEFAULT 0, cost_usd REAL,
            api_model TEXT, system_prompt_hash TEXT,
            prompt_text TEXT, response_text TEXT
        )
    """)

    # Insert test data
    now = datetime(2025, 1, 1, 12, 0, 0)
    conn.execute(
        "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("run1", "v1", "t1", 0, "completed", 10.0, 0.05, 1000, 500),
    )
    for i in range(3):
        start = now + timedelta(seconds=i * 2)
        end = start + timedelta(seconds=1.5)
        conn.execute(
            "INSERT INTO traces VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"trace_{i}",
                "run1",
                f"agent_{i % 2}",
                "designer",
                "generation",
                1,
                start.isoformat(),
                end.isoformat(),
                100 + i * 10,
                50 + i * 5,
                0,  # cached_input_tokens
                0.01 + i * 0.005,
                "gpt-4",
                "hash123",
                None,
                None,
            ),
        )
    conn.commit()

    db_mock = MagicMock()
    db_mock.conn = conn
    return db_mock


class TestTimelineBuilder:
    def test_build_run_timeline(self, mock_db):
        builder = TimelineBuilder(mock_db)
        timeline = builder.build_run_timeline("run1")

        assert isinstance(timeline, RunTimeline)
        assert timeline.run_id == "run1"
        assert timeline.variant_id == "v1"
        assert timeline.task_id == "t1"
        assert timeline.total_tokens > 0
        assert timeline.total_cost_usd > 0
        assert len(timeline.swim_lanes) == 2  # agent_0 and agent_1

    def test_build_variant_summary(self, mock_db):
        builder = TimelineBuilder(mock_db)
        summary = builder.build_variant_summary("v1", "t1")

        assert isinstance(summary, VariantSummary)
        assert summary.variant_id == "v1"
        assert summary.run_count == 1
        assert summary.mean_cost_usd > 0

    def test_build_variant_summary_no_data(self, mock_db):
        builder = TimelineBuilder(mock_db)
        summary = builder.build_variant_summary("v999", "t1")

        assert summary.run_count == 0
        assert summary.mean_cost_usd == 0.0

    def test_build_comparison(self, mock_db):
        builder = TimelineBuilder(mock_db)
        comparison = builder.build_comparison(["v1", "v2"], "t1")

        assert comparison.task_id == "t1"
        assert len(comparison.variants) == 2

    def test_detect_parallelism(self):
        now = datetime(2025, 1, 1, 12, 0, 0)
        events = [
            TraceEvent(
                trace_id="a",
                run_id="r",
                agent_id="a0",
                step="gen",
                round_num=1,
                started_at=now,
                ended_at=now + timedelta(seconds=2),
                duration_ms=2000,
                input_tokens=10,
                output_tokens=5,
                cost_usd=0.01,
                model="m",
                prompt_hash="h",
                cached_input_tokens=0,
            ),
            TraceEvent(
                trace_id="b",
                run_id="r",
                agent_id="a1",
                step="gen",
                round_num=1,
                started_at=now + timedelta(seconds=1),
                ended_at=now + timedelta(seconds=3),
                duration_ms=2000,
                input_tokens=10,
                output_tokens=5,
                cost_usd=0.01,
                model="m",
                prompt_hash="h",
                cached_input_tokens=0,
            ),
        ]
        groups = TimelineBuilder._detect_parallelism(events)
        assert len(groups) >= 1  # overlapping events detected

    def test_no_parallelism(self):
        now = datetime(2025, 1, 1, 12, 0, 0)
        events = [
            TraceEvent(
                trace_id="a",
                run_id="r",
                agent_id="a0",
                step="gen",
                round_num=1,
                started_at=now,
                ended_at=now + timedelta(seconds=1),
                duration_ms=1000,
                input_tokens=10,
                output_tokens=5,
                cost_usd=0.01,
                model="m",
                prompt_hash="h",
                cached_input_tokens=0,
            ),
            TraceEvent(
                trace_id="b",
                run_id="r",
                agent_id="a1",
                step="gen",
                round_num=1,
                started_at=now + timedelta(seconds=2),
                ended_at=now + timedelta(seconds=3),
                duration_ms=1000,
                input_tokens=10,
                output_tokens=5,
                cost_usd=0.01,
                model="m",
                prompt_hash="h",
                cached_input_tokens=0,
            ),
        ]
        groups = TimelineBuilder._detect_parallelism(events)
        assert len(groups) == 0
