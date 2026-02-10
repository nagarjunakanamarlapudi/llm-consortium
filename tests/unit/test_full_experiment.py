"""Tests for the FullExperimentRunner streaming pipeline."""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from consortium.config.models import (
    ExperimentConfig,
    FullConfig,
    LimitsConfig,
    ModelConfig,
    BatchingConfig,
)
from consortium.pipeline.full_experiment import FullExperimentRunner, FullExperimentStats


# ── Helpers ──────────────────────────────────────────────────────────────────


def _create_in_memory_db() -> MagicMock:
    """Create a Database mock wrapping a real in-memory SQLite connection."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE runs (
            run_id TEXT PRIMARY KEY,
            variant_id TEXT NOT NULL,
            sub_variant TEXT,
            task_id TEXT NOT NULL,
            repetition INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            variant_config TEXT NOT NULL DEFAULT '{}',
            task_config TEXT NOT NULL DEFAULT '{}',
            model_configs TEXT NOT NULL DEFAULT '{}',
            total_input_tokens INTEGER DEFAULT 0,
            total_output_tokens INTEGER DEFAULT 0,
            total_cost_usd REAL DEFAULT 0.0,
            started_at TEXT,
            ended_at TEXT,
            duration_seconds REAL,
            checkpoint_step TEXT,
            checkpoint_data TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            UNIQUE(variant_id, task_id, repetition)
        );

        CREATE TABLE designs (
            design_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(run_id),
            round INTEGER NOT NULL,
            agent_role TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            full_text TEXT NOT NULL,
            token_count INTEGER,
            is_final BOOLEAN DEFAULT FALSE,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE scores_median (
            design_id TEXT PRIMARY KEY REFERENCES designs(design_id),
            run_id TEXT NOT NULL REFERENCES runs(run_id),
            dimension_medians TEXT NOT NULL DEFAULT '{}',
            overall_median REAL NOT NULL DEFAULT 0.0,
            disagreement_flags TEXT,
            krippendorff_alpha REAL,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE coherence_checks (
            check_id TEXT PRIMARY KEY,
            design_id TEXT NOT NULL REFERENCES designs(design_id),
            section_pair TEXT NOT NULL,
            contradicts BOOLEAN NOT NULL,
            explanation TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
        """
    )

    db_mock = MagicMock()
    db_mock.conn = conn
    db_mock.close = MagicMock()
    db_mock.init_schema = MagicMock()
    return db_mock


def _create_minimal_config(
    variants: list[str] | None = None,
    tasks: list[str] | None = None,
) -> FullConfig:
    """Create a minimal FullConfig for testing."""
    variant_ids = variants or ["v1", "v2"]
    task_ids = tasks or ["t1"]

    return FullConfig(
        experiment=ExperimentConfig(
            variants=variant_ids,
            tasks=task_ids,
            repetitions=1,
            limits=LimitsConfig(max_concurrent_runs=5),
        ),
        models={
            "test-model": ModelConfig(
                id="test-model",
                provider="openai",
                api_model="test-model-api",
                batching=BatchingConfig(enabled=False),
            ),
        },
    )


def _seed_completed_run(db_mock, variant_id: str, task_id: str, rep: int = 0) -> str:
    """Insert a completed run with final design into the test DB."""
    run_id = f"{variant_id}-{task_id}-rep{rep}-test"
    design_id = f"design-{run_id}"
    conn = db_mock.conn
    conn.execute(
        """INSERT INTO runs (run_id, variant_id, task_id, repetition, status,
           variant_config, task_config, model_configs)
           VALUES (?, ?, ?, ?, 'completed', '{}', '{}', '{}')""",
        (run_id, variant_id, task_id, rep),
    )
    conn.execute(
        """INSERT INTO designs (design_id, run_id, round, agent_role, agent_id,
           full_text, is_final)
           VALUES (?, ?, 0, 'designer', 'designer_0', 'Design text...', TRUE)""",
        (design_id, run_id),
    )
    conn.commit()
    return run_id


def _seed_evaluation(db_mock, run_id: str) -> None:
    """Mark a run as evaluated."""
    design_id = f"design-{run_id}"
    db_mock.conn.execute(
        """INSERT INTO scores_median (design_id, run_id, dimension_medians, overall_median)
           VALUES (?, ?, '{}', 7.5)""",
        (design_id, run_id),
    )
    db_mock.conn.commit()


def _seed_coherence(db_mock, run_id: str) -> None:
    """Mark a run as coherence-checked."""
    design_id = f"design-{run_id}"
    db_mock.conn.execute(
        """INSERT INTO coherence_checks (check_id, design_id, section_pair, contradicts)
           VALUES (?, ?, 'A vs B', FALSE)""",
        (f"chk-{run_id}", design_id),
    )
    db_mock.conn.commit()


# ── Stats ────────────────────────────────────────────────────────────────────


class TestFullExperimentStats:
    def test_all_stages_complete_true(self) -> None:
        stats = FullExperimentStats(
            total_runs=5,
            generation_completed=5,
            evaluation_completed=5,
            coherence_completed=5,
        )
        assert stats.all_stages_complete is True

    def test_all_stages_complete_with_failures(self) -> None:
        stats = FullExperimentStats(
            total_runs=5,
            generation_completed=3,
            evaluation_completed=3,
            coherence_completed=3,
            failed=2,
        )
        assert stats.all_stages_complete is True

    def test_all_stages_not_complete(self) -> None:
        stats = FullExperimentStats(
            total_runs=5,
            generation_completed=3,
            evaluation_completed=2,
            coherence_completed=1,
        )
        assert stats.all_stages_complete is False

    def test_defaults(self) -> None:
        stats = FullExperimentStats()
        assert stats.total_runs == 0
        assert stats.failed == 0
        assert stats.analysis_complete is False


# ── Pending Runs Logic ───────────────────────────────────────────────────────


class TestGetPendingRuns:
    def test_all_pending_when_no_runs_exist(self) -> None:
        db = _create_in_memory_db()
        config = _create_minimal_config(variants=["v1"], tasks=["t1"])
        runner = FullExperimentRunner(config, db, "prompts")

        run_matrix = [("v1", "t1", 0)]
        pending = runner._get_pending_runs(run_matrix, force=False)

        assert ("v1", "t1", 0) in pending["generate"]
        assert ("v1", "t1", 0) in pending["evaluate"]
        assert ("v1", "t1", 0) in pending["coherence"]

    def test_nothing_pending_when_fully_complete(self) -> None:
        db = _create_in_memory_db()
        config = _create_minimal_config(variants=["v1"], tasks=["t1"])
        runner = FullExperimentRunner(config, db, "prompts")

        run_id = _seed_completed_run(db, "v1", "t1")
        _seed_evaluation(db, run_id)
        _seed_coherence(db, run_id)

        run_matrix = [("v1", "t1", 0)]
        pending = runner._get_pending_runs(run_matrix, force=False)

        assert len(pending["generate"]) == 0
        assert len(pending["evaluate"]) == 0
        assert len(pending["coherence"]) == 0

    def test_eval_pending_when_not_evaluated(self) -> None:
        db = _create_in_memory_db()
        config = _create_minimal_config(variants=["v1"], tasks=["t1"])
        runner = FullExperimentRunner(config, db, "prompts")

        run_id = _seed_completed_run(db, "v1", "t1")
        _seed_coherence(db, run_id)
        # No evaluation seeded

        run_matrix = [("v1", "t1", 0)]
        pending = runner._get_pending_runs(run_matrix, force=False)

        assert len(pending["generate"]) == 0
        assert ("v1", "t1", 0) in pending["evaluate"]
        assert len(pending["coherence"]) == 0

    def test_coherence_pending_when_not_checked(self) -> None:
        db = _create_in_memory_db()
        config = _create_minimal_config(variants=["v1"], tasks=["t1"])
        runner = FullExperimentRunner(config, db, "prompts")

        run_id = _seed_completed_run(db, "v1", "t1")
        _seed_evaluation(db, run_id)
        # No coherence seeded

        run_matrix = [("v1", "t1", 0)]
        pending = runner._get_pending_runs(run_matrix, force=False)

        assert len(pending["generate"]) == 0
        assert len(pending["evaluate"]) == 0
        assert ("v1", "t1", 0) in pending["coherence"]

    def test_force_makes_everything_pending(self) -> None:
        db = _create_in_memory_db()
        config = _create_minimal_config(variants=["v1"], tasks=["t1"])
        runner = FullExperimentRunner(config, db, "prompts")

        run_id = _seed_completed_run(db, "v1", "t1")
        _seed_evaluation(db, run_id)
        _seed_coherence(db, run_id)

        run_matrix = [("v1", "t1", 0)]
        pending = runner._get_pending_runs(run_matrix, force=True)

        assert ("v1", "t1", 0) in pending["generate"]
        assert ("v1", "t1", 0) in pending["evaluate"]
        assert ("v1", "t1", 0) in pending["coherence"]

    def test_multiple_runs_mixed_status(self) -> None:
        db = _create_in_memory_db()
        config = _create_minimal_config(variants=["v1", "v2"], tasks=["t1"])
        runner = FullExperimentRunner(config, db, "prompts")

        # v1-t1 fully complete
        run_id = _seed_completed_run(db, "v1", "t1")
        _seed_evaluation(db, run_id)
        _seed_coherence(db, run_id)

        # v2-t1 not started

        run_matrix = [("v1", "t1", 0), ("v2", "t1", 0)]
        pending = runner._get_pending_runs(run_matrix, force=False)

        assert ("v1", "t1", 0) not in pending["generate"]
        assert ("v2", "t1", 0) in pending["generate"]
        assert ("v2", "t1", 0) in pending["evaluate"]
        assert ("v2", "t1", 0) in pending["coherence"]


# ── Database Helpers ─────────────────────────────────────────────────────────


class TestDatabaseHelpers:
    def test_get_run_status_existing(self) -> None:
        db = _create_in_memory_db()
        config = _create_minimal_config()
        runner = FullExperimentRunner(config, db, "prompts")

        _seed_completed_run(db, "v1", "t1")
        assert runner._get_run_status("v1", "t1", 0) == "completed"

    def test_get_run_status_nonexistent(self) -> None:
        db = _create_in_memory_db()
        config = _create_minimal_config()
        runner = FullExperimentRunner(config, db, "prompts")
        assert runner._get_run_status("v1", "t1", 0) is None

    def test_get_run_id(self) -> None:
        db = _create_in_memory_db()
        config = _create_minimal_config()
        runner = FullExperimentRunner(config, db, "prompts")

        _seed_completed_run(db, "v1", "t1")
        run_id = runner._get_run_id("v1", "t1", 0)
        assert run_id == "v1-t1-rep0-test"

    def test_has_evaluation(self) -> None:
        db = _create_in_memory_db()
        config = _create_minimal_config()
        runner = FullExperimentRunner(config, db, "prompts")

        run_id = _seed_completed_run(db, "v1", "t1")
        assert runner._has_evaluation(run_id) is False

        _seed_evaluation(db, run_id)
        assert runner._has_evaluation(run_id) is True

    def test_has_coherence(self) -> None:
        db = _create_in_memory_db()
        config = _create_minimal_config()
        runner = FullExperimentRunner(config, db, "prompts")

        run_id = _seed_completed_run(db, "v1", "t1")
        assert runner._has_coherence(run_id) is False

        _seed_coherence(db, run_id)
        assert runner._has_coherence(run_id) is True
