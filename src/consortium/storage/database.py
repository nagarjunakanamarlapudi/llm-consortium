"""SQLite database wrapper with schema management."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import structlog

logger = structlog.get_logger()

SCHEMA_VERSION = 1

SCHEMA_SQL = """\
-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER PRIMARY KEY,
    applied_at  TEXT DEFAULT (datetime('now'))
);

-- Core experiment tracking
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    variant_id      TEXT NOT NULL,
    sub_variant     TEXT,
    task_id         TEXT NOT NULL,
    repetition      INTEGER NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',

    variant_config  TEXT NOT NULL,
    task_config     TEXT NOT NULL,
    model_configs   TEXT NOT NULL,

    total_input_tokens   INTEGER DEFAULT 0,
    total_output_tokens  INTEGER DEFAULT 0,
    total_cost_usd       REAL DEFAULT 0.0,

    started_at      TEXT,
    ended_at        TEXT,
    duration_seconds REAL,

    checkpoint_step TEXT,
    checkpoint_data TEXT,

    created_at      TEXT DEFAULT (datetime('now')),
    UNIQUE(variant_id, task_id, repetition)
);

-- Design artifacts
CREATE TABLE IF NOT EXISTS designs (
    design_id       TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(run_id),
    round           INTEGER NOT NULL,
    agent_role      TEXT NOT NULL,
    agent_id        TEXT NOT NULL,
    full_text       TEXT NOT NULL,
    token_count     INTEGER,
    is_final        BOOLEAN DEFAULT FALSE,
    created_at      TEXT DEFAULT (datetime('now'))
);

-- Review artifacts
CREATE TABLE IF NOT EXISTS reviews (
    review_id       TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(run_id),
    design_id       TEXT NOT NULL REFERENCES designs(design_id),
    round           INTEGER NOT NULL,
    agent_role      TEXT NOT NULL,
    agent_id        TEXT NOT NULL,
    review_text     TEXT NOT NULL,
    verdict         TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

-- Evaluation scores (3 per final design)
CREATE TABLE IF NOT EXISTS evaluations (
    evaluation_id   TEXT PRIMARY KEY,
    design_id       TEXT NOT NULL REFERENCES designs(design_id),
    evaluator_run   INTEGER NOT NULL,
    evaluator_model TEXT NOT NULL,

    dimension_scores TEXT NOT NULL,
    overall_score   REAL NOT NULL,
    qualitative_summary TEXT,

    input_tokens    INTEGER,
    output_tokens   INTEGER,
    cost_usd        REAL,

    batch_id        TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

-- Median scores (aggregated from 3 evaluator runs)
CREATE TABLE IF NOT EXISTS scores_median (
    design_id       TEXT PRIMARY KEY REFERENCES designs(design_id),
    run_id          TEXT NOT NULL REFERENCES runs(run_id),

    dimension_medians TEXT NOT NULL,
    overall_median  REAL NOT NULL,

    disagreement_flags TEXT,
    krippendorff_alpha REAL,

    created_at      TEXT DEFAULT (datetime('now'))
);

-- Coherence checks
CREATE TABLE IF NOT EXISTS coherence_checks (
    check_id        TEXT PRIMARY KEY,
    design_id       TEXT NOT NULL REFERENCES designs(design_id),
    section_pair    TEXT NOT NULL,
    contradicts     BOOLEAN NOT NULL,
    explanation     TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

-- LLM call traces
CREATE TABLE IF NOT EXISTS traces (
    trace_id        TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(run_id),
    variant_id      TEXT NOT NULL,
    task_id         TEXT NOT NULL,
    repetition      INTEGER NOT NULL,

    agent_role      TEXT NOT NULL,
    agent_id        TEXT NOT NULL,
    step            TEXT NOT NULL,
    round           INTEGER NOT NULL,

    model_config_id TEXT NOT NULL,
    api_model       TEXT NOT NULL,
    provider        TEXT NOT NULL,

    system_prompt_hash TEXT,
    prompt_template TEXT,

    input_tokens    INTEGER NOT NULL,
    output_tokens   INTEGER NOT NULL,
    cached_input_tokens INTEGER DEFAULT 0,
    cost_usd        REAL NOT NULL,
    latency_ms      REAL NOT NULL,

    started_at      TEXT NOT NULL,
    ended_at        TEXT NOT NULL,

    batch_id        TEXT,
    status          TEXT DEFAULT 'success',
    error           TEXT,
    retry_count     INTEGER DEFAULT 0,

    created_at      TEXT DEFAULT (datetime('now'))
);

-- Batch API tracking
CREATE TABLE IF NOT EXISTS batches (
    batch_id        TEXT PRIMARY KEY,
    provider        TEXT NOT NULL,
    purpose         TEXT NOT NULL,
    status          TEXT NOT NULL,
    request_count   INTEGER NOT NULL,
    completed_count INTEGER DEFAULT 0,
    failed_count    INTEGER DEFAULT 0,

    submitted_at    TEXT,
    completed_at    TEXT,

    request_file    TEXT,
    response_file   TEXT,

    created_at      TEXT DEFAULT (datetime('now'))
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_runs_variant_task ON runs(variant_id, task_id);
CREATE INDEX IF NOT EXISTS idx_designs_run ON designs(run_id);
CREATE INDEX IF NOT EXISTS idx_designs_final ON designs(is_final) WHERE is_final = TRUE;
CREATE INDEX IF NOT EXISTS idx_evaluations_design ON evaluations(design_id);
CREATE INDEX IF NOT EXISTS idx_traces_run ON traces(run_id);
CREATE INDEX IF NOT EXISTS idx_traces_variant_task ON traces(variant_id, task_id);
CREATE INDEX IF NOT EXISTS idx_traces_agent ON traces(agent_role, agent_id);
CREATE INDEX IF NOT EXISTS idx_batches_status ON batches(status);
"""


class Database:
    """SQLite database wrapper for experiment storage."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        """Get the active connection, opening one if needed."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def init_schema(self) -> None:
        """Create all tables and indexes."""
        self.conn.executescript(SCHEMA_SQL)
        # Record schema version
        self.conn.execute(
            "INSERT OR IGNORE INTO schema_version (version) VALUES (?)",
            (SCHEMA_VERSION,),
        )
        self.conn.commit()
        logger.info("database_initialized", path=str(self._db_path), version=SCHEMA_VERSION)

    def get_schema_version(self) -> int | None:
        """Get the current schema version, or None if not initialized."""
        try:
            row = self.conn.execute(
                "SELECT MAX(version) FROM schema_version"
            ).fetchone()
            return row[0] if row else None
        except sqlite3.OperationalError:
            return None

    def stats(self) -> dict[str, int]:
        """Return counts for all major tables."""
        tables = ["runs", "designs", "reviews", "evaluations", "scores_median", "traces", "batches"]
        result: dict[str, int] = {}
        for table in tables:
            try:
                row = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()  # noqa: S608
                result[table] = row[0] if row else 0
            except sqlite3.OperationalError:
                result[table] = 0
        return result

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
