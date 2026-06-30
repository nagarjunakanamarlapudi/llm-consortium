"""Tests for the code_results table and the v3 -> v4 schema migration."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import pytest

from consortium.storage.database import SCHEMA_VERSION, Database

if TYPE_CHECKING:
    from pathlib import Path


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row is not None


def _index_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name=?",
        (name,),
    ).fetchone()
    return row is not None


class TestCodeResultsSchema:
    def test_fresh_db_has_code_results_at_v4(self, tmp_db: Path) -> None:
        """A freshly initialized database is at version 4 with code_results present."""
        with Database(tmp_db) as db:
            db.init_schema()

            assert db.get_schema_version() == SCHEMA_VERSION
            assert SCHEMA_VERSION == 4

            assert _table_exists(db.conn, "code_results")
            assert _index_exists(db.conn, "idx_code_results_run")
            assert _index_exists(db.conn, "idx_code_results_benchmark")

    def test_code_results_counted_in_stats(self, tmp_db: Path) -> None:
        """stats() reports both the new code_results and previously-missing tables."""
        with Database(tmp_db) as db:
            db.init_schema()
            stats = db.stats()

        assert stats["code_results"] == 0
        assert stats["coherence_checks"] == 0

    def test_code_results_insert_and_query(self, tmp_db: Path) -> None:
        """A code_results row can be inserted against a real design/run."""
        with Database(tmp_db) as db:
            db.init_schema()
            db.conn.execute(
                """INSERT INTO runs (run_id, variant_id, task_id, repetition, status,
                   variant_config, task_config, model_configs)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                ("run-1", "v1", "t1", 1, "completed", "{}", "{}", "{}"),
            )
            db.conn.execute(
                """INSERT INTO designs (design_id, run_id, round, agent_role,
                   agent_id, full_text)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                ("d-1", "run-1", 0, "designer", "leader", "text"),
            )
            db.conn.execute(
                """INSERT INTO code_results (result_id, design_id, run_id, benchmark,
                   problem_id, passed, n_pass, n_total, error_type, exec_ms, harness_meta)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                ("cr-1", "d-1", "run-1", "humaneval", "HE/1", 1, 5, 5, None, 12.5, "{}"),
            )
            db.conn.commit()

            row = db.conn.execute(
                "SELECT passed, n_pass, n_total FROM code_results WHERE result_id = ?",
                ("cr-1",),
            ).fetchone()
            assert row["passed"] == 1
            assert row["n_pass"] == 5
            assert row["n_total"] == 5

    def test_code_results_design_id_unique(self, tmp_db: Path) -> None:
        """Only one code_results row may exist per design_id."""
        with Database(tmp_db) as db:
            db.init_schema()
            db.conn.execute(
                """INSERT INTO runs (run_id, variant_id, task_id, repetition, status,
                   variant_config, task_config, model_configs)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                ("run-1", "v1", "t1", 1, "completed", "{}", "{}", "{}"),
            )
            db.conn.execute(
                """INSERT INTO designs (design_id, run_id, round, agent_role,
                   agent_id, full_text)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                ("d-1", "run-1", 0, "designer", "leader", "text"),
            )
            db.conn.execute(
                """INSERT INTO code_results (result_id, design_id, run_id, benchmark,
                   problem_id, passed)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                ("cr-1", "d-1", "run-1", "humaneval", "HE/1", 1),
            )
            db.conn.commit()
            with pytest.raises(sqlite3.IntegrityError):
                db.conn.execute(
                    """INSERT INTO code_results (result_id, design_id, run_id, benchmark,
                       problem_id, passed)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    ("cr-2", "d-1", "run-1", "humaneval", "HE/1", 0),
                )

    @pytest.mark.parametrize(
        ("column", "values"),
        [
            ("design_id", ("cr-1", None, "run-1", "humaneval", "HE/1", 1)),
            ("run_id", ("cr-1", "d-1", None, "humaneval", "HE/1", 1)),
            ("benchmark", ("cr-1", "d-1", "run-1", None, "HE/1", 1)),
            ("problem_id", ("cr-1", "d-1", "run-1", "humaneval", None, 1)),
        ],
    )
    def test_code_results_key_columns_not_null(
        self, tmp_db: Path, column: str, values: tuple[object, ...]
    ) -> None:
        """design_id/run_id/benchmark/problem_id are NOT NULL (matching siblings)."""
        with Database(tmp_db) as db:
            db.init_schema()
            db.conn.execute(
                """INSERT INTO runs (run_id, variant_id, task_id, repetition, status,
                   variant_config, task_config, model_configs)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                ("run-1", "v1", "t1", 1, "completed", "{}", "{}", "{}"),
            )
            db.conn.execute(
                """INSERT INTO designs (design_id, run_id, round, agent_role,
                   agent_id, full_text)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                ("d-1", "run-1", 0, "designer", "leader", "text"),
            )
            with pytest.raises(sqlite3.IntegrityError, match=f"NOT NULL.*{column}"):
                db.conn.execute(
                    """INSERT INTO code_results (result_id, design_id, run_id, benchmark,
                       problem_id, passed)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    values,
                )


class TestCodeResultsMigration:
    @staticmethod
    def _build_v3_db(path: Path) -> None:
        """Create a database resembling schema version 3 (no code_results table)."""
        conn = sqlite3.connect(str(path))
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(
            """
            CREATE TABLE schema_version (
                version     INTEGER PRIMARY KEY,
                applied_at  TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE runs (
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
                seed            INTEGER,
                UNIQUE(variant_id, task_id, repetition)
            );
            CREATE TABLE designs (
                design_id   TEXT PRIMARY KEY,
                run_id      TEXT NOT NULL REFERENCES runs(run_id),
                round       INTEGER NOT NULL,
                agent_role  TEXT NOT NULL,
                agent_id    TEXT NOT NULL,
                full_text   TEXT NOT NULL,
                token_count INTEGER,
                is_final    BOOLEAN DEFAULT FALSE,
                created_at  TEXT DEFAULT (datetime('now'))
            );
            """
        )
        conn.execute(
            """INSERT INTO runs (run_id, variant_id, task_id, repetition, status,
               variant_config, task_config, model_configs)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("run-1", "v1", "t1", 1, "completed", "{}", "{}", "{}"),
        )
        conn.execute(
            """INSERT INTO designs (design_id, run_id, round, agent_role,
               agent_id, full_text)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("d-1", "run-1", 0, "designer", "leader", "text"),
        )
        conn.execute("INSERT INTO schema_version (version) VALUES (3)")
        conn.commit()
        conn.close()

    def test_v3_db_lacks_code_results_before_upgrade(self, tmp_db: Path) -> None:
        self._build_v3_db(tmp_db)
        conn = sqlite3.connect(str(tmp_db))
        try:
            assert not _table_exists(conn, "code_results")
        finally:
            conn.close()

    def test_init_schema_upgrades_v3_to_v4(self, tmp_db: Path) -> None:
        """init_schema on a v3 DB adds code_results and bumps the version to 4."""
        self._build_v3_db(tmp_db)

        with Database(tmp_db) as db:
            assert db.get_schema_version() == 3

            db.init_schema()

            assert db.get_schema_version() == SCHEMA_VERSION == 4
            assert _table_exists(db.conn, "code_results")
            assert _index_exists(db.conn, "idx_code_results_run")
            assert _index_exists(db.conn, "idx_code_results_benchmark")

    def test_migration_preserves_existing_rows(self, tmp_db: Path) -> None:
        """Existing data survives the v3 -> v4 upgrade and code_results is usable."""
        self._build_v3_db(tmp_db)

        with Database(tmp_db) as db:
            db.init_schema()

            assert db.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
            assert db.conn.execute("SELECT COUNT(*) FROM designs").fetchone()[0] == 1

            # The migrated table accepts rows referencing the preserved design/run.
            db.conn.execute(
                """INSERT INTO code_results (result_id, design_id, run_id, benchmark,
                   problem_id, passed)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                ("cr-1", "d-1", "run-1", "humaneval", "HE/1", 1),
            )
            db.conn.commit()
            assert db.conn.execute("SELECT COUNT(*) FROM code_results").fetchone()[0] == 1

    def test_init_schema_is_idempotent_on_v4(self, tmp_db: Path) -> None:
        """Running init_schema twice does not error or duplicate the version."""
        self._build_v3_db(tmp_db)

        with Database(tmp_db) as db:
            db.init_schema()
            db.init_schema()
            assert db.get_schema_version() == SCHEMA_VERSION
            assert _table_exists(db.conn, "code_results")
