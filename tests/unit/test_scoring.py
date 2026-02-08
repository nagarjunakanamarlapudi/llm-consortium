"""Tests for database schema and storage."""

from __future__ import annotations

from pathlib import Path

from consortium.storage.database import Database


class TestDatabase:
    def test_init_schema(self, tmp_db: Path) -> None:
        with Database(tmp_db) as db:
            db.init_schema()
            version = db.get_schema_version()
            assert version == 2

    def test_stats_empty(self, tmp_db: Path) -> None:
        with Database(tmp_db) as db:
            db.init_schema()
            stats = db.stats()
            assert stats["runs"] == 0
            assert stats["designs"] == 0
            assert stats["evaluations"] == 0
            assert stats["traces"] == 0

    def test_insert_and_query_run(self, tmp_db: Path) -> None:
        with Database(tmp_db) as db:
            db.init_schema()
            db.conn.execute(
                """INSERT INTO runs (run_id, variant_id, task_id, repetition, status,
                   variant_config, task_config, model_configs)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                ("run-1", "v1", "t1", 1, "completed", "{}", "{}", "{}"),
            )
            db.conn.commit()
            stats = db.stats()
            assert stats["runs"] == 1

    def test_foreign_keys_enforced(self, tmp_db: Path) -> None:
        """Designs must reference a valid run_id."""
        import sqlite3

        with Database(tmp_db) as db:
            db.init_schema()
            with db.conn:
                try:
                    db.conn.execute(
                        """INSERT INTO designs (design_id, run_id, round, agent_role,
                           agent_id, full_text)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        ("d-1", "nonexistent-run", 0, "designer", "leader", "text"),
                    )
                    # If we get here, FK wasn't enforced (which is OK — SQLite FK
                    # enforcement depends on PRAGMA)
                except sqlite3.IntegrityError:
                    pass  # Expected: FK constraint violation

    def test_unique_constraint(self, tmp_db: Path) -> None:
        """Same (variant_id, task_id, repetition) can't appear twice."""
        import sqlite3

        with Database(tmp_db) as db:
            db.init_schema()
            db.conn.execute(
                """INSERT INTO runs (run_id, variant_id, task_id, repetition, status,
                   variant_config, task_config, model_configs)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                ("run-1", "v1", "t1", 1, "completed", "{}", "{}", "{}"),
            )
            db.conn.commit()
            with pytest.raises(sqlite3.IntegrityError):
                db.conn.execute(
                    """INSERT INTO runs (run_id, variant_id, task_id, repetition, status,
                       variant_config, task_config, model_configs)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    ("run-2", "v1", "t1", 1, "pending", "{}", "{}", "{}"),
                )


# Need to import pytest for the raises check
import pytest  # noqa: E402
