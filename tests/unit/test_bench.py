"""Tests for the ``consortium bench`` CLI (Phase 0: ``score`` / ``list``).

These drive the Typer app end-to-end with ``CliRunner``. Phase 0 scores via the
host-subprocess ``LocalSandbox`` (dev/test only), which runs code on the host
with no isolation, so the ``score`` path requires an explicit opt-in
(``CONSORTIUM_ALLOW_LOCAL_EXEC=1``); without it the command refuses and exits 1.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from consortium.cli.bench import app as bench_app
from consortium.storage.database import Database

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

runner = CliRunner()

_FENCED_SOLUTION = "Here is my solution:\n```python\nprint('hi')\n```\n"


def _seed_coding_design(db_path: Path) -> None:
    """Seed one completed coding run with a final design carrying fenced code.

    The problem targets the built-in ``function-smoke`` runner with a trivially
    passing visible test, so a real LocalSandbox execution yields ``passed``.
    """
    task_config = {
        "id": "t1",
        "name": "t1",
        "task_type": "coding",
        "coding": {
            "id": "P1",
            "benchmark": "function-smoke",
            "prompt": "solve it",
            "visible_tests": "assert True",
        },
    }
    with Database(db_path) as db:
        db.init_schema()
        db.conn.execute(
            """INSERT INTO runs (run_id, variant_id, sub_variant, task_id, repetition,
               status, variant_config, task_config, model_configs)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("run-1", "v1", None, "t1", 1, "completed", "{}", json.dumps(task_config), "{}"),
        )
        db.conn.execute(
            """INSERT INTO designs (design_id, run_id, round, agent_role, agent_id,
               full_text, token_count, is_final)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("d1", "run-1", 1, "leader", "a1", _FENCED_SOLUTION, 10, 1),
        )
        db.conn.commit()


def test_bench_list_includes_function_smoke() -> None:
    result = runner.invoke(bench_app, ["list"])
    assert result.exit_code == 0
    assert "function-smoke" in result.output


def test_bench_score_dry_run_reports_without_writing(tmp_path: Path) -> None:
    db_path = tmp_path / "bench.db"
    _seed_coding_design(db_path)

    result = runner.invoke(bench_app, ["score", "--database", str(db_path), "--dry-run"])

    assert result.exit_code == 0
    assert "total" in result.output
    with Database(db_path) as db:
        assert db.conn.execute("SELECT COUNT(*) FROM code_results").fetchone()[0] == 0


def test_bench_score_runs_via_local_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the opt-in set, ``score`` executes the function-smoke runner in a LocalSandbox."""
    monkeypatch.setenv("CONSORTIUM_ALLOW_LOCAL_EXEC", "1")
    db_path = tmp_path / "bench.db"
    _seed_coding_design(db_path)

    result = runner.invoke(bench_app, ["score", "--database", str(db_path)])

    assert result.exit_code == 0
    with Database(db_path) as db:
        row = db.conn.execute(
            "SELECT passed, benchmark FROM code_results WHERE design_id = ?", ("d1",)
        ).fetchone()
    assert row is not None
    assert row["benchmark"] == "function-smoke"
    assert row["passed"] == 1


def test_bench_score_refuses_local_sandbox_without_optin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the opt-in, ``score`` refuses to run code on the host and exits 1."""
    monkeypatch.delenv("CONSORTIUM_ALLOW_LOCAL_EXEC", raising=False)
    db_path = tmp_path / "bench.db"
    _seed_coding_design(db_path)

    result = runner.invoke(bench_app, ["score", "--database", str(db_path)])

    assert result.exit_code == 1
    assert "failed" in result.output.lower()
    with Database(db_path) as db:
        assert db.conn.execute("SELECT COUNT(*) FROM code_results").fetchone()[0] == 0
