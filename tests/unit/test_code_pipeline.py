"""Tests for the synchronous code evaluation pipeline.

These run against the *real* collaborators that ship in the Paper-#2 build:
``consortium.agents.code_extract.extract_code`` for extraction, the real
``TaskConfig`` / ``CodingProblemConfig`` for reconstruction, the v4
``code_results`` schema, and the real ``execution`` harness registry. Only the
benchmark runner is a stand-in: a fake :class:`HarnessRunner` registered via the
public ``register_runner`` helper so pass/fail outcomes are deterministic.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from consortium.evaluation.code_pipeline import CodeEvaluationPipeline
from consortium.execution import (
    ExecLimits,
    ExecResult,
    HarnessRunner,
    LocalSandbox,
    Sandbox,
    SandboxResult,
    register_runner,
)
from consortium.execution import harness as _harness
from consortium.storage.database import Database

if TYPE_CHECKING:
    from pathlib import Path

_FAKE_BENCH = "fake-bench"

_GOOD_FULL_TEXT = "Here is my solution:\n```python\nprint('hi')\n```\n"

_PASS_RESULT = ExecResult(
    passed=True,
    n_pass=3,
    n_total=3,
    error_type="none",
    exec_ms=4.2,
    stdout_tail="all good",
    detail="ok",
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


class _FakeSandbox(Sandbox):
    """A sandbox the fake runner never touches; ``.image`` feeds harness_meta."""

    image = "fake-image:1"

    def run(self, **_: Any) -> SandboxResult:  # pragma: no cover - must not be called
        raise AssertionError("the fake runner must not invoke the sandbox")


@pytest.fixture
def fake_runner() -> Any:
    """Register a controllable fake runner under ``fake-bench``.

    Yields a mutable state dict: set ``state["result"]`` to control the
    returned :class:`ExecResult` and read ``state["calls"]`` to inspect what the
    pipeline passed to ``evaluate``.
    """
    state: dict[str, Any] = {"result": _PASS_RESULT, "calls": []}

    class FakeBenchRunner(HarnessRunner):
        benchmark = _FAKE_BENCH

        def evaluate(
            self, problem: Any, solution_code: str, sandbox: Any, limits: Any = None
        ) -> ExecResult:
            state["calls"].append(
                {
                    "problem": problem,
                    "solution_code": solution_code,
                    "sandbox": sandbox,
                    "limits": limits,
                }
            )
            return state["result"]

    register_runner(FakeBenchRunner())
    yield state
    _harness._RUNNERS.pop(_FAKE_BENCH, None)


# ── Seeding helpers ──────────────────────────────────────────────────────────


def _make_db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "code.db")
    db.init_schema()
    return db


def _seed_run(
    db: Database,
    run_id: str,
    task_config: dict[str, Any],
    *,
    status: str = "completed",
) -> None:
    db.conn.execute(
        """INSERT INTO runs (run_id, variant_id, sub_variant, task_id, repetition,
           status, variant_config, task_config, model_configs)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_id, "v1", None, task_config["id"], 1, status, "{}", json.dumps(task_config), "{}"),
    )
    db.conn.commit()


def _seed_design(
    db: Database,
    design_id: str,
    run_id: str,
    full_text: str,
    *,
    is_final: int = 1,
) -> None:
    db.conn.execute(
        """INSERT INTO designs (design_id, run_id, round, agent_role, agent_id,
           full_text, token_count, is_final)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (design_id, run_id, 1, "leader", "a1", full_text, 10, is_final),
    )
    db.conn.commit()


def _coding_task(
    task_id: str,
    *,
    problem_id: str = "P1",
    benchmark: str = _FAKE_BENCH,
    visible_tests: str = "assert True",
) -> dict[str, Any]:
    return {
        "id": task_id,
        "name": task_id,
        "task_type": "coding",
        "coding": {
            "id": problem_id,
            "benchmark": benchmark,
            "prompt": "solve the problem",
            "visible_tests": visible_tests,
        },
    }


def _count_results(db: Database) -> int:
    return db.conn.execute("SELECT COUNT(*) FROM code_results").fetchone()[0]


# ── Tests ────────────────────────────────────────────────────────────────────


def test_evaluate_scores_one_design(tmp_path: Path, fake_runner: dict[str, Any]) -> None:
    db = _make_db(tmp_path)
    _seed_run(db, "run-1", _coding_task("t-good", problem_id="HE/7", visible_tests="assert True"))
    _seed_design(db, "d-good", "run-1", _GOOD_FULL_TEXT)

    sandbox = _FakeSandbox()
    limits = ExecLimits()
    pipeline = CodeEvaluationPipeline(db, sandbox=sandbox, limits=limits)
    stats = pipeline.evaluate()

    assert stats == {"total": 1, "scored": 1, "failed": 0, "passed": 1}

    rows = db.conn.execute(
        """SELECT design_id, run_id, benchmark, problem_id, passed, n_pass,
           n_total, error_type, exec_ms, harness_meta FROM code_results"""
    ).fetchall()
    assert len(rows) == 1
    row = rows[0]
    assert row["design_id"] == "d-good"
    assert row["run_id"] == "run-1"
    assert row["benchmark"] == _FAKE_BENCH
    assert row["problem_id"] == "HE/7"
    assert row["passed"] == 1
    assert row["n_pass"] == 3
    assert row["n_total"] == 3
    assert row["error_type"] == "none"
    assert row["exec_ms"] == pytest.approx(4.2)

    meta = json.loads(row["harness_meta"])
    assert meta == {
        "runner": "FakeBenchRunner",
        "image_or_version": "fake-image:1",
        "reportable": False,
        "detail": "ok",
    }

    # The pipeline extracted the fenced code and forwarded the problem (carrying
    # its visible tests), the configured sandbox, and the limits to the runner.
    assert len(fake_runner["calls"]) == 1
    call = fake_runner["calls"][0]
    assert call["solution_code"] == "print('hi')"
    assert call["problem"].visible_tests == "assert True"
    assert call["sandbox"] is sandbox
    assert call["limits"] is limits


def test_dry_run_writes_nothing(tmp_path: Path, fake_runner: dict[str, Any]) -> None:
    db = _make_db(tmp_path)
    _seed_run(db, "run-1", _coding_task("t-good"))
    _seed_design(db, "d-good", "run-1", _GOOD_FULL_TEXT)

    pipeline = CodeEvaluationPipeline(db, sandbox=_FakeSandbox())
    stats = pipeline.evaluate(dry_run=True)

    assert stats == {"total": 1, "scored": 0, "failed": 0, "passed": 0}
    assert _count_results(db) == 0
    assert fake_runner["calls"] == []


def test_force_replaces_existing_row(tmp_path: Path, fake_runner: dict[str, Any]) -> None:
    db = _make_db(tmp_path)
    _seed_run(db, "run-1", _coding_task("t-good"))
    _seed_design(db, "d-good", "run-1", _GOOD_FULL_TEXT)

    pipeline = CodeEvaluationPipeline(db, sandbox=_FakeSandbox())

    first = pipeline.evaluate()
    assert first == {"total": 1, "scored": 1, "failed": 0, "passed": 1}
    assert _count_results(db) == 1

    # Without --force the already-scored design is skipped entirely.
    skipped = pipeline.evaluate()
    assert skipped == {"total": 0, "scored": 0, "failed": 0, "passed": 0}
    assert _count_results(db) == 1

    # A re-run under --force replaces the row in place (stays one row, not two)
    # and reflects the new result.
    fake_runner["result"] = ExecResult(
        passed=False,
        n_pass=1,
        n_total=3,
        error_type="wrong_answer",
        stdout_tail="boom",
        exec_ms=9.0,
    )
    forced = pipeline.evaluate(force=True)
    assert forced == {"total": 1, "scored": 1, "failed": 0, "passed": 0}
    assert _count_results(db) == 1

    row = db.conn.execute(
        "SELECT passed, n_pass, error_type FROM code_results WHERE design_id = ?",
        ("d-good",),
    ).fetchone()
    assert row["passed"] == 0
    assert row["n_pass"] == 1
    assert row["error_type"] == "wrong_answer"


def test_malformed_coding_counts_failed_without_aborting(
    tmp_path: Path, fake_runner: dict[str, Any]
) -> None:
    db = _make_db(tmp_path)

    # A well-formed coding design...
    _seed_run(db, "run-good", _coding_task("t-good", problem_id="HE/1"))
    _seed_design(db, "d-good", "run-good", _GOOD_FULL_TEXT)

    # ...alongside one tagged coding but missing its problem spec.
    _seed_run(db, "run-bad", {"id": "t-bad", "name": "t-bad", "task_type": "coding"})
    _seed_design(db, "d-bad", "run-bad", _GOOD_FULL_TEXT)

    pipeline = CodeEvaluationPipeline(db, sandbox=_FakeSandbox())
    stats = pipeline.evaluate()

    # The malformed row is counted as failed but does not abort the batch:
    # the good design is still scored.
    assert stats == {"total": 2, "scored": 1, "failed": 1, "passed": 1}
    assert _count_results(db) == 1

    scored_ids = [r["design_id"] for r in db.conn.execute("SELECT design_id FROM code_results")]
    assert scored_ids == ["d-good"]


def test_empty_extracted_code_counts_failed(
    tmp_path: Path, fake_runner: dict[str, Any]
) -> None:
    db = _make_db(tmp_path)
    _seed_run(db, "run-1", _coding_task("t-good"))
    # Whitespace-only design text -> extractor yields nothing executable.
    _seed_design(db, "d-empty", "run-1", "   \n  ")

    pipeline = CodeEvaluationPipeline(db, sandbox=_FakeSandbox())
    stats = pipeline.evaluate()

    assert stats == {"total": 1, "scored": 0, "failed": 1, "passed": 0}
    assert _count_results(db) == 0
    # The runner is never invoked when there is no code to execute.
    assert fake_runner["calls"] == []


def test_default_sandbox_is_unset(tmp_path: Path) -> None:
    """Omitting ``sandbox`` leaves it unset — never an implicit host sandbox.

    LocalSandbox runs code on the host with no isolation, so it is resolved
    lazily and only behind the opt-in guard (see the refuse/opt-in tests below).
    """
    db = _make_db(tmp_path)
    pipeline = CodeEvaluationPipeline(db)
    assert pipeline.sandbox is None


def test_evaluate_refuses_local_sandbox_without_optin(
    tmp_path: Path, fake_runner: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """With real work and no explicit sandbox or opt-in, evaluate() refuses to
    run model code on the host rather than silently falling back to LocalSandbox."""
    monkeypatch.delenv("CONSORTIUM_ALLOW_LOCAL_EXEC", raising=False)
    db = _make_db(tmp_path)
    _seed_run(db, "run-1", _coding_task("t-good"))
    _seed_design(db, "d-good", "run-1", _GOOD_FULL_TEXT)

    pipeline = CodeEvaluationPipeline(db)
    with pytest.raises(RuntimeError, match="LocalSandbox"):
        pipeline.evaluate()

    assert _count_results(db) == 0
    assert fake_runner["calls"] == []


def test_no_designs_does_not_require_optin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty selection returns cleanly without resolving/guarding a sandbox."""
    monkeypatch.delenv("CONSORTIUM_ALLOW_LOCAL_EXEC", raising=False)
    db = _make_db(tmp_path)  # no runs seeded
    pipeline = CodeEvaluationPipeline(db)
    assert pipeline.evaluate() == {"total": 0, "scored": 0, "failed": 0, "passed": 0}


def test_dry_run_does_not_require_optin(
    tmp_path: Path, fake_runner: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """dry_run reports the selection without resolving/guarding a sandbox."""
    monkeypatch.delenv("CONSORTIUM_ALLOW_LOCAL_EXEC", raising=False)
    db = _make_db(tmp_path)
    _seed_run(db, "run-1", _coding_task("t-good"))
    _seed_design(db, "d-good", "run-1", _GOOD_FULL_TEXT)

    pipeline = CodeEvaluationPipeline(db)  # no opt-in
    assert pipeline.evaluate(dry_run=True) == {
        "total": 1,
        "scored": 0,
        "failed": 0,
        "passed": 0,
    }


def test_allow_local_sandbox_opt_in_resolves_and_scores(
    tmp_path: Path, fake_runner: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """allow_local_sandbox=True permits the LocalSandbox default and scores."""
    monkeypatch.delenv("CONSORTIUM_ALLOW_LOCAL_EXEC", raising=False)
    db = _make_db(tmp_path)
    _seed_run(db, "run-1", _coding_task("t-good"))
    _seed_design(db, "d-good", "run-1", _GOOD_FULL_TEXT)

    pipeline = CodeEvaluationPipeline(db, allow_local_sandbox=True)
    assert isinstance(pipeline._resolve_sandbox(), LocalSandbox)
    assert pipeline.evaluate() == {"total": 1, "scored": 1, "failed": 0, "passed": 1}

    # LocalSandbox carries no image tag, so harness_meta records None.
    meta = json.loads(
        db.conn.execute("SELECT harness_meta FROM code_results").fetchone()["harness_meta"]
    )
    assert meta["image_or_version"] is None


def test_env_var_opt_in_allows_local_sandbox(
    tmp_path: Path, fake_runner: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """CONSORTIUM_ALLOW_LOCAL_EXEC=1 is an equivalent opt-in to the constructor arg."""
    monkeypatch.setenv("CONSORTIUM_ALLOW_LOCAL_EXEC", "1")
    db = _make_db(tmp_path)
    _seed_run(db, "run-1", _coding_task("t-good"))
    _seed_design(db, "d-good", "run-1", _GOOD_FULL_TEXT)

    pipeline = CodeEvaluationPipeline(db)
    assert pipeline.evaluate() == {"total": 1, "scored": 1, "failed": 0, "passed": 1}


def test_malformed_task_config_json_is_skipped(
    tmp_path: Path, fake_runner: dict[str, Any]
) -> None:
    """A row whose task_config is not valid JSON is filtered out by json_valid
    rather than aborting the whole selection query (json_extract would raise)."""
    db = _make_db(tmp_path)
    # A well-formed coding run...
    _seed_run(db, "run-good", _coding_task("t-good", problem_id="P-good"))
    _seed_design(db, "d-good", "run-good", _GOOD_FULL_TEXT)
    # ...alongside one whose task_config JSON is corrupt (e.g. a bad `db load`).
    db.conn.execute(
        """INSERT INTO runs (run_id, variant_id, sub_variant, task_id, repetition,
           status, variant_config, task_config, model_configs)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("run-bad", "v1", None, "t-bad", 1, "completed", "{}", "{not valid json", "{}"),
    )
    db.conn.commit()
    _seed_design(db, "d-bad", "run-bad", _GOOD_FULL_TEXT)

    pipeline = CodeEvaluationPipeline(db, sandbox=_FakeSandbox())
    stats = pipeline.evaluate()

    # The malformed-config row is invisible to selection; only the good one scores.
    assert stats == {"total": 1, "scored": 1, "failed": 0, "passed": 1}
    scored = [r["design_id"] for r in db.conn.execute("SELECT design_id FROM code_results")]
    assert scored == ["d-good"]


# ── Selection scope: --run-id ────────────────────────────────────────────────


def test_run_id_scopes_scoring_to_one_run(
    tmp_path: Path, fake_runner: dict[str, Any]
) -> None:
    """evaluate(run_id=...) scores only that run's final design."""
    db = _make_db(tmp_path)
    _seed_run(db, "run-1", _coding_task("t1", problem_id="P1"))
    _seed_design(db, "d1", "run-1", _GOOD_FULL_TEXT)
    _seed_run(db, "run-2", _coding_task("t2", problem_id="P2"))
    _seed_design(db, "d2", "run-2", _GOOD_FULL_TEXT)

    pipeline = CodeEvaluationPipeline(db, sandbox=_FakeSandbox())
    stats = pipeline.evaluate(run_id="run-1")

    assert stats == {"total": 1, "scored": 1, "failed": 0, "passed": 1}
    scored = [r["design_id"] for r in db.conn.execute("SELECT design_id FROM code_results")]
    assert scored == ["d1"]


def test_run_id_with_force_rescopes_to_one_run(
    tmp_path: Path, fake_runner: dict[str, Any]
) -> None:
    """force=True combined with run_id re-scores only that run, leaving others intact."""
    db = _make_db(tmp_path)
    _seed_run(db, "run-1", _coding_task("t1", problem_id="P1"))
    _seed_design(db, "d1", "run-1", _GOOD_FULL_TEXT)
    _seed_run(db, "run-2", _coding_task("t2", problem_id="P2"))
    _seed_design(db, "d2", "run-2", _GOOD_FULL_TEXT)

    pipeline = CodeEvaluationPipeline(db, sandbox=_FakeSandbox())
    assert pipeline.evaluate() == {"total": 2, "scored": 2, "failed": 0, "passed": 2}
    assert _count_results(db) == 2

    # Re-score under --force but scoped to run-1: only run-1's design is touched.
    fake_runner["calls"].clear()
    forced = pipeline.evaluate(run_id="run-1", force=True)
    assert forced == {"total": 1, "scored": 1, "failed": 0, "passed": 1}
    assert _count_results(db) == 2
    assert len(fake_runner["calls"]) == 1
    assert fake_runner["calls"][0]["problem"].id == "P1"


# ── Selection filters: negative cases ────────────────────────────────────────


def test_design_type_run_excluded(tmp_path: Path, fake_runner: dict[str, Any]) -> None:
    """A non-coding (design) run is never selected for code scoring."""
    db = _make_db(tmp_path)
    _seed_run(db, "run-d", {"id": "t-design", "name": "t-design", "task_type": "design"})
    _seed_design(db, "d-design", "run-d", _GOOD_FULL_TEXT)

    pipeline = CodeEvaluationPipeline(db, sandbox=_FakeSandbox())
    assert pipeline.evaluate() == {"total": 0, "scored": 0, "failed": 0, "passed": 0}
    assert _count_results(db) == 0
    assert fake_runner["calls"] == []


def test_non_final_design_excluded(tmp_path: Path, fake_runner: dict[str, Any]) -> None:
    """A non-final coding design is never selected (only is_final = TRUE)."""
    db = _make_db(tmp_path)
    _seed_run(db, "run-1", _coding_task("t1"))
    _seed_design(db, "d-draft", "run-1", _GOOD_FULL_TEXT, is_final=0)

    pipeline = CodeEvaluationPipeline(db, sandbox=_FakeSandbox())
    assert pipeline.evaluate() == {"total": 0, "scored": 0, "failed": 0, "passed": 0}
    assert _count_results(db) == 0


def test_non_completed_run_excluded(tmp_path: Path, fake_runner: dict[str, Any]) -> None:
    """A coding run that is not completed is excluded from the default (all-runs) scope."""
    db = _make_db(tmp_path)
    _seed_run(db, "run-1", _coding_task("t1"), status="running")
    _seed_design(db, "d1", "run-1", _GOOD_FULL_TEXT)

    pipeline = CodeEvaluationPipeline(db, sandbox=_FakeSandbox())
    assert pipeline.evaluate() == {"total": 0, "scored": 0, "failed": 0, "passed": 0}
    assert _count_results(db) == 0


# ── Error path: runner lookup / execution failure ────────────────────────────


def test_unregistered_benchmark_fails_without_aborting(
    tmp_path: Path, fake_runner: dict[str, Any]
) -> None:
    """A benchmark with no registered runner makes get_runner raise; the pipeline
    rolls back that design, counts it failed, and keeps scoring the batch."""
    db = _make_db(tmp_path)
    # A good design on the registered fake bench...
    _seed_run(db, "run-good", _coding_task("t-good", problem_id="P-good"))
    _seed_design(db, "d-good", "run-good", _GOOD_FULL_TEXT)
    # ...alongside one whose benchmark has no registered runner (KeyError in get_runner).
    _seed_run(
        db, "run-bad", _coding_task("t-bad", problem_id="P-bad", benchmark="no-such-bench")
    )
    _seed_design(db, "d-bad", "run-bad", _GOOD_FULL_TEXT)

    pipeline = CodeEvaluationPipeline(db, sandbox=_FakeSandbox())
    stats = pipeline.evaluate()

    assert stats == {"total": 2, "scored": 1, "failed": 1, "passed": 1}
    scored = [r["design_id"] for r in db.conn.execute("SELECT design_id FROM code_results")]
    assert scored == ["d-good"]
