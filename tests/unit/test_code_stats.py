"""Tests for objective-result statistics (pass@1 / pass@k / McNemar)."""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING

from consortium.analysis.code_stats import condition_results, mcnemar
from consortium.storage.database import Database

if TYPE_CHECKING:
    from pathlib import Path

_BENCH = "humanevalplus"


def _seed(db: Database, variant: str, problem: str, rep: int, passed: bool) -> None:
    run_id = uuid.uuid4().hex
    tc = {"id": problem, "name": problem, "task_type": "coding",
          "coding": {"id": problem, "benchmark": _BENCH, "prompt": "x"}}
    db.conn.execute(
        """INSERT INTO runs (run_id, variant_id, sub_variant, task_id, repetition,
           status, variant_config, task_config, model_configs)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (run_id, variant, None, problem, rep, "completed", "{}", json.dumps(tc), "{}"),
    )
    design_id = uuid.uuid4().hex
    db.conn.execute(
        """INSERT INTO designs (design_id, run_id, round, agent_role, agent_id,
           full_text, is_final) VALUES (?,?,?,?,?,?,1)""",
        (design_id, run_id, 0, "designer", "a1", "code"),
    )
    db.conn.execute(
        """INSERT INTO code_results (result_id, design_id, run_id, benchmark,
           problem_id, passed, n_pass, n_total) VALUES (?,?,?,?,?,?,?,1)""",
        (uuid.uuid4().hex, design_id, run_id, _BENCH, problem, passed, int(passed)),
    )
    db.conn.commit()


def test_pass_at_1_and_k(tmp_path: Path) -> None:
    db = Database(tmp_path / "s.db")
    db.init_schema()
    # condition A: 2 problems, p1 pass, p2 fail -> pass@1 = 50%
    _seed(db, "v1a_a", "P1", 0, passed=True)
    _seed(db, "v1a_a", "P2", 0, passed=False)
    results = {r.variant_id: r for r in condition_results(db, _BENCH)}
    a = results["v1a_a"]
    assert a.n_problems == 2
    assert a.pass_at_1 == 0.5
    assert a.pass_at_k == 0.5
    db.close()


def test_pass_at_k_uses_any_rep(tmp_path: Path) -> None:
    db = Database(tmp_path / "s.db")
    db.init_schema()
    # one problem, 2 reps: one pass one fail -> pass@1=50%, pass@k=100%
    _seed(db, "v1a_a", "P1", 0, passed=True)
    _seed(db, "v1a_a", "P1", 1, passed=False)
    a = {r.variant_id: r for r in condition_results(db, _BENCH)}["v1a_a"]
    assert a.pass_at_1 == 0.5
    assert a.pass_at_k == 1.0
    db.close()


def test_mcnemar_counts(tmp_path: Path) -> None:
    db = Database(tmp_path / "s.db")
    db.init_schema()
    # A passes P1,P2; B passes P1 only -> a_only=1 (P2), both=1 (P1)
    _seed(db, "v2b_x", "P1", 0, passed=True)
    _seed(db, "v2b_x", "P2", 0, passed=True)
    _seed(db, "v1a_b", "P1", 0, passed=True)
    _seed(db, "v1a_b", "P2", 0, passed=False)
    m = mcnemar(db, _BENCH, "v2b_x", "v1a_b")
    assert m["n_paired"] == 2
    assert m["both_pass"] == 1
    assert m["a_only"] == 1
    assert m["b_only"] == 0
    assert 0.0 <= m["p_value"] <= 1.0
    db.close()
