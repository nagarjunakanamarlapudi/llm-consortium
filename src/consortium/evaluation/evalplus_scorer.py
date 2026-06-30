"""Official EvalPlus scoring (HumanEval+/MBPP+) for objective pass@1.

This is the Phase-1 counterpart to the Phase-0 ``function-smoke`` runner: it
scores final code artifacts with the **official EvalPlus harness** so single-
model baselines reproduce published leaderboard pass@1.

Design:

* Final coding designs are grouped by ``(variant_id, repetition)`` — within a
  group every problem appears at most once, so one EvalPlus run yields a clean
  per-condition pass@1.
* EvalPlus asserts the samples cover the *whole* dataset, so missing generations
  are padded with an empty solution (which the harness scores as a fail).
* The harness runs inside the ``consortium-evalplus`` Docker image with
  ``--network none`` — Linux avoids the macOS ``reliability_guard`` crash and the
  disabled network contains the model-generated code under test.
* One ``code_results`` row is written per design (``passed`` = EvalPlus *plus*
  status, i.e. the hardened HumanEval+/MBPP+ metric).
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from consortium.agents.code_extract import extract_code
from consortium.config.models import TaskConfig

if TYPE_CHECKING:
    from consortium.storage.database import Database

logger = structlog.get_logger()

#: benchmark key -> EvalPlus ``--dataset`` value.
_DATASET = {"humanevalplus": "humaneval", "mbppplus": "mbpp"}

#: Per-batch wall-clock ceiling for the harness container (seconds).
_HARNESS_TIMEOUT_S = 1800.0


def _all_problem_ids(benchmark: str) -> list[str]:
    """Full ordered list of official problem ids for *benchmark*."""
    if benchmark == "humanevalplus":
        from evalplus.data import get_human_eval_plus

        return list(get_human_eval_plus().keys())
    if benchmark == "mbppplus":
        from evalplus.data import get_mbpp_plus

        return list(get_mbpp_plus().keys())
    msg = f"unsupported EvalPlus benchmark: {benchmark!r}"
    raise KeyError(msg)


class EvalPlusScorer:
    """Scores final coding designs with the official EvalPlus Docker harness."""

    def __init__(
        self,
        database: Database,
        *,
        image: str = "consortium-evalplus",
        docker_path: str = "docker",
    ) -> None:
        self.database = database
        self.image = image
        self.docker_path = docker_path

    # ── selection ──────────────────────────────────────────────────────────

    def _select(
        self, benchmark: str, run_id: str | None, *, force: bool
    ) -> list[dict[str, Any]]:
        scope = "r.run_id = ?" if run_id else "r.status = 'completed'"
        params: tuple[str, ...] = (run_id,) if run_id else ()
        unscored = (
            "" if force else " AND d.design_id NOT IN (SELECT design_id FROM code_results)"
        )
        rows = self.database.conn.execute(
            "SELECT d.design_id, d.run_id, d.full_text, r.variant_id, r.repetition, "
            "r.task_config FROM designs d JOIN runs r ON d.run_id = r.run_id "
            f"WHERE d.is_final = TRUE AND {scope} "
            "AND json_valid(r.task_config) "
            "AND json_extract(r.task_config, '$.task_type') = 'coding' "
            "AND json_extract(r.task_config, '$.coding.benchmark') = ?"
            f"{unscored}",
            (*params, benchmark),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── harness ────────────────────────────────────────────────────────────

    def _run_harness(
        self, dataset: str, benchmark: str, code_by_pid: dict[str, str]
    ) -> dict[str, dict[str, str]]:
        """Run EvalPlus over a full-coverage sample set; return per-problem status.

        Returns ``{problem_id: {"base_status": ..., "plus_status": ...}}``.
        """
        all_ids = _all_problem_ids(benchmark)
        with tempfile.TemporaryDirectory(prefix="consortium-evalplus-") as tmp:
            samples = Path(tmp) / "samples.jsonl"
            with samples.open("w", encoding="utf-8") as fh:
                for pid in all_ids:
                    sol = code_by_pid.get(pid) or "# no solution generated\n"
                    fh.write(json.dumps({"task_id": pid, "solution": sol}) + "\n")

            script = (
                "python -m evalplus.sanitize --samples /work/samples.jsonl && "
                f"python -m evalplus.evaluate --dataset {dataset} "
                "--samples /work/samples-sanitized.jsonl"
            )
            argv = [
                self.docker_path, "run", "--rm", "--network", "none",
                "-v", f"{tmp}:/work", self.image, "bash", "-lc", script,
            ]
            proc = subprocess.run(  # fixed argv, shell=False
                argv, capture_output=True, text=True, timeout=_HARNESS_TIMEOUT_S, check=False
            )
            results_path = Path(tmp) / "samples-sanitized_eval_results.json"
            if not results_path.exists():
                logger.error(
                    "evalplus_no_results",
                    returncode=proc.returncode,
                    stderr=proc.stderr[-1500:],
                )
                msg = f"EvalPlus produced no results (exit {proc.returncode})"
                raise RuntimeError(msg)
            data = json.loads(results_path.read_text(encoding="utf-8"))

        out: dict[str, dict[str, str]] = {}
        for pid, attempts in data["eval"].items():
            first = attempts[0] if attempts else {}
            out[pid] = {
                "base_status": first.get("base_status", "fail"),
                "plus_status": first.get("plus_status", "fail"),
            }
        return out

    # ── public API ─────────────────────────────────────────────────────────

    def score(
        self, benchmark: str, *, run_id: str | None = None, force: bool = False
    ) -> dict[str, int]:
        """Score final coding designs for *benchmark* via the EvalPlus harness."""
        if benchmark not in _DATASET:
            msg = f"EvalPlusScorer does not handle benchmark {benchmark!r}"
            raise KeyError(msg)
        dataset = _DATASET[benchmark]
        designs = self._select(benchmark, run_id, force=force)

        log = logger.bind(benchmark=benchmark, designs=len(designs))
        log.info("evalplus_scoring_start")
        stats = {"total": len(designs), "scored": 0, "failed": 0, "passed": 0}
        if not designs:
            log.info("evalplus_scoring_complete", **stats)
            return stats

        # Group by (variant_id, repetition): one sample per problem per group.
        groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for d in designs:
            coding = TaskConfig.model_validate(json.loads(d["task_config"])).coding
            if coding is None:  # selection filters to coding tasks, so this is unreachable
                continue
            d["problem_id"] = coding.id
            d["code"] = extract_code(d["full_text"])
            groups.setdefault((d["variant_id"], d["repetition"]), []).append(d)

        conn = self.database.conn
        for (vid, rep), items in groups.items():
            g_log = log.bind(variant=vid, rep=rep, n=len(items))
            try:
                code_by_pid = {it["problem_id"]: it["code"] for it in items}
                status = self._run_harness(dataset, benchmark, code_by_pid)
            except Exception as e:  # one group failing must not abort the rest
                stats["failed"] += len(items)
                g_log.error("evalplus_group_failed", error=str(e))
                continue

            for it in items:
                st = status.get(it["problem_id"], {"base_status": "fail", "plus_status": "fail"})
                passed = st["plus_status"] == "pass"
                meta = {
                    "runner": "EvalPlusScorer",
                    "image_or_version": self.image,
                    "reportable": True,
                    "dataset": dataset,
                    "base_status": st["base_status"],
                    "plus_status": st["plus_status"],
                }
                if force:
                    conn.execute(
                        "DELETE FROM code_results WHERE design_id = ?", (it["design_id"],)
                    )
                conn.execute(
                    """INSERT INTO code_results (
                        result_id, design_id, run_id, benchmark, problem_id,
                        passed, n_pass, n_total, error_type, exec_ms, harness_meta
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        uuid.uuid4().hex, it["design_id"], it["run_id"], benchmark,
                        it["problem_id"], passed, 1 if passed else 0, 1,
                        "none" if passed else "wrong_answer", None, json.dumps(meta),
                    ),
                )
                stats["scored"] += 1
                stats["passed"] += int(passed)
            conn.commit()
            g_log.info("evalplus_group_scored", passed=sum(
                1 for it in items
                if status.get(it["problem_id"], {}).get("plus_status") == "pass"
            ))

        log.info("evalplus_scoring_complete", **stats)
        return stats
