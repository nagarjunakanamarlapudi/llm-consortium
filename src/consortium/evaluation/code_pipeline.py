"""Code evaluation pipeline — scores final coding designs via executable harnesses.

Paper #2 counterpart to :class:`~consortium.evaluation.pipeline.EvaluationPipeline`.
Where the design pipeline asks an LLM judge for a rubric score, this pipeline
extracts the model's solution from each final design and runs the benchmark's
*official* harness inside a sandbox, recording an objective pass/fail to the
``code_results`` table.

Selection mirrors ``EvaluationPipeline._get_unevaluated_designs`` /
``_get_all_final_designs`` (final, completed designs), additionally restricted to
runs whose stored ``task_config`` declares ``task_type == "coding"``.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import TYPE_CHECKING, Any

import structlog

from consortium.agents.code_extract import extract_code
from consortium.config.models import TaskConfig
from consortium.execution import LocalSandbox, get_runner

if TYPE_CHECKING:
    from consortium.execution.base import ExecLimits, Sandbox
    from consortium.storage.database import Database

logger = structlog.get_logger()

_LOCAL_EXEC_ENV = "CONSORTIUM_ALLOW_LOCAL_EXEC"


def _env_truthy(name: str) -> bool:
    """Whether environment variable *name* is set to a truthy value."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


class CodeEvaluationPipeline:
    """Scores final coding designs by executing them against benchmark harnesses.

    For each final design whose run is a coding task:

        1. Reconstruct the :class:`TaskConfig` from the run's stored
           ``task_config`` JSON and read its embedded coding problem.
        2. Extract the candidate solution from the design's full text.
        3. Look up the official harness runner for the problem's benchmark.
        4. Execute the solution in the configured sandbox and persist the
           objective result (pass/fail, counts, error type, timing) to
           ``code_results``.

    This pipeline is intentionally *synchronous*: harness execution is
    process/Docker bound rather than network bound, so there is no async LLM
    fan-out to coordinate.
    """

    def __init__(
        self,
        database: Database,
        *,
        sandbox: Sandbox | None = None,
        limits: ExecLimits | None = None,
        allow_local_sandbox: bool = False,
    ) -> None:
        self.database = database
        # Raw sandbox; may be None and is resolved lazily in evaluate() so that
        # dry runs and empty selections never construct or require one.
        self.sandbox = sandbox
        self.limits = limits
        self._allow_local_sandbox = allow_local_sandbox or _env_truthy(_LOCAL_EXEC_ENV)

    def _resolve_sandbox(self) -> Sandbox:
        """Return the sandbox to execute in, enforcing the local-exec guard.

        An explicit ``sandbox=`` is always honored. Otherwise we would fall back
        to ``LocalSandbox``, which runs code on the host with NO isolation, so we
        refuse unless explicitly allowed (``allow_local_sandbox=True`` or
        ``CONSORTIUM_ALLOW_LOCAL_EXEC=1``).

        CARRY-FORWARD (Phase 1): make the hardened ``DockerSandbox`` the default
        here before any model-generated code is executed.
        See CODE_CONSORTIUM_PAPER_PLAN.md §§4.3, 7.
        """
        if self.sandbox is not None:
            return self.sandbox
        if not self._allow_local_sandbox:
            msg = (
                "refusing to execute solutions with the unsandboxed LocalSandbox "
                "by default (it runs code on the host with no isolation). Pass an "
                "explicit sandbox=..., set allow_local_sandbox=True, or export "
                f"{_LOCAL_EXEC_ENV}=1. DockerSandbox becomes the default in Phase 1."
            )
            raise RuntimeError(msg)
        return LocalSandbox()

    # ── Selection ────────────────────────────────────────────────────────────

    def _select_coding_designs(
        self, run_id: str | None = None, *, unscored_only: bool
    ) -> list[dict[str, Any]]:
        """Final coding designs to score.

        Scope is a single run when *run_id* is given, otherwise every
        ``completed`` run. When *unscored_only* is set, designs already present
        in ``code_results`` are excluded (the default ``evaluate`` selection);
        ``--force`` clears it to re-score everything in scope.
        """
        scope = "r.run_id = ?" if run_id else "r.status = 'completed'"
        params: tuple[str, ...] = (run_id,) if run_id else ()
        unscored_clause = (
            " AND d.design_id NOT IN (SELECT design_id FROM code_results)"
            if unscored_only
            else ""
        )
        rows = self.database.conn.execute(
            "SELECT d.design_id, d.run_id, d.full_text, r.task_id, r.task_config "
            "FROM designs d "
            "JOIN runs r ON d.run_id = r.run_id "
            f"WHERE d.is_final = TRUE AND {scope} "
            # json_valid guards against a malformed task_config (e.g. a corrupt
            # `db load`) — json_extract would raise and abort the whole batch.
            "AND json_valid(r.task_config) "
            "AND json_extract(r.task_config, '$.task_type') = 'coding'"
            f"{unscored_clause}",
            params,
        ).fetchall()
        return [dict(row) for row in rows]

    # ── Evaluation ───────────────────────────────────────────────────────────

    def evaluate(
        self,
        *,
        run_id: str | None = None,
        force: bool = False,
        dry_run: bool = False,
    ) -> dict[str, int]:
        """Score final coding designs.

        Args:
            run_id: Score only this run's designs.
            force: Re-score already-scored designs (replaces their rows).
            dry_run: Report what would be scored without writing anything.

        Returns:
            Stats dict with ``total``, ``scored``, ``failed`` and ``passed``
            counts. ``scored + failed == total`` (except under ``dry_run``, which
            reports ``total`` only); ``passed <= scored``.
        """
        designs = self._select_coding_designs(run_id, unscored_only=not force)

        log = logger.bind(designs_to_score=len(designs), force=force)
        log.info("code_evaluation_start")

        stats = {"total": len(designs), "scored": 0, "failed": 0, "passed": 0}

        if dry_run:
            log.info("code_evaluation_dry_run", **stats)
            return stats

        if not designs:
            log.info("code_evaluation_complete", **stats)
            return stats

        # Resolve (and guard) the sandbox only once there is real work to run.
        sandbox = self._resolve_sandbox()
        conn = self.database.conn

        for design_row in designs:
            design_id = design_row["design_id"]
            d_run_id = design_row["run_id"]
            task_id = design_row["task_id"]
            full_text = design_row["full_text"]
            d_log = log.bind(design_id=design_id, task=task_id)

            try:
                task = TaskConfig.model_validate(json.loads(design_row["task_config"]))
                problem = task.coding

                # Guard: the run is tagged coding but carries no problem spec.
                if problem is None:
                    stats["failed"] += 1
                    d_log.warning("code_eval_no_problem")
                    continue

                code = extract_code(full_text)

                # Guard: nothing parseable to execute.
                if not (code and code.strip()):
                    stats["failed"] += 1
                    d_log.warning("code_eval_empty_code")
                    continue

                runner = get_runner(problem.benchmark)
                # HarnessRunner.evaluate(problem, solution_code, sandbox, limits)
                # -> ExecResult. The runner reads the problem's visible/example
                # tests; a benchmark's hidden official tests live inside the
                # runner itself.
                result = runner.evaluate(problem, code, sandbox, self.limits)

                passed = bool(result.passed)
                harness_meta = {
                    "runner": type(runner).__name__,
                    "image_or_version": getattr(sandbox, "image", None),
                    "reportable": runner.reportable,
                    "detail": result.detail,
                }

                # Under --force a prior result may exist; design_id is unique in
                # code_results, so remove it before inserting the replacement.
                if force:
                    conn.execute(
                        "DELETE FROM code_results WHERE design_id = ?",
                        (design_id,),
                    )

                conn.execute(
                    """INSERT INTO code_results (
                        result_id, design_id, run_id, benchmark, problem_id,
                        passed, n_pass, n_total, error_type, exec_ms, harness_meta
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        uuid.uuid4().hex,
                        design_id,
                        d_run_id,
                        problem.benchmark,
                        problem.id,
                        passed,
                        result.n_pass,
                        result.n_total,
                        result.error_type,
                        result.exec_ms,
                        json.dumps(harness_meta),
                    ),
                )
                conn.commit()

                stats["scored"] += 1
                if passed:
                    stats["passed"] += 1
                d_log.info(
                    "code_scored",
                    passed=passed,
                    benchmark=problem.benchmark,
                    error_type=result.error_type,
                )
                if not passed and result.stdout_tail:
                    d_log.debug("code_failed_output", stdout_tail=result.stdout_tail[-1000:])

            except Exception as e:
                conn.rollback()
                stats["failed"] += 1
                d_log.error("code_eval_failed", error=str(e), error_type=type(e).__name__)

        log.info("code_evaluation_complete", **stats)
        return stats
