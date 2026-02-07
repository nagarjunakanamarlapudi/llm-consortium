"""Orchestrator engine — top-level runner for experiment executions."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import structlog

from consortium.config.models import FullConfig
from consortium.orchestrator.context import DesignArtifact, RunContext
from consortium.orchestrator.factory import create_variant_orchestrator, instantiate_agents
from consortium.prompts.renderer import PromptRenderer
from consortium.storage.database import Database

logger = structlog.get_logger()


class OrchestratorEngine:
    """Top-level runner that executes a single variant×task×repetition run.

    Handles:
        - Config resolution (variant, task, rubric, models)
        - Agent instantiation via factory
        - Variant orchestrator execution
        - Database persistence of all artifacts and traces
        - Checkpoint/resume for crash recovery
    """

    def __init__(
        self,
        config: FullConfig,
        database: Database,
        prompts_dir: str | Path,
    ) -> None:
        self.config = config
        self.database = database
        self.renderer = PromptRenderer(prompts_dir)

    def _find_existing_run(
        self, variant_id: str, task_id: str, repetition: int,
    ) -> tuple[str | None, str | None]:
        """Check if a completed or in-progress run already exists.

        Returns:
            (status, run_id) — status is None if no run exists.
        """
        row = self.database.conn.execute(
            "SELECT run_id, status FROM runs "
            "WHERE variant_id = ? AND task_id = ? AND repetition = ?",
            (variant_id, task_id, repetition),
        ).fetchone()
        if row is None:
            return None, None
        return row["status"], row["run_id"]

    async def run(
        self,
        variant_id: str,
        task_id: str,
        repetition: int,
        *,
        resume: bool = False,
        force: bool = False,
    ) -> DesignArtifact:
        """Execute a single experiment run.

        Args:
            variant_id: Which variant to execute (v1-v8).
            task_id: Which design task to use.
            repetition: Repetition index (0-based).
            resume: If True, resume a previously failed/incomplete run.
            force: If True, overwrite a previously completed run.

        Returns:
            The final design artifact with ``is_final=True``.

        Raises:
            RuntimeError: If token/cost limits are exceeded during execution.
            KeyError: If variant, task, model, or rubric config is not found.
        """
        variant_config = self.config.get_variant(variant_id)
        task_config = self.config.get_task(task_id)
        rubric_config = self.config.get_rubric(task_config.rubric)
        limits = self.config.experiment.limits

        # Check for existing run
        existing_status, existing_run_id = self._find_existing_run(
            variant_id, task_id, repetition,
        )

        if existing_status == "completed" and not force:
            msg = (
                f"Run already completed: {existing_run_id}. "
                "Use --force to overwrite."
            )
            raise RuntimeError(msg)

        if existing_status in ("running", "failed") and not resume and not force:
            msg = (
                f"Run exists with status '{existing_status}': {existing_run_id}. "
                "Use --resume to continue or --force to restart."
            )
            raise RuntimeError(msg)

        if force and existing_run_id:
            self._delete_run(existing_run_id)

        # Generate unique run ID
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        run_id = f"{variant_id}-{task_id}-rep{repetition}-{timestamp}"

        # Create run context
        context = RunContext(
            run_id=run_id,
            variant_id=variant_id,
            task_id=task_id,
            repetition=repetition,
            started_at=datetime.now(UTC),
            status="running",
        )

        # Attach rubric dimensions to context for orchestrators to access
        context.rubric_dimensions = list(rubric_config.dimensions)

        log = logger.bind(
            run_id=run_id, variant=variant_id, task=task_id, rep=repetition,
        )
        log.info("run_start")

        # Persist initial run record for checkpoint tracking
        self._persist_run_start(context, variant_config, task_config)

        try:
            # Instantiate agents
            agents = instantiate_agents(
                variant_config=variant_config,
                full_config=self.config,
                renderer=self.renderer,
                limits=limits,
            )

            # Create and execute the variant orchestrator
            orchestrator = create_variant_orchestrator(variant_config, agents)
            design = await orchestrator.execute(task_config, context)

            context.status = "completed"
            context.ended_at = datetime.now(UTC)

            log.info(
                "run_complete",
                tokens_in=context.total_input_tokens,
                tokens_out=context.total_output_tokens,
                cost=f"${context.total_cost_usd:.4f}",
                traces=len(context.traces),
            )

        except Exception:
            context.status = "failed"
            context.ended_at = datetime.now(UTC)
            import traceback
            context.error = traceback.format_exc()
            log.error("run_failed", error=context.error)
            self._persist_run(context, variant_config, task_config)
            raise

        # Persist all artifacts
        self._persist_run(context, variant_config, task_config)

        return design

    def _delete_run(self, run_id: str) -> None:
        """Delete an existing run and all associated data."""
        conn = self.database.conn
        conn.execute("DELETE FROM traces WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM reviews WHERE run_id = ?", (run_id,))
        # Delete evaluations and scores for designs in this run
        design_ids = [
            row[0] for row in conn.execute(
                "SELECT design_id FROM designs WHERE run_id = ?", (run_id,)
            ).fetchall()
        ]
        for did in design_ids:
            conn.execute("DELETE FROM evaluations WHERE design_id = ?", (did,))
            conn.execute("DELETE FROM scores_median WHERE design_id = ?", (did,))
        conn.execute("DELETE FROM designs WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        conn.commit()
        logger.info("run_deleted", run_id=run_id)

    def _persist_run_start(self, context: RunContext, variant_config, task_config) -> None:
        """Persist the initial run record so checkpointing works even if we crash."""
        conn = self.database.conn
        conn.execute(
            """INSERT OR REPLACE INTO runs (
                run_id, variant_id, sub_variant, task_id, repetition,
                status, variant_config, task_config, model_configs,
                total_input_tokens, total_output_tokens, total_cost_usd,
                started_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                context.run_id,
                context.variant_id,
                variant_config.sub_variant,
                context.task_id,
                context.repetition,
                "running",
                variant_config.model_dump_json(),
                task_config.model_dump_json(),
                "{}",
                0, 0, 0.0,
                context.started_at.isoformat() if context.started_at else None,
            ),
        )
        conn.commit()

    def _persist_run(self, context: RunContext, variant_config, task_config) -> None:
        """Persist run, designs, reviews, and traces to the database."""
        conn = self.database.conn

        # Compute duration
        duration = None
        if context.started_at and context.ended_at:
            duration = (context.ended_at - context.started_at).total_seconds()

        # Collect model configs used in this run
        model_ids = {t.model_config_id for t in context.traces}
        model_configs = {}
        for mid in model_ids:
            try:
                model_configs[mid] = self.config.get_model(mid).model_dump()
            except KeyError:
                pass

        # INSERT run
        conn.execute(
            """INSERT OR REPLACE INTO runs (
                run_id, variant_id, sub_variant, task_id, repetition,
                status, variant_config, task_config, model_configs,
                total_input_tokens, total_output_tokens, total_cost_usd,
                started_at, ended_at, duration_seconds,
                checkpoint_step, checkpoint_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                context.run_id,
                context.variant_id,
                variant_config.sub_variant,
                context.task_id,
                context.repetition,
                context.status,
                variant_config.model_dump_json(),
                task_config.model_dump_json(),
                json.dumps(model_configs),
                context.total_input_tokens,
                context.total_output_tokens,
                context.total_cost_usd,
                context.started_at.isoformat() if context.started_at else None,
                context.ended_at.isoformat() if context.ended_at else None,
                duration,
                context.checkpoint_step,
                context.checkpoint_data,
            ),
        )

        # INSERT designs
        for d in context.designs:
            conn.execute(
                """INSERT OR REPLACE INTO designs (
                    design_id, run_id, round, agent_role, agent_id,
                    full_text, token_count, is_final, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    d.design_id, d.run_id, d.round, d.agent_role,
                    d.agent_id, d.full_text, d.token_count, d.is_final,
                    d.created_at.isoformat() if d.created_at else None,
                ),
            )

        # INSERT reviews
        for r in context.reviews:
            conn.execute(
                """INSERT OR REPLACE INTO reviews (
                    review_id, run_id, design_id, round, agent_role,
                    agent_id, review_text, verdict, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    r.review_id, r.run_id, r.design_id, r.round,
                    r.agent_role, r.agent_id, r.review_text, r.verdict,
                    r.created_at.isoformat() if r.created_at else None,
                ),
            )

        # INSERT traces
        for t in context.traces:
            conn.execute(
                """INSERT OR REPLACE INTO traces (
                    trace_id, run_id, variant_id, task_id, repetition,
                    agent_role, agent_id, step, round,
                    model_config_id, api_model, provider,
                    system_prompt_hash, prompt_template,
                    input_tokens, output_tokens, cached_input_tokens,
                    cost_usd, latency_ms,
                    started_at, ended_at,
                    batch_id, status, error, retry_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    t.trace_id, t.run_id, t.variant_id, t.task_id,
                    t.repetition, t.agent_role, t.agent_id, t.step,
                    t.round, t.model_config_id, t.api_model, t.provider,
                    t.system_prompt_hash, t.prompt_template,
                    t.input_tokens, t.output_tokens, t.cached_input_tokens,
                    t.cost_usd, t.latency_ms,
                    t.started_at.isoformat(), t.ended_at.isoformat(),
                    t.batch_id, t.status, t.error, t.retry_count,
                ),
            )

        conn.commit()
        logger.info(
            "run_persisted",
            run_id=context.run_id,
            designs=len(context.designs),
            reviews=len(context.reviews),
            traces=len(context.traces),
        )


class ExperimentRunner:
    """Orchestrates the full experiment matrix: variants × tasks × repetitions.

    Handles skip-if-done logic, progress tracking, cost accounting,
    and checkpoint-based resumption after crashes.
    """

    def __init__(
        self,
        config: FullConfig,
        database: Database,
        prompts_dir: str | Path,
    ) -> None:
        self.config = config
        self.database = database
        self.engine = OrchestratorEngine(config, database, prompts_dir)

    def _get_completed_runs(self) -> set[tuple[str, str, int]]:
        """Get set of (variant_id, task_id, repetition) for completed runs."""
        rows = self.database.conn.execute(
            "SELECT variant_id, task_id, repetition FROM runs WHERE status = 'completed'"
        ).fetchall()
        return {(r["variant_id"], r["task_id"], r["repetition"]) for r in rows}

    async def run_experiment(
        self,
        *,
        variants: list[str] | None = None,
        tasks: list[str] | None = None,
        repetitions: int | None = None,
        resume: bool = True,
        force: bool = False,
        dry_run: bool = False,
    ) -> dict[str, int]:
        """Execute the full experiment matrix.

        Args:
            variants: Override variant list (defaults to config).
            tasks: Override task list (defaults to config).
            repetitions: Override repetition count (defaults to config).
            resume: Skip completed runs (default True).
            force: Re-run even completed runs.
            dry_run: Just report what would be run.

        Returns:
            Summary dict with counts of completed, skipped, failed runs.
        """
        variant_ids = variants or [
            v.split("_")[0] if "_" in v else v
            for v in self.config.experiment.variants
        ]
        task_ids = tasks or [
            t.split("_")[0] if "_" in t else t
            for t in self.config.experiment.tasks
        ]
        reps = repetitions or self.config.experiment.repetitions

        # Build the run matrix
        run_matrix = [
            (vid, tid, rep)
            for vid in variant_ids
            for tid in task_ids
            for rep in range(reps)
        ]

        # Filter out completed runs if resuming
        completed = self._get_completed_runs() if resume and not force else set()
        pending = [r for r in run_matrix if r not in completed]

        log = logger.bind(
            total_runs=len(run_matrix),
            pending=len(pending),
            skipped=len(run_matrix) - len(pending),
        )
        log.info("experiment_start")

        if dry_run:
            log.info("dry_run_complete")
            return {
                "total": len(run_matrix),
                "pending": len(pending),
                "skipped": len(run_matrix) - len(pending),
                "completed": 0,
                "failed": 0,
            }

        stats = {"completed": 0, "failed": 0, "skipped": len(completed)}
        total_cost = 0.0

        for i, (vid, tid, rep) in enumerate(pending):
            run_log = log.bind(
                progress=f"{i + 1}/{len(pending)}",
                variant=vid, task=tid, rep=rep,
            )
            run_log.info("run_queued")

            try:
                design = await self.engine.run(
                    vid, tid, rep, resume=resume, force=force,
                )
                stats["completed"] += 1

                # Track cost from latest run
                row = self.database.conn.execute(
                    "SELECT total_cost_usd FROM runs WHERE run_id = "
                    "(SELECT run_id FROM designs WHERE design_id = ?)",
                    (design.design_id,),
                ).fetchone()
                if row:
                    total_cost += row["total_cost_usd"]

                # Check total cost limit
                limits = self.config.experiment.limits
                if total_cost > limits.max_total_cost_usd:
                    run_log.error(
                        "total_cost_limit_exceeded",
                        total_cost=f"${total_cost:.2f}",
                        limit=f"${limits.max_total_cost_usd:.2f}",
                    )
                    break

            except Exception as e:
                stats["failed"] += 1
                run_log.error("run_failed", error=str(e))
                # Continue to next run on failure

        log.info(
            "experiment_complete",
            total_cost=f"${total_cost:.2f}",
            **stats,
        )
        return {
            "total": len(run_matrix),
            "pending": len(pending),
            **stats,
        }
