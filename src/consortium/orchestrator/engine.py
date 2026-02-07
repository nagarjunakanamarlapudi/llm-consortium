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

    async def run(
        self,
        variant_id: str,
        task_id: str,
        repetition: int,
    ) -> DesignArtifact:
        """Execute a single experiment run.

        Args:
            variant_id: Which variant to execute (v1-v8).
            task_id: Which design task to use.
            repetition: Repetition index (0-based).

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
        context._rubric_dimensions = rubric_config.dimensions  # type: ignore[attr-defined]

        log = logger.bind(
            run_id=run_id, variant=variant_id, task=task_id, rep=repetition,
        )
        log.info("run_start")

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
                started_at, ended_at, duration_seconds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
