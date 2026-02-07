"""v8 — Structured Debate: debaters argue positions, judge synthesizes."""

from __future__ import annotations

import asyncio

from consortium.config.models import TaskConfig
from consortium.orchestrator.context import DesignArtifact, RunContext
from consortium.orchestrator.variants.base import VariantOrchestrator


class V8StructuredDebateOrchestrator(VariantOrchestrator):
    """Debaters argue for competing approaches; a judge synthesizes the best.

    Workflow:
        Round 0: debaters assigned positions → generate position designs
        Rounds 1..N-1: rebuttals — each debater sees others' designs
        Final round: judge evaluates all positions and produces ruling
    """

    async def execute(self, task: TaskConfig, context: RunContext) -> DesignArtifact:
        debaters = self._get_agents("debaters")
        judge = self._get_agent("judge")
        rubric_dims = context._rubric_dimensions  # type: ignore[attr-defined]
        rebuttal_template = self.config.workflow.rebuttal_template

        # Assign perspectives to debaters
        perspectives = ["performance-first", "simplicity-first", "resilience-first"]
        while len(perspectives) < len(debaters):
            perspectives.append(f"perspective-{len(perspectives) + 1}")

        # Round 0: position statements
        self._log.info("round_start", round=0, step="position", count=len(debaters))
        designs = list(await asyncio.gather(*(
            debater.act(
                context=context,
                round_num=0,
                task=task,
                rubric_dimensions=rubric_dims,
                perspective=perspectives[i],
            )
            for i, debater in enumerate(debaters)
        )))

        # Rebuttal rounds
        max_rebuttal_rounds = max(self.config.workflow.max_rounds - 1, 1)
        for round_num in range(1, max_rebuttal_rounds):
            self._log.info("round_start", round=round_num, step="rebuttal")

            new_designs = list(await asyncio.gather(*(
                self._rebuttal(
                    debater=debaters[i],
                    context=context,
                    round_num=round_num,
                    task=task,
                    own_design=designs[i],
                    other_designs=[d for j, d in enumerate(designs) if j != i],
                    rubric_dims=rubric_dims,
                    perspective=perspectives[i],
                    rebuttal_template=rebuttal_template,
                )
                for i in range(len(debaters))
            )))

            designs = new_designs

        # Final: judge ruling
        final_round = self.config.workflow.max_rounds
        self._log.info("round_start", round=final_round, step="judge")

        positions = [
            {
                "position": perspectives[i],
                "perspective": perspectives[i],
                "final_design": designs[i].full_text,
            }
            for i in range(len(debaters))
        ]

        ruling = await judge.act(
            context=context,
            round_num=final_round,
            task=task,
            positions=positions,
            rubric_dimensions=rubric_dims,
        )

        ruling.is_final = True
        self._log.info("complete", design_id=ruling.design_id)
        return ruling

    async def _rebuttal(
        self,
        *,
        debater,
        context: RunContext,
        round_num: int,
        task: TaskConfig,
        own_design: DesignArtifact,
        other_designs: list[DesignArtifact],
        rubric_dims,
        perspective: str,
        rebuttal_template: str | None,
    ) -> DesignArtifact:
        """Generate a rebuttal: revise own position considering others' arguments."""
        from consortium.orchestrator.context import ReviewArtifact
        from datetime import UTC, datetime
        import uuid

        # Convert other designs to review-like feedback
        reviews = [
            ReviewArtifact(
                review_id=uuid.uuid4().hex,
                run_id=context.run_id,
                design_id=d.design_id,
                round=d.round,
                agent_role=d.agent_role,
                agent_id=d.agent_id,
                review_text=f"[Opposing position from {d.agent_id}]\n\n{d.full_text}",
                created_at=datetime.now(UTC),
            )
            for d in other_designs
        ]

        # Use rebuttal template if available, otherwise use debater's own template
        template = rebuttal_template or debater.prompt_template

        original_template = debater.prompt_template
        debater.prompt_template = template
        try:
            design = await debater.act(
                context=context,
                round_num=round_num,
                task=task,
                rubric_dimensions=rubric_dims,
                previous_design=own_design,
                reviews=reviews,
                perspective=perspective,
            )
        finally:
            debater.prompt_template = original_template

        return design
