"""v5 — Specialist Panel: domain experts review a centralized design."""

from __future__ import annotations

import asyncio

from consortium.config.models import TaskConfig
from consortium.orchestrator.context import DesignArtifact, RunContext
from consortium.orchestrator.variants.base import VariantOrchestrator


class V5SpecialistPanelOrchestrator(VariantOrchestrator):
    """Leader generates; specialist reviewers critique from domain expertise.

    Workflow:
        Round 0: leader generates initial design
        Rounds 1..N: specialists review in parallel → leader revises
    """

    async def execute(self, task: TaskConfig, context: RunContext) -> DesignArtifact:
        leader = self._get_agent("leader")
        specialists = self._get_agents("specialists")
        rubric_dims = context.rubric_dimensions

        # Round 0: initial generation
        self._log.info("round_start", round=0, step="generation")
        design = await leader.act(
            context=context, round_num=0, task=task, rubric_dimensions=rubric_dims,
        )

        # Specialist review-revise rounds
        for round_num in range(1, self.config.workflow.max_rounds):
            # Specialists review in parallel
            self._log.info(
                "round_start", round=round_num, step="review",
                specialist_count=len(specialists),
            )
            reviews = list(await asyncio.gather(*(
                specialist.act(
                    context=context,
                    round_num=round_num,
                    task=task,
                    design=design,
                    rubric_dimensions=rubric_dims,
                )
                for specialist in specialists
            )))

            # Leader revises with specialist feedback
            self._log.info("round_start", round=round_num, step="revision")
            design = await leader.act(
                context=context,
                round_num=round_num,
                task=task,
                rubric_dimensions=rubric_dims,
                previous_design=design,
                reviews=reviews,
            )

        design.is_final = True
        self._log.info("complete", design_id=design.design_id)
        return design
