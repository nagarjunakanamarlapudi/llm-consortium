"""v2 — Leader + Reviewers: centralized design with review feedback loops."""

from __future__ import annotations

import asyncio

from consortium.config.models import TaskConfig
from consortium.orchestrator.context import DesignArtifact, RunContext
from consortium.orchestrator.variants.base import VariantOrchestrator


class V2LeaderReviewersOrchestrator(VariantOrchestrator):
    """Centralized leader generates a design; homogeneous reviewers critique it.

    Workflow:
        Round 0: leader generates initial design
        Rounds 1..N: reviewers critique → leader revises with feedback
    """

    async def execute(self, task: TaskConfig, context: RunContext) -> DesignArtifact:
        leader = self._get_agent("leader")
        reviewers = self._get_agents("reviewers")
        rubric_dims = context.rubric_dimensions
        sees_others = self.config.workflow.reviewer_sees_other_reviews

        # Round 0: initial generation
        self._log.info("round_start", round=0, step="generation")
        design = await leader.act(
            context=context, round_num=0, task=task, rubric_dimensions=rubric_dims,
        )

        # Review-revise rounds
        for round_num in range(1, self.config.workflow.max_rounds):
            # Reviewers critique
            self._log.info("round_start", round=round_num, step="review")
            if sees_others:
                # Sequential: each reviewer sees previous reviews
                reviews = []
                for reviewer in reviewers:
                    review = await reviewer.act(
                        context=context,
                        round_num=round_num,
                        task=task,
                        design=design,
                        rubric_dimensions=rubric_dims,
                        other_reviews=reviews,
                    )
                    reviews.append(review)
            else:
                # Parallel: independent reviews
                reviews = list(await asyncio.gather(*(
                    reviewer.act(
                        context=context,
                        round_num=round_num,
                        task=task,
                        design=design,
                        rubric_dimensions=rubric_dims,
                    )
                    for reviewer in reviewers
                )))

            # Leader revises with feedback
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
