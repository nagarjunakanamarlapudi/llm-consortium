"""v4 — Adversarial: leader vs. adversarial reviewer accept/reject gate."""

from __future__ import annotations

from consortium.config.models import TaskConfig
from consortium.orchestrator.context import DesignArtifact, RunContext
from consortium.orchestrator.variants.base import VariantOrchestrator


class V4AdversarialOrchestrator(VariantOrchestrator):
    """Leader generates; adversarial reviewer accepts or rejects.

    Workflow:
        Round 0: leader generates initial design
        Rounds 1..N: adversary reviews → if REJECT, leader revises; if ACCEPT, stop
    """

    async def execute(self, task: TaskConfig, context: RunContext) -> DesignArtifact:
        leader = self._get_agent("leader")
        adversary = self._get_agent("adversarial_reviewer")
        rubric_dims = context.rubric_dimensions

        # Round 0: initial generation
        self._log.info("round_start", round=0, step="generation")
        design = await leader.act(
            context=context, round_num=0, task=task, rubric_dimensions=rubric_dims,
        )

        # Adversarial review-revise rounds
        for round_num in range(1, self.config.workflow.max_rounds):
            # Adversary reviews
            self._log.info("round_start", round=round_num, step="review")
            review = await adversary.act(
                context=context,
                round_num=round_num,
                task=task,
                design=design,
                rubric_dimensions=rubric_dims,
            )

            if review.verdict == "accept":
                self._log.info("adversarial_accepted", round=round_num)
                break

            # Rejected — leader revises
            self._log.info("round_start", round=round_num, step="revision")
            design = await leader.act(
                context=context,
                round_num=round_num,
                task=task,
                rubric_dimensions=rubric_dims,
                previous_design=design,
                reviews=[review],
            )

        design.is_final = True
        self._log.info("complete", design_id=design.design_id)
        return design
