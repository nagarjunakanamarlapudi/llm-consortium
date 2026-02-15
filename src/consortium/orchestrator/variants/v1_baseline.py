"""v1 — Single designer self-refinement baseline."""

from __future__ import annotations

from consortium.agents.reviewer import ReviewerAgent
from consortium.config.models import TaskConfig
from consortium.orchestrator.context import DesignArtifact, RunContext
from consortium.orchestrator.variants.base import VariantOrchestrator


class V1BaselineOrchestrator(VariantOrchestrator):
    """Baseline: one designer generates and iteratively self-refines.

    Workflow per the implementation plan:
        Round 0: generate initial design
        Rounds 1..N:
            a) Self-review: same model critiques own design against rubric
            b) Revise: incorporate self-feedback into improved design
    """

    async def execute(self, task: TaskConfig, context: RunContext) -> DesignArtifact:
        designer = self._get_agent("designer")
        rubric_dims = context.rubric_dimensions

        # Round 0: initial generation
        self._log.info("round_start", round=0, step="generation")
        design = await designer.act(
            context=context, round_num=0, task=task, rubric_dimensions=rubric_dims,
        )

        # Single-shot mode (v1a): skip all refinement when max_rounds == 0
        if self.config.workflow.max_rounds == 0:
            design.is_final = True
            self._log.info("complete_single_shot", design_id=design.design_id)
            return design

        # Build a self-reviewer using the same model as the designer.
        # v1 is a single-agent baseline: the same LLM reviews its own work.
        self_reviewer = ReviewerAgent(
            agent_id="self_reviewer",
            role="reviewer",
            model_config=designer.model_config,
            provider=designer.provider,
            renderer=designer.renderer,
            prompt_template="review/general_review.j2",
            parameters=designer.parameters,
            limits=designer.limits,
        )

        # Self-refinement rounds: review → revise
        for round_num in range(1, self.config.workflow.max_rounds):
            # Step a: self-review against rubric
            self._log.info("round_start", round=round_num, step="self_review")
            review = await self_reviewer.act(
                context=context,
                round_num=round_num,
                task=task,
                design=design,
                rubric_dimensions=rubric_dims,
            )

            # Step b: revise design incorporating self-feedback
            self._log.info("round_start", round=round_num, step="revision")
            design = await designer.act(
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
