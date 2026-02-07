"""v1 — Single designer self-refinement baseline."""

from __future__ import annotations

from consortium.config.models import TaskConfig
from consortium.orchestrator.context import DesignArtifact, RunContext
from consortium.orchestrator.variants.base import VariantOrchestrator


class V1BaselineOrchestrator(VariantOrchestrator):
    """Baseline: one designer generates and iteratively self-refines.

    Workflow:
        Round 0: generate initial design
        Rounds 1..N: revise design (self-review via rubric in prompt)
    """

    async def execute(self, task: TaskConfig, context: RunContext) -> DesignArtifact:
        designer = self._get_agent("designer")
        rubric_dims = context._rubric_dimensions  # type: ignore[attr-defined]

        # Initial generation
        self._log.info("round_start", round=0, step="generation")
        design = await designer.act(
            context=context, round_num=0, task=task, rubric_dimensions=rubric_dims,
        )

        # Self-refinement rounds
        for round_num in range(1, self.config.workflow.max_rounds):
            self._log.info("round_start", round=round_num, step="revision")
            design = await designer.act(
                context=context,
                round_num=round_num,
                task=task,
                rubric_dimensions=rubric_dims,
                previous_design=design,
            )

        design.is_final = True
        self._log.info("complete", design_id=design.design_id)
        return design
