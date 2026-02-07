"""v3 — Parallel Leaders + Merge: diverse generation then synthesis."""

from __future__ import annotations

import asyncio

from consortium.config.models import TaskConfig
from consortium.orchestrator.context import DesignArtifact, RunContext
from consortium.orchestrator.variants.base import VariantOrchestrator


class V3ParallelMergeOrchestrator(VariantOrchestrator):
    """Multiple leaders generate diverse designs; merger synthesizes them.

    Workflow:
        Round 0: K parallel leaders generate designs (with diversity perspectives)
        Round 1: merger synthesizes into unified design
    """

    async def execute(self, task: TaskConfig, context: RunContext) -> DesignArtifact:
        parallel_leaders = self._get_agents("parallel_leaders")
        merger = self._get_agent("merger")
        rubric_dims = context.rubric_dimensions

        # Extract diversity perspectives from variant config
        diversity = self.config.agents.parallel_leaders
        perspectives: list[str | None] = [None] * len(parallel_leaders)
        if diversity and diversity.diversity and diversity.diversity.enabled:
            for i, p in enumerate(diversity.diversity.perspectives):
                if i < len(perspectives):
                    perspectives[i] = p

        # Round 0: parallel generation with diversity perspectives
        self._log.info("round_start", round=0, step="generation", count=len(parallel_leaders))
        designs = list(await asyncio.gather(*(
            leader.act(
                context=context,
                round_num=0,
                task=task,
                rubric_dimensions=rubric_dims,
                perspective=perspectives[i],
            )
            for i, leader in enumerate(parallel_leaders)
        )))

        # Round 1: merge
        self._log.info("round_start", round=1, step="merge")
        merged = await merger.act(
            context=context,
            round_num=1,
            task=task,
            designs=designs,
            rubric_dimensions=rubric_dims,
        )

        merged.is_final = True
        self._log.info("complete", design_id=merged.design_id)
        return merged
