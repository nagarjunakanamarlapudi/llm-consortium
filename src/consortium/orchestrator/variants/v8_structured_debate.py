"""v8 — Structured Debate: debaters argue positions, judge synthesizes."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Any

from consortium.config.models import TaskConfig
from consortium.orchestrator.context import (
    ConversationTurn,
    DesignArtifact,
    RunContext,
)
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
        rubric_dims = context.rubric_dimensions
        rebuttal_template = self.config.workflow.rebuttal_template

        # Assign perspectives to debaters
        perspectives = ["performance-first", "simplicity-first", "resilience-first"]
        while len(perspectives) < len(debaters):
            perspectives.append(f"perspective-{len(perspectives) + 1}")

        # Round 0: position statements — use position_assignment template directly
        self._log.info("round_start", round=0, step="position", count=len(debaters))
        designs = list(await asyncio.gather(*(
            self._position_statement(
                debater=debaters[i],
                context=context,
                task=task,
                rubric_dims=rubric_dims,
                perspective=perspectives[i],
            )
            for i in range(len(debaters))
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
                    other_perspectives=[p for j, p in enumerate(perspectives) if j != i],
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

    async def _position_statement(
        self,
        *,
        debater,
        context: RunContext,
        task: TaskConfig,
        rubric_dims,
        perspective: str,
    ) -> DesignArtifact:
        """Generate an initial position statement using position_assignment template."""
        template_vars: dict[str, Any] = {
            "system_name": task.variables.system_name,
            "complexity": task.complexity,
            "design_type": task.design_type,
            "position": perspective,
            "perspective": perspective,
            "hard_constraints": task.variables.hard_constraints,
            "use_cases": task.variables.use_cases,
            "rubric_dimensions": (
                [d.model_dump() for d in rubric_dims] if rubric_dims else None
            ),
        }

        response = await debater._call_llm(
            template=debater.prompt_template,  # position_assignment.j2
            template_vars=template_vars,
            context=context,
            step="position",
            round_num=0,
        )

        design = DesignArtifact(
            design_id=uuid.uuid4().hex,
            run_id=context.run_id,
            round=0,
            agent_role=debater.role,
            agent_id=debater.agent_id,
            full_text=response.content,
            token_count=response.output_tokens,
            is_final=False,
            created_at=datetime.now(UTC),
        )

        context.designs.append(design)
        context.conversation_history.append(
            ConversationTurn(
                round=0,
                agent_role=debater.role,
                agent_id=debater.agent_id,
                step="position",
                content=response.content,
                timestamp=datetime.now(UTC),
            )
        )

        return design

    async def _rebuttal(
        self,
        *,
        debater,
        context: RunContext,
        round_num: int,
        task: TaskConfig,
        own_design: DesignArtifact,
        other_designs: list[DesignArtifact],
        other_perspectives: list[str],
        rubric_dims,
        perspective: str,
        rebuttal_template: str | None,
    ) -> DesignArtifact:
        """Generate a rebuttal: revise own position considering others' arguments."""
        template = rebuttal_template or debater.prompt_template

        # Build template vars matching rebuttal.j2's expected structure
        other_positions = [
            {
                "position": other_perspectives[i],
                "perspective": other_perspectives[i],
                "design": d.full_text,
            }
            for i, d in enumerate(other_designs)
        ]

        template_vars: dict[str, Any] = {
            "round": round_num,
            "design_type": task.design_type,
            "position": perspective,
            "perspective": perspective,
            "own_design": own_design.full_text,
            "other_positions": other_positions,
            "system_name": task.variables.system_name,
            "complexity": task.complexity,
            "rubric_dimensions": (
                [d.model_dump() for d in rubric_dims] if rubric_dims else None
            ),
        }

        response = await debater._call_llm(
            template=template,
            template_vars=template_vars,
            context=context,
            step="rebuttal",
            round_num=round_num,
        )

        design = DesignArtifact(
            design_id=uuid.uuid4().hex,
            run_id=context.run_id,
            round=round_num,
            agent_role=debater.role,
            agent_id=debater.agent_id,
            full_text=response.content,
            token_count=response.output_tokens,
            is_final=False,
            created_at=datetime.now(UTC),
        )

        context.designs.append(design)
        context.conversation_history.append(
            ConversationTurn(
                round=round_num,
                agent_role=debater.role,
                agent_id=debater.agent_id,
                step="rebuttal",
                content=response.content,
                timestamp=datetime.now(UTC),
            )
        )

        return design
