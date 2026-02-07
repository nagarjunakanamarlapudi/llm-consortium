"""Designer agent — generates and revises design documents."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from consortium.agents.base import BaseAgent
from consortium.config.models import RubricDimensionConfig, TaskConfig
from consortium.orchestrator.context import (
    ConversationTurn,
    DesignArtifact,
    ReviewArtifact,
    RunContext,
)


class DesignerAgent(BaseAgent):
    """Agent that generates or revises design documents.

    Used as the designer in v1, leader in v2/v4/v5, parallel leader in v3,
    participant in v6/v7, and debater in v8.
    """

    async def act(
        self,
        *,
        context: RunContext,
        round_num: int,
        task: TaskConfig,
        rubric_dimensions: list[RubricDimensionConfig] | None = None,
        previous_design: DesignArtifact | None = None,
        reviews: list[ReviewArtifact] | None = None,
        perspective: str | None = None,
        **kwargs: Any,
    ) -> DesignArtifact:
        """Generate or revise a design document.

        Args:
            context: Current run context.
            round_num: Current workflow round.
            task: The design task configuration.
            rubric_dimensions: Rubric dimensions for evaluation-aware generation.
            previous_design: Previous design to revise (None for initial generation).
            reviews: Review feedback to incorporate during revision.
            perspective: Optional diversity perspective for parallel generation.
        """
        template_vars: dict[str, Any] = {
            "system_name": task.variables.system_name,
            "problem_statement": task.variables.problem_statement,
            "hard_constraints": task.variables.hard_constraints,
            "use_cases": task.variables.use_cases,
            "complexity_drivers": task.variables.complexity_drivers,
            "rubric_dimensions": (
                [d.model_dump() for d in rubric_dimensions]
                if rubric_dimensions
                else None
            ),
            "review_feedback": (
                [r.review_text for r in reviews] if reviews else None
            ),
            "previous_design": (
                previous_design.full_text if previous_design else None
            ),
        }

        step = "revision" if previous_design else "generation"
        response = await self._call_llm(
            template=self.prompt_template,
            template_vars=template_vars,
            context=context,
            step=step,
            round_num=round_num,
        )

        design = DesignArtifact(
            design_id=uuid.uuid4().hex,
            run_id=context.run_id,
            round=round_num,
            agent_role=self.role,
            agent_id=self.agent_id,
            full_text=response.content,
            token_count=response.output_tokens,
            is_final=False,
            created_at=datetime.now(UTC),
        )

        context.designs.append(design)
        context.conversation_history.append(
            ConversationTurn(
                round=round_num,
                agent_role=self.role,
                agent_id=self.agent_id,
                step=step,
                content=response.content,
                timestamp=datetime.now(UTC),
            )
        )

        return design
