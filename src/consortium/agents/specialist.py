"""Specialist reviewer agent — domain-focused design critique."""

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


class SpecialistReviewer(BaseAgent):
    """Agent that provides domain-specialist critique of design documents.

    Each specialist focuses on specific rubric dimensions (e.g. security,
    scalability, reliability). Used in v5 specialist panel.
    """

    def __init__(
        self,
        *,
        specialty: str,
        focus_dimensions: list[str],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.specialty = specialty
        self.focus_dimensions = focus_dimensions

    async def act(
        self,
        *,
        context: RunContext,
        round_num: int,
        task: TaskConfig,
        design: DesignArtifact,
        rubric_dimensions: list[RubricDimensionConfig] | None = None,
        **kwargs: Any,
    ) -> ReviewArtifact:
        """Produce a specialist review focused on specific dimensions.

        Args:
            context: Current run context.
            round_num: Current workflow round.
            task: The design task configuration.
            design: The design document to review.
            rubric_dimensions: Full rubric (template filters to focus dims).
        """
        template_vars: dict[str, Any] = {
            "design_text": design.full_text,
            "system_name": task.variables.system_name,
            "complexity": task.complexity,
            "design_type": task.design_type,
            "specialty": self.specialty,
            "focus_dimensions": self.focus_dimensions,
            "rubric_dimensions": (
                [d.model_dump() for d in rubric_dimensions]
                if rubric_dimensions
                else None
            ),
        }

        response = await self._call_llm(
            template=self.prompt_template,
            template_vars=template_vars,
            context=context,
            step="review",
            round_num=round_num,
        )

        review = ReviewArtifact(
            review_id=uuid.uuid4().hex,
            run_id=context.run_id,
            design_id=design.design_id,
            round=round_num,
            agent_role=self.role,
            agent_id=self.agent_id,
            review_text=response.content,
            created_at=datetime.now(UTC),
        )

        context.reviews.append(review)
        context.conversation_history.append(
            ConversationTurn(
                round=round_num,
                agent_role=self.role,
                agent_id=self.agent_id,
                step="review",
                content=response.content,
                timestamp=datetime.now(UTC),
            )
        )

        return review
