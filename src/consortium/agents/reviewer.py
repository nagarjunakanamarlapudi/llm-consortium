"""Reviewer agent — critiques design documents."""

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


class ReviewerAgent(BaseAgent):
    """Agent that provides qualitative critique of design documents.

    Used as general reviewer in v2, and as review role in v6 rotating leader.
    """

    async def act(
        self,
        *,
        context: RunContext,
        round_num: int,
        task: TaskConfig,
        design: DesignArtifact,
        rubric_dimensions: list[RubricDimensionConfig] | None = None,
        other_reviews: list[ReviewArtifact] | None = None,
        **kwargs: Any,
    ) -> ReviewArtifact:
        """Review a design document.

        Args:
            context: Current run context.
            round_num: Current workflow round.
            task: The design task configuration.
            design: The design document to review.
            rubric_dimensions: Rubric dimensions for structured critique.
            other_reviews: Other reviews visible to this reviewer (if configured).
        """
        template_vars: dict[str, Any] = {
            "design_text": design.full_text,
            "system_name": task.variables.system_name,
            "complexity": task.complexity,
            "design_type": task.design_type,
            "rubric_dimensions": (
                [d.model_dump() for d in rubric_dimensions]
                if rubric_dimensions
                else None
            ),
            "other_reviews": (
                [r.review_text for r in other_reviews] if other_reviews else None
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
