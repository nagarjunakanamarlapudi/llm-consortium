"""Adversarial reviewer agent — accept/reject gate for designs."""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime
from typing import Any

import structlog

from consortium.agents.base import BaseAgent
from consortium.config.models import RubricDimensionConfig, TaskConfig
from consortium.orchestrator.context import (
    ConversationTurn,
    DesignArtifact,
    ReviewArtifact,
    RunContext,
)

logger = structlog.get_logger()

_VERDICT_PATTERN = re.compile(r"VERDICT:\s*(ACCEPT|REJECT)", re.IGNORECASE)


def parse_verdict(text: str) -> str:
    """Extract ACCEPT or REJECT verdict from response text.

    Returns ``"accept"`` or ``"reject"``. Defaults to ``"reject"`` if the
    verdict cannot be parsed (fail-closed: unparseable responses are treated
    as rejections to avoid silently passing low-quality designs).
    """
    match = _VERDICT_PATTERN.search(text)
    if match:
        return match.group(1).lower()
    logger.warning("verdict_unparseable_defaulting_reject", text_length=len(text))
    return "reject"


class AdversarialReviewer(BaseAgent):
    """Agent that acts as a binary accept/reject gate for designs.

    Parses ``VERDICT: ACCEPT`` or ``VERDICT: REJECT`` from the LLM response.
    Used in v4 adversarial variant.
    """

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
        """Review a design with an accept/reject verdict.

        Args:
            context: Current run context.
            round_num: Current workflow round.
            task: The design task configuration.
            design: The design document to judge.
            rubric_dimensions: Rubric dimensions for quality threshold check.
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
        }

        response = await self._call_llm(
            template=self.prompt_template,
            template_vars=template_vars,
            context=context,
            step="review",
            round_num=round_num,
        )

        verdict = parse_verdict(response.content)

        review = ReviewArtifact(
            review_id=uuid.uuid4().hex,
            run_id=context.run_id,
            design_id=design.design_id,
            round=round_num,
            agent_role=self.role,
            agent_id=self.agent_id,
            review_text=response.content,
            verdict=verdict,
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

        self._log.info("adversarial_verdict", round=round_num, verdict=verdict)
        return review
