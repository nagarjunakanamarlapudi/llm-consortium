"""v6 — Rotating Leader: participants take turns leading and reviewing."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime

from consortium.config.models import TaskConfig
from consortium.orchestrator.context import (
    ConversationTurn,
    DesignArtifact,
    ReviewArtifact,
    RunContext,
)
from consortium.orchestrator.variants.base import VariantOrchestrator


class V6RotatingLeaderOrchestrator(VariantOrchestrator):
    """Participants rotate leadership; non-leaders review each round.

    Workflow:
        Round 0: participant[0] generates; others review
        Round 1: participant[1] revises with reviews; others review
        Round N: participant[N % K] leads…
    """

    async def execute(self, task: TaskConfig, context: RunContext) -> DesignArtifact:
        participants = self._get_agents("participants")
        rubric_dims = context._rubric_dimensions  # type: ignore[attr-defined]
        review_template = self.config.workflow.review_template

        design: DesignArtifact | None = None
        pending_reviews: list[ReviewArtifact] = []

        for round_num in range(self.config.workflow.max_rounds):
            leader_idx = round_num % len(participants)
            leader = participants[leader_idx]
            reviewers = [p for i, p in enumerate(participants) if i != leader_idx]

            if design is None:
                # First round: generate
                self._log.info(
                    "round_start", round=round_num, step="generation",
                    leader=leader.agent_id,
                )
                design = await leader.act(
                    context=context,
                    round_num=round_num,
                    task=task,
                    rubric_dimensions=rubric_dims,
                )
            else:
                # Subsequent rounds: leader revises with previous round's reviews
                self._log.info(
                    "round_start", round=round_num, step="revision",
                    leader=leader.agent_id,
                )
                design = await leader.act(
                    context=context,
                    round_num=round_num,
                    task=task,
                    rubric_dimensions=rubric_dims,
                    previous_design=design,
                    reviews=pending_reviews if pending_reviews else None,
                )

            # Non-leaders review the current design
            self._log.info(
                "round_start", round=round_num, step="review",
                reviewer_count=len(reviewers),
            )
            pending_reviews = list(await asyncio.gather(*(
                self._review_as_designer(
                    reviewer, context, round_num, task, design, rubric_dims,
                    review_template,
                )
                for reviewer in reviewers
            )))

        assert design is not None
        design.is_final = True
        self._log.info("complete", design_id=design.design_id)
        return design

    @staticmethod
    async def _review_as_designer(
        agent, context, round_num, task, design, rubric_dims, review_template,
    ) -> ReviewArtifact:
        """Use a DesignerAgent to produce review feedback via the review template."""
        template = review_template or agent.prompt_template

        template_vars = {
            "design_text": design.full_text,
            "system_name": task.variables.system_name,
            "complexity": task.complexity,
            "rubric_dimensions": (
                [d.model_dump() for d in rubric_dims] if rubric_dims else None
            ),
            "other_reviews": None,
        }

        response = await agent._call_llm(
            template=template,
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
            agent_role=agent.role,
            agent_id=agent.agent_id,
            review_text=response.content,
            created_at=datetime.now(UTC),
        )

        context.reviews.append(review)
        context.conversation_history.append(
            ConversationTurn(
                round=round_num,
                agent_role=agent.role,
                agent_id=agent.agent_id,
                step="review",
                content=response.content,
                timestamp=datetime.now(UTC),
            )
        )

        return review
