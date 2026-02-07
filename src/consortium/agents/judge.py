"""Judge agent — evaluates debate positions and produces a ruling."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from consortium.agents.base import BaseAgent
from consortium.config.models import RubricDimensionConfig, TaskConfig
from consortium.orchestrator.context import (
    ConversationTurn,
    DesignArtifact,
    RunContext,
)


class JudgeAgent(BaseAgent):
    """Agent that judges a structured debate and produces a final design.

    Evaluates multiple debaters' positions, picks a winner, and synthesizes
    the strongest elements into a final design. Used in v8 structured debate.
    """

    async def act(
        self,
        *,
        context: RunContext,
        round_num: int,
        task: TaskConfig,
        positions: list[dict[str, str]],
        rubric_dimensions: list[RubricDimensionConfig] | None = None,
        **kwargs: Any,
    ) -> DesignArtifact:
        """Judge debate positions and produce a final design.

        Args:
            context: Current run context.
            round_num: Current workflow round.
            task: The design task configuration.
            positions: List of dicts with keys ``position``, ``perspective``,
                ``final_design`` for each debater.
            rubric_dimensions: Rubric dimensions for evaluation.
        """
        template_vars: dict[str, Any] = {
            "positions": positions,
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
            step="judge",
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
                step="judge",
                content=response.content,
                timestamp=datetime.now(UTC),
            )
        )

        return design
