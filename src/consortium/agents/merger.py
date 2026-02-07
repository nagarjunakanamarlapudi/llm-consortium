"""Merger agent — synthesizes multiple designs into one."""

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


class MergerAgent(BaseAgent):
    """Agent that synthesizes multiple design documents into a unified design.

    Used in v3 parallel merge and optionally in v8 structured debate.
    The merge strategy (naive, rubric-guided, dialectical) is determined by
    the prompt template specified in the variant config.
    """

    async def act(
        self,
        *,
        context: RunContext,
        round_num: int,
        task: TaskConfig,
        designs: list[DesignArtifact],
        rubric_dimensions: list[RubricDimensionConfig] | None = None,
        **kwargs: Any,
    ) -> DesignArtifact:
        """Merge multiple designs into a single synthesized design.

        Args:
            context: Current run context.
            round_num: Current workflow round.
            task: The design task configuration.
            designs: The design documents to merge.
            rubric_dimensions: Rubric dimensions for guided merging.
        """
        template_vars: dict[str, Any] = {
            "designs": [
                {"agent_id": d.agent_id, "text": d.full_text} for d in designs
            ],
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
            step="merge",
            round_num=round_num,
        )

        merged = DesignArtifact(
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

        context.designs.append(merged)
        context.conversation_history.append(
            ConversationTurn(
                round=round_num,
                agent_role=self.role,
                agent_id=self.agent_id,
                step="merge",
                content=response.content,
                timestamp=datetime.now(UTC),
            )
        )

        return merged
