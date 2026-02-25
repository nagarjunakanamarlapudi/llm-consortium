"""v7 — Consensus Convergence: parallel generation → iterative convergence.

Supports two convergence detection modes:
1. Epsilon-based (preferred): Uses quick evaluations to compute score range.
   Converges when score range < convergence_epsilon.
2. LLM-based (fallback): Uses a synthesis template with STATUS: CONVERGED pattern.
"""

from __future__ import annotations

import asyncio
import json
import re

from consortium.config.models import TaskConfig
from consortium.orchestrator.context import DesignArtifact, RunContext
from consortium.orchestrator.variants.base import VariantOrchestrator

_CONVERGENCE_PATTERN = re.compile(
    r"STATUS:\s*(CONVERGED|NOT_CONVERGED)",
    re.IGNORECASE,
)


class V7ConsensusOrchestrator(VariantOrchestrator):
    """Participants independently generate, share, revise, and converge.

    Workflow:
        Round 0: each participant generates independently
        Rounds 1..N:
            Each participant sees all other designs and revises
            Convergence check (renders synthesis template)
            If CONVERGED → use synthesis as final
        Fallback: pick latest merged/synthesized design
    """

    async def execute(self, task: TaskConfig, context: RunContext) -> DesignArtifact:
        participants = self._get_agents("participants")
        rubric_dims = context.rubric_dimensions
        convergence_template = self.config.workflow.convergence_template

        # Round 0: independent parallel generation
        self._log.info("round_start", round=0, step="generation", count=len(participants))
        designs = list(
            await asyncio.gather(
                *(
                    participant.act(
                        context=context,
                        round_num=0,
                        task=task,
                        rubric_dimensions=rubric_dims,
                    )
                    for participant in participants
                )
            )
        )

        # Convergence rounds
        for round_num in range(1, self.config.workflow.max_rounds):
            # Each participant revises seeing all other designs
            self._log.info("round_start", round=round_num, step="revision")
            new_designs = list(
                await asyncio.gather(
                    *(
                        participant.act(
                            context=context,
                            round_num=round_num,
                            task=task,
                            rubric_dimensions=rubric_dims,
                            previous_design=designs[i],
                            reviews=self._designs_as_reviews(designs, exclude_idx=i),
                        )
                        for i, participant in enumerate(participants)
                    )
                )
            )

            designs = new_designs

            # Convergence check — prefer epsilon-based, fall back to LLM-based
            epsilon = self.config.workflow.convergence_epsilon
            self._log.info("round_start", round=round_num, step="convergence_check")

            # Epsilon-based convergence: quick-evaluate all designs
            converged = False
            try:
                scores = await asyncio.gather(
                    *(
                        self._quick_evaluate(context, d, task, rubric_dims, round_num)
                        for d in designs
                    )
                )
                score_range = max(scores) - min(scores)
                self._log.info(
                    "epsilon_convergence_check",
                    round=round_num,
                    scores=scores,
                    score_range=f"{score_range:.3f}",
                    epsilon=epsilon,
                )
                if score_range < epsilon:
                    converged = True
                    # Pick the best-scoring design
                    best_idx = scores.index(max(scores))
                    designs[best_idx].is_final = True
                    self._log.info(
                        "consensus_epsilon_converged",
                        round=round_num,
                        design_id=designs[best_idx].design_id,
                        best_score=scores[best_idx],
                    )
                    return designs[best_idx]
            except Exception as e:
                self._log.warning(
                    "epsilon_convergence_fallback",
                    round=round_num,
                    error=str(e),
                )

            # Fallback: LLM-based convergence via synthesis template
            if not converged and convergence_template:
                converged_llm, synthesis = await self._check_convergence(
                    context=context,
                    round_num=round_num,
                    task=task,
                    designs=designs,
                    rubric_dims=rubric_dims,
                    template=convergence_template,
                )
                if converged_llm and synthesis:
                    synthesis.is_final = True
                    self._log.info(
                        "consensus_llm_converged",
                        round=round_num,
                        design_id=synthesis.design_id,
                    )
                    return synthesis

        # Did not converge: pick the last design from participant 0
        final = designs[0]
        final.is_final = True
        self._log.info("consensus_max_rounds", design_id=final.design_id)
        return final

    def _designs_as_reviews(
        self,
        designs: list[DesignArtifact],
        exclude_idx: int,
    ) -> list:
        """Convert peer designs to ReviewArtifact-like objects for the designer's reviews param."""
        from consortium.orchestrator.context import ReviewArtifact
        from datetime import UTC, datetime
        import uuid

        return [
            ReviewArtifact(
                review_id=uuid.uuid4().hex,
                run_id=designs[i].run_id,
                design_id=designs[i].design_id,
                round=designs[i].round,
                agent_role=designs[i].agent_role,
                agent_id=designs[i].agent_id,
                review_text=f"[Peer design from {designs[i].agent_id}]\n\n{designs[i].full_text}",
                created_at=datetime.now(UTC),
            )
            for i in range(len(designs))
            if i != exclude_idx
        ]

    async def _quick_evaluate(
        self,
        context: RunContext,
        design: DesignArtifact,
        task: TaskConfig,
        rubric_dims,
        convergence_round: int,
    ) -> float:
        """Quick single-call evaluation returning an overall score.

        Uses a dedicated quick-eval template that returns only a JSON score.
        Used for epsilon-based convergence detection only.

        *convergence_round* is the current convergence iteration, used for
        trace recording (not the round the design was created in).
        """
        agent = self._get_agents("participants")[0]

        template_vars = {
            "design_text": design.full_text,
            "system_name": task.variables.system_name,
            "complexity": task.complexity,
            "design_type": task.design_type,
            "rubric_dimensions": ([d.model_dump() for d in rubric_dims] if rubric_dims else None),
        }

        response = await agent._call_llm(
            template="evaluation/quick_eval.j2",
            template_vars=template_vars,
            context=context,
            step="quick_eval",
            round_num=convergence_round,
        )

        # Extract overall_score from JSON response
        try:
            json_match = re.search(r'\{[^}]*"overall_score"[^}]*\}', response.content)
            if json_match:
                data = json.loads(json_match.group())
                return float(data["overall_score"])
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

        # Fallback: try simple score patterns
        score_patterns = [
            r"(?:overall|total|final)\s*(?:score|rating)?:?\s*(\d+(?:\.\d+)?)\s*(?:/\s*10)?",
            r"(\d+(?:\.\d+)?)\s*/\s*10",
            r"score[:\s]+(\d+(?:\.\d+)?)",
        ]
        for pattern in score_patterns:
            match = re.search(pattern, response.content, re.IGNORECASE)
            if match:
                score = float(match.group(1))
                return min(score, 10.0)

        # Default: return 5.0 (neutral score) to avoid breaking convergence
        self._log.warning("quick_eval_score_unparseable", design_id=design.design_id)
        return 5.0

    async def _check_convergence(
        self,
        *,
        context,
        round_num,
        task,
        designs,
        rubric_dims,
        template,
    ) -> tuple[bool, DesignArtifact | None]:
        """Render convergence template and check for CONVERGED status."""
        from consortium.orchestrator.context import ConversationTurn, DesignArtifact as DA

        import uuid
        from datetime import UTC, datetime

        # Use the first participant to call the convergence template
        agent = self._get_agents("participants")[0]

        template_vars = {
            "designs": [{"agent_id": d.agent_id, "text": d.full_text} for d in designs],
            "system_name": task.variables.system_name,
            "complexity": task.complexity,
            "design_type": task.design_type,
            "round": round_num,
            "rubric_dimensions": ([d.model_dump() for d in rubric_dims] if rubric_dims else None),
        }

        response = await agent._call_llm(
            template=template,
            template_vars=template_vars,
            context=context,
            step="convergence_check",
            round_num=round_num,
        )

        context.conversation_history.append(
            ConversationTurn(
                round=round_num,
                agent_role=agent.role,
                agent_id=agent.agent_id,
                step="convergence_check",
                content=response.content,
                timestamp=datetime.now(UTC),
            )
        )

        match = _CONVERGENCE_PATTERN.search(response.content)
        if match and match.group(1).upper() == "CONVERGED":
            # Strip the STATUS marker line and any leading meta-commentary
            # so the stored design text contains only the actual design.
            design_text = _CONVERGENCE_PATTERN.sub("", response.content).strip()
            synthesis = DA(
                design_id=uuid.uuid4().hex,
                run_id=context.run_id,
                round=round_num,
                agent_role=agent.role,
                agent_id=agent.agent_id,
                full_text=design_text,
                token_count=response.output_tokens,
                is_final=False,
                created_at=datetime.now(UTC),
            )
            context.designs.append(synthesis)
            return True, synthesis

        return False, None
