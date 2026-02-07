"""Tests for BaseAgent and agent subclasses."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from consortium.agents.adversary import AdversarialReviewer, parse_verdict
from consortium.agents.base import BaseAgent
from consortium.agents.designer import DesignerAgent
from consortium.agents.reviewer import ReviewerAgent
from consortium.config.models import (
    LimitsConfig,
    ModelConfig,
    ModelParametersConfig,
    PricingConfig,
)
from consortium.orchestrator.context import DesignArtifact, RunContext
from consortium.providers.base import LLMResponse


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _model_config() -> ModelConfig:
    return ModelConfig(
        id="test-model",
        provider="openai",
        api_model="gpt-4.1-test",
        pricing=PricingConfig(input=3.0, output=15.0),
        parameters=ModelParametersConfig(temperature=0.7, max_tokens=1024, top_p=1.0),
    )


def _mock_provider(content: str = "test output") -> AsyncMock:
    provider = AsyncMock()
    provider.complete.return_value = LLMResponse(
        content=content,
        model="gpt-4.1-test",
        input_tokens=100,
        output_tokens=50,
        latency_ms=500.0,
        cost_usd=0.001,
        timestamp=datetime.now(UTC),
        request_id="req-001",
    )
    return provider


def _mock_renderer() -> MagicMock:
    renderer = MagicMock()
    renderer.render.return_value = "rendered prompt"
    return renderer


def _run_context() -> RunContext:
    return RunContext(
        run_id="test-run",
        variant_id="v1",
        task_id="task1",
        repetition=0,
    )


def _task_config():
    from consortium.config.models import TaskConfig, TaskVariables
    return TaskConfig(
        id="task1",
        name="Test Task",
        complexity="medium",
        design_type="system",
        rubric="system_design",
        variables=TaskVariables(
            system_name="TestSystem",
            problem_statement="Design a test system.",
            hard_constraints=["fast"],
            use_cases=["uc1"],
            complexity_drivers=["driver1"],
        ),
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestBaseAgentCallLLM:
    def test_call_llm_records_trace(self) -> None:
        ctx = _run_context()
        provider = _mock_provider()
        renderer = _mock_renderer()

        agent = DesignerAgent(
            agent_id="d0",
            role="designer",
            model_config=_model_config(),
            provider=provider,
            renderer=renderer,
            prompt_template="generation/design_system.j2",
        )

        response = asyncio.get_event_loop().run_until_complete(
            agent._call_llm(
                template="generation/design_system.j2",
                template_vars={"key": "val"},
                context=ctx,
                step="generation",
                round_num=0,
            )
        )

        assert response.content == "test output"
        assert len(ctx.traces) == 1
        assert ctx.traces[0].agent_id == "d0"
        assert ctx.traces[0].step == "generation"
        assert ctx.total_input_tokens == 100
        assert ctx.total_output_tokens == 50

    def test_call_llm_checks_limits(self) -> None:
        ctx = _run_context()
        ctx.total_input_tokens = 999_999
        ctx.total_output_tokens = 999_999
        provider = _mock_provider()
        renderer = _mock_renderer()
        limits = LimitsConfig(max_tokens_per_run=100, max_cost_per_run_usd=100.0)

        agent = DesignerAgent(
            agent_id="d0",
            role="designer",
            model_config=_model_config(),
            provider=provider,
            renderer=renderer,
            prompt_template="generation/design_system.j2",
            limits=limits,
        )

        with pytest.raises(RuntimeError, match="limit"):
            asyncio.get_event_loop().run_until_complete(
                agent._call_llm(
                    template="generation/design_system.j2",
                    template_vars={},
                    context=ctx,
                    step="generation",
                    round_num=0,
                )
            )


class TestDesignerAgent:
    def test_act_creates_design_artifact(self) -> None:
        ctx = _run_context()
        provider = _mock_provider("A full design document")
        renderer = _mock_renderer()
        task = _task_config()

        agent = DesignerAgent(
            agent_id="designer_0",
            role="designer",
            model_config=_model_config(),
            provider=provider,
            renderer=renderer,
            prompt_template="generation/design_system.j2",
        )

        design = asyncio.get_event_loop().run_until_complete(
            agent.act(context=ctx, round_num=0, task=task)
        )

        assert isinstance(design, DesignArtifact)
        assert design.full_text == "A full design document"
        assert design.agent_id == "designer_0"
        assert design.is_final is False
        assert len(ctx.designs) == 1
        assert len(ctx.conversation_history) == 1
        assert ctx.conversation_history[0].step == "generation"

    def test_act_revision_with_previous_design(self) -> None:
        ctx = _run_context()
        provider = _mock_provider("revised design")
        renderer = _mock_renderer()
        task = _task_config()

        agent = DesignerAgent(
            agent_id="designer_0",
            role="designer",
            model_config=_model_config(),
            provider=provider,
            renderer=renderer,
            prompt_template="generation/design_system.j2",
        )

        prev = DesignArtifact(
            design_id="prev", run_id="test-run", round=0,
            agent_role="designer", agent_id="designer_0",
            full_text="old design", token_count=10,
            is_final=False, created_at=datetime.now(UTC),
        )

        design = asyncio.get_event_loop().run_until_complete(
            agent.act(
                context=ctx, round_num=1, task=task,
                previous_design=prev,
            )
        )

        assert design.full_text == "revised design"
        assert ctx.conversation_history[0].step == "revision"


class TestParseVerdict:
    def test_accept(self) -> None:
        assert parse_verdict("Some analysis\nVERDICT: ACCEPT\n") == "accept"

    def test_reject(self) -> None:
        assert parse_verdict("Issues found.\nVERDICT: REJECT\nDetails...") == "reject"

    def test_case_insensitive(self) -> None:
        assert parse_verdict("verdict: accept") == "accept"

    def test_unparseable_defaults_to_accept(self) -> None:
        assert parse_verdict("No verdict here, just text.") == "accept"

    def test_embedded_in_longer_text(self) -> None:
        text = """
        After careful review:
        - Issue 1
        - Issue 2
        VERDICT: REJECT
        Please address the above.
        """
        assert parse_verdict(text) == "reject"
