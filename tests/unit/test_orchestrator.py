"""Tests for variant orchestrators and factory."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from consortium.config.models import (
    AgentConfig,
    ModelConfig,
    ModelParametersConfig,
    PricingConfig,
    VariantAgentsConfig,
    VariantConfig,
    WorkflowConfig,
)
from consortium.orchestrator.context import DesignArtifact, ReviewArtifact, RunContext
from consortium.orchestrator.factory import _VARIANT_REGISTRY, create_variant_orchestrator
from consortium.orchestrator.variants.base import VariantOrchestrator


# ── Helpers ───────────────────────────────────────────────────────────────────


def _run_context() -> RunContext:
    ctx = RunContext(
        run_id="test-run",
        variant_id="v1",
        task_id="task1",
        repetition=0,
    )
    ctx._rubric_dimensions = []  # type: ignore[attr-defined]
    return ctx


def _task_config():
    from consortium.config.models import TaskConfig, TaskVariables

    return TaskConfig(
        id="task1",
        name="Test Task",
        complexity="medium",
        rubric="system_design",
        variables=TaskVariables(
            system_name="TestSystem",
            problem_statement="Test.",
            hard_constraints=["c1"],
            use_cases=["u1"],
            complexity_drivers=["d1"],
        ),
    )


def _mock_designer_agent(agent_id: str = "designer") -> MagicMock:
    """Create a mock agent that returns a DesignArtifact from act()."""
    agent = MagicMock()
    agent.agent_id = agent_id
    agent.role = "designer"
    agent.prompt_template = "generation/design_system.j2"

    # v1 constructs a real ReviewerAgent from designer attrs
    agent.model_config = MagicMock()
    agent.model_config.id = "mock-model"
    agent.model_config.provider = "mock"
    agent.renderer = MagicMock()
    agent.renderer.render = MagicMock(return_value="mocked prompt text")
    agent.parameters = MagicMock()
    agent.parameters.temperature = 0.7
    agent.parameters.max_tokens = 8192
    agent.parameters.top_p = 1.0
    agent.limits = MagicMock()
    agent.limits.max_tokens_per_run = 999_999
    agent.limits.max_cost_per_run_usd = 999.0

    mock_response = MagicMock()
    mock_response.content = "Mock review feedback"
    mock_response.model = "mock-model"
    mock_response.input_tokens = 100
    mock_response.output_tokens = 50
    mock_response.cached_input_tokens = 0
    mock_response.cost_usd = 0.0
    mock_response.latency_ms = 10.0
    mock_response.batch_id = None
    agent.provider = MagicMock()
    agent.provider.complete = AsyncMock(return_value=mock_response)

    call_count = 0

    async def mock_act(**kwargs):
        nonlocal call_count
        call_count += 1
        return DesignArtifact(
            design_id=f"design-{agent_id}-{call_count}",
            run_id=kwargs["context"].run_id,
            round=kwargs["round_num"],
            agent_role="designer",
            agent_id=agent_id,
            full_text=f"Design from {agent_id} round {kwargs['round_num']}",
            token_count=100,
            is_final=False,
            created_at=datetime.now(UTC),
        )

    agent.act = mock_act
    return agent


def _mock_reviewer_agent(agent_id: str = "reviewer") -> MagicMock:
    """Create a mock reviewer that returns a ReviewArtifact from act()."""
    agent = MagicMock()
    agent.agent_id = agent_id
    agent.role = "reviewer"

    call_count = 0

    async def mock_act(**kwargs):
        nonlocal call_count
        call_count += 1
        return ReviewArtifact(
            review_id=f"review-{agent_id}-{call_count}",
            run_id=kwargs["context"].run_id,
            design_id=kwargs["design"].design_id,
            round=kwargs["round_num"],
            agent_role="reviewer",
            agent_id=agent_id,
            review_text=f"Review from {agent_id}",
            created_at=datetime.now(UTC),
        )

    agent.act = mock_act
    return agent


def _mock_adversary_agent(
    verdicts: list[str] | None = None,
) -> MagicMock:
    """Create a mock adversary that returns ReviewArtifacts with programmed verdicts."""
    agent = MagicMock()
    agent.agent_id = "adversary"
    agent.role = "adversarial"

    call_count = 0
    verdict_list = verdicts or ["reject", "accept"]

    async def mock_act(**kwargs):
        nonlocal call_count
        v = verdict_list[min(call_count, len(verdict_list) - 1)]
        call_count += 1
        return ReviewArtifact(
            review_id=f"adv-review-{call_count}",
            run_id=kwargs["context"].run_id,
            design_id=kwargs["design"].design_id,
            round=kwargs["round_num"],
            agent_role="adversarial",
            agent_id="adversary",
            review_text=f"VERDICT: {v.upper()}",
            verdict=v,
            created_at=datetime.now(UTC),
        )

    agent.act = mock_act
    return agent


def _mock_merger_agent() -> MagicMock:
    agent = MagicMock()
    agent.agent_id = "merger"
    agent.role = "merger"

    async def mock_act(**kwargs):
        return DesignArtifact(
            design_id="merged-design",
            run_id=kwargs["context"].run_id,
            round=kwargs["round_num"],
            agent_role="merger",
            agent_id="merger",
            full_text="Merged design",
            token_count=200,
            is_final=False,
            created_at=datetime.now(UTC),
        )

    agent.act = mock_act
    return agent


def _mock_judge_agent() -> MagicMock:
    agent = MagicMock()
    agent.agent_id = "judge"
    agent.role = "judge"

    async def mock_act(**kwargs):
        return DesignArtifact(
            design_id="judge-ruling",
            run_id=kwargs["context"].run_id,
            round=kwargs["round_num"],
            agent_role="judge",
            agent_id="judge",
            full_text="Judge ruling",
            token_count=150,
            is_final=False,
            created_at=datetime.now(UTC),
        )

    agent.act = mock_act
    return agent


# ── Variant Registry Tests ────────────────────────────────────────────────────


class TestVariantRegistry:
    def test_all_variants_registered(self) -> None:
        for vid in ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"]:
            assert vid in _VARIANT_REGISTRY

    def test_create_unknown_variant_raises(self) -> None:
        config = VariantConfig(id="v99", name="Unknown")
        with pytest.raises(ValueError, match="Unknown variant"):
            create_variant_orchestrator(config, {})


# ── V1 Baseline Tests ────────────────────────────────────────────────────────


class TestV1Baseline:
    def test_single_round(self) -> None:
        config = VariantConfig(
            id="v1",
            name="Baseline",
            workflow=WorkflowConfig(max_rounds=1),
        )
        designer = _mock_designer_agent()
        orch = create_variant_orchestrator(config, {"designer": designer})

        ctx = _run_context()
        task = _task_config()
        result = asyncio.get_event_loop().run_until_complete(orch.execute(task, ctx))

        assert result.is_final is True
        assert "designer" in result.agent_id

    def test_multi_round_self_refinement(self) -> None:
        config = VariantConfig(
            id="v1",
            name="Baseline",
            workflow=WorkflowConfig(max_rounds=3),
        )
        designer = _mock_designer_agent()
        orch = create_variant_orchestrator(config, {"designer": designer})

        ctx = _run_context()
        result = asyncio.get_event_loop().run_until_complete(orch.execute(_task_config(), ctx))

        assert result.is_final is True
        assert result.round == 2  # rounds 0, 1, 2


# ── V2 Leader + Reviewers Tests ──────────────────────────────────────────────


class TestV2LeaderReviewers:
    def test_leader_reviewer_workflow(self) -> None:
        config = VariantConfig(
            id="v2",
            name="Leader + Reviewers",
            workflow=WorkflowConfig(max_rounds=2),
        )
        leader = _mock_designer_agent("leader")
        reviewers = [_mock_reviewer_agent("rev_0"), _mock_reviewer_agent("rev_1")]
        orch = create_variant_orchestrator(
            config,
            {"leader": leader, "reviewers": reviewers},
        )

        ctx = _run_context()
        result = asyncio.get_event_loop().run_until_complete(orch.execute(_task_config(), ctx))

        assert result.is_final is True


# ── V4 Adversarial Tests ─────────────────────────────────────────────────────


class TestV4Adversarial:
    def test_early_accept_stops(self) -> None:
        config = VariantConfig(
            id="v4",
            name="Adversarial",
            workflow=WorkflowConfig(max_rounds=5),
        )
        leader = _mock_designer_agent("leader")
        adversary = _mock_adversary_agent(["accept"])  # accepts immediately
        orch = create_variant_orchestrator(
            config,
            {"leader": leader, "adversarial_reviewer": adversary},
        )

        ctx = _run_context()
        result = asyncio.get_event_loop().run_until_complete(orch.execute(_task_config(), ctx))

        assert result.is_final is True
        # Only round 0 (gen) + round 1 (review → accept) = leader called once
        assert result.round == 0

    def test_reject_then_accept(self) -> None:
        config = VariantConfig(
            id="v4",
            name="Adversarial",
            workflow=WorkflowConfig(max_rounds=5),
        )
        leader = _mock_designer_agent("leader")
        adversary = _mock_adversary_agent(["reject", "accept"])
        orch = create_variant_orchestrator(
            config,
            {"leader": leader, "adversarial_reviewer": adversary},
        )

        ctx = _run_context()
        result = asyncio.get_event_loop().run_until_complete(orch.execute(_task_config(), ctx))

        assert result.is_final is True


# ── V3 Parallel Merge Tests ──────────────────────────────────────────────────


class TestV3ParallelMerge:
    def test_parallel_then_merge(self) -> None:
        config = VariantConfig(
            id="v3",
            name="Parallel Merge",
            agents=VariantAgentsConfig(
                parallel_leaders=AgentConfig(
                    role="designer",
                    model="gpt-4.1",
                    system_prompt_template="generation/design_system.j2",
                    count=3,
                ),
            ),
            workflow=WorkflowConfig(max_rounds=1),
        )
        leaders = [_mock_designer_agent(f"pl_{i}") for i in range(3)]
        merger = _mock_merger_agent()
        orch = create_variant_orchestrator(
            config,
            {"parallel_leaders": leaders, "merger": merger},
        )

        ctx = _run_context()
        result = asyncio.get_event_loop().run_until_complete(orch.execute(_task_config(), ctx))

        assert result.is_final is True
        assert result.agent_id == "merger"


# ── V8 Structured Debate Tests ───────────────────────────────────────────────


class TestV8StructuredDebate:
    def test_debate_then_judge(self) -> None:
        config = VariantConfig(
            id="v8",
            name="Structured Debate",
            workflow=WorkflowConfig(max_rounds=2),
        )

        # v8 calls debater._call_llm() directly, so we need AsyncMock for it
        def _mock_debater(agent_id: str) -> MagicMock:
            agent = _mock_designer_agent(agent_id)
            agent.role = "debater"

            call_count = 0

            async def mock_call_llm(**kwargs):
                nonlocal call_count
                call_count += 1
                resp = MagicMock()
                resp.content = f"Position from {agent_id}"
                resp.output_tokens = 100
                return resp

            agent._call_llm = mock_call_llm
            return agent

        debaters = [_mock_debater(f"debater_{i}") for i in range(3)]
        judge = _mock_judge_agent()
        orch = create_variant_orchestrator(
            config,
            {"debaters": debaters, "judge": judge},
        )

        ctx = _run_context()
        result = asyncio.get_event_loop().run_until_complete(orch.execute(_task_config(), ctx))

        assert result.is_final is True
        assert result.agent_id == "judge"


# ── Base Class Tests ──────────────────────────────────────────────────────────


class TestVariantOrchestratorBase:
    def test_get_agent_missing_raises(self) -> None:
        config = VariantConfig(id="v1", name="Test")
        orch = create_variant_orchestrator(config, {"designer": _mock_designer_agent()})
        with pytest.raises(KeyError, match="not found"):
            orch._get_agent("nonexistent")

    def test_get_agent_returns_single(self) -> None:
        designer = _mock_designer_agent()
        config = VariantConfig(id="v1", name="Test")
        orch = create_variant_orchestrator(config, {"designer": designer})
        assert orch._get_agent("designer") is designer

    def test_get_agents_returns_list(self) -> None:
        reviewers = [_mock_reviewer_agent("r0"), _mock_reviewer_agent("r1")]
        config = VariantConfig(id="v1", name="Test")
        orch = create_variant_orchestrator(config, {"reviewers": reviewers})
        assert orch._get_agents("reviewers") == reviewers

    def test_get_agent_on_list_raises(self) -> None:
        reviewers = [_mock_reviewer_agent("r0")]
        config = VariantConfig(id="v1", name="Test")
        orch = create_variant_orchestrator(config, {"reviewers": reviewers})
        with pytest.raises(TypeError, match="is a list"):
            orch._get_agent("reviewers")
