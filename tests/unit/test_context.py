"""Tests for RunContext and artifact dataclasses."""

from __future__ import annotations

from datetime import UTC, datetime

from consortium.config.models import LimitsConfig
from consortium.orchestrator.context import (
    ConversationTurn,
    DesignArtifact,
    LLMCallTrace,
    ReviewArtifact,
    RunContext,
)


def _make_context(**overrides) -> RunContext:
    defaults = {
        "run_id": "test-run-001",
        "variant_id": "v1",
        "task_id": "task1",
        "repetition": 0,
    }
    defaults.update(overrides)
    return RunContext(**defaults)


def _make_trace(**overrides) -> LLMCallTrace:
    now = datetime.now(UTC)
    defaults = {
        "trace_id": "trace-001",
        "run_id": "test-run-001",
        "variant_id": "v1",
        "task_id": "task1",
        "repetition": 0,
        "agent_role": "designer",
        "agent_id": "designer_0",
        "step": "generation",
        "round": 0,
        "model_config_id": "gpt-4.1",
        "api_model": "gpt-4.1-2025-04-14",
        "provider": "openai",
        "system_prompt_hash": "abc123",
        "prompt_template": "generation/design_system.j2",
        "input_tokens": 1000,
        "output_tokens": 500,
        "cached_input_tokens": 0,
        "cost_usd": 0.01,
        "latency_ms": 1200.0,
        "started_at": now,
        "ended_at": now,
    }
    defaults.update(overrides)
    return LLMCallTrace(**defaults)


class TestRunContext:
    def test_initial_state(self) -> None:
        ctx = _make_context()
        assert ctx.run_id == "test-run-001"
        assert ctx.total_input_tokens == 0
        assert ctx.total_output_tokens == 0
        assert ctx.total_cost_usd == 0.0
        assert ctx.status == "pending"
        assert ctx.conversation_history == []
        assert ctx.designs == []
        assert ctx.reviews == []
        assert ctx.traces == []

    def test_record_trace_updates_accounting(self) -> None:
        ctx = _make_context()
        trace = _make_trace(input_tokens=1000, output_tokens=500, cost_usd=0.05)
        ctx.record_trace(trace)

        assert ctx.total_input_tokens == 1000
        assert ctx.total_output_tokens == 500
        assert ctx.total_cost_usd == 0.05
        assert len(ctx.traces) == 1

    def test_record_multiple_traces_accumulates(self) -> None:
        ctx = _make_context()
        ctx.record_trace(_make_trace(
            trace_id="t1", input_tokens=100, output_tokens=50, cost_usd=0.01,
        ))
        ctx.record_trace(_make_trace(
            trace_id="t2", input_tokens=200, output_tokens=100, cost_usd=0.02,
        ))

        assert ctx.total_input_tokens == 300
        assert ctx.total_output_tokens == 150
        assert abs(ctx.total_cost_usd - 0.03) < 1e-9
        assert len(ctx.traces) == 2

    def test_check_limits_within(self) -> None:
        ctx = _make_context()
        limits = LimitsConfig(max_tokens_per_run=10000, max_cost_per_run_usd=1.0)
        ok, msg = ctx.check_limits(limits)
        assert ok is True
        assert msg is None

    def test_check_limits_token_exceeded(self) -> None:
        ctx = _make_context()
        ctx.total_input_tokens = 8000
        ctx.total_output_tokens = 3000
        limits = LimitsConfig(max_tokens_per_run=10000, max_cost_per_run_usd=10.0)
        ok, msg = ctx.check_limits(limits)
        assert ok is False
        assert "Token limit" in msg

    def test_check_limits_cost_exceeded(self) -> None:
        ctx = _make_context()
        ctx.total_cost_usd = 5.0
        limits = LimitsConfig(max_tokens_per_run=999999, max_cost_per_run_usd=2.0)
        ok, msg = ctx.check_limits(limits)
        assert ok is False
        assert "Cost limit" in msg

    def test_get_latest_design_empty(self) -> None:
        ctx = _make_context()
        assert ctx.get_latest_design() is None

    def test_get_latest_design(self) -> None:
        ctx = _make_context()
        d1 = DesignArtifact(
            design_id="d1", run_id="r", round=0, agent_role="designer",
            agent_id="a", full_text="v1", token_count=10,
            is_final=False, created_at=datetime.now(UTC),
        )
        d2 = DesignArtifact(
            design_id="d2", run_id="r", round=1, agent_role="designer",
            agent_id="a", full_text="v2", token_count=20,
            is_final=True, created_at=datetime.now(UTC),
        )
        ctx.designs.extend([d1, d2])
        assert ctx.get_latest_design() is d2

    def test_get_reviews_for_round(self) -> None:
        ctx = _make_context()
        r1 = ReviewArtifact(
            review_id="r1", run_id="r", design_id="d1", round=1,
            agent_role="reviewer", agent_id="rev_0", review_text="Good.",
            created_at=datetime.now(UTC),
        )
        r2 = ReviewArtifact(
            review_id="r2", run_id="r", design_id="d1", round=2,
            agent_role="reviewer", agent_id="rev_0", review_text="Better.",
            created_at=datetime.now(UTC),
        )
        r3 = ReviewArtifact(
            review_id="r3", run_id="r", design_id="d1", round=1,
            agent_role="reviewer", agent_id="rev_1", review_text="Decent.",
            created_at=datetime.now(UTC),
        )
        ctx.reviews.extend([r1, r2, r3])
        round1 = ctx.get_reviews_for_round(1)
        assert len(round1) == 2
        assert r1 in round1
        assert r3 in round1
        assert r2 not in round1


class TestArtifacts:
    def test_design_artifact_fields(self) -> None:
        now = datetime.now(UTC)
        d = DesignArtifact(
            design_id="d1", run_id="r1", round=0,
            agent_role="designer", agent_id="a1",
            full_text="design text", token_count=100,
            is_final=False, created_at=now,
        )
        assert d.design_id == "d1"
        assert d.full_text == "design text"
        assert d.is_final is False

    def test_review_artifact_verdict_optional(self) -> None:
        r = ReviewArtifact(
            review_id="r1", run_id="r1", design_id="d1", round=1,
            agent_role="reviewer", agent_id="rev_0",
            review_text="looks good",
        )
        assert r.verdict is None

    def test_review_artifact_with_verdict(self) -> None:
        r = ReviewArtifact(
            review_id="r1", run_id="r1", design_id="d1", round=1,
            agent_role="adversarial", agent_id="adv_0",
            review_text="VERDICT: REJECT\nNeeds work.",
            verdict="reject",
        )
        assert r.verdict == "reject"

    def test_conversation_turn(self) -> None:
        now = datetime.now(UTC)
        turn = ConversationTurn(
            round=0, agent_role="designer", agent_id="d0",
            step="generation", content="output", timestamp=now,
        )
        assert turn.step == "generation"
        assert turn.content == "output"
