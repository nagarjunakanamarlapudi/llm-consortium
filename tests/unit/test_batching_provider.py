"""Tests for BatchingProvider and ProviderRegistry integration layers."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from async_batcher import Batcher
from consortium.config.models import BatchingConfig, ModelConfig
from consortium.providers.base import LLMProvider, LLMRequest, LLMResponse
from consortium.providers.batching import BatchingProvider
from consortium.providers.registry import ProviderRegistry


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_request(content: str = "hello") -> LLMRequest:
    return LLMRequest(
        system_prompt="",
        messages=[{"role": "user", "content": content}],
        model_config_id="test-model",
    )


def _make_response(content: str = "hi") -> LLMResponse:
    return LLMResponse(
        content=content,
        model="test-model",
        input_tokens=10,
        output_tokens=5,
        latency_ms=50.0,
        cost_usd=0.001,
        timestamp=datetime.now(UTC),
        request_id="req-001",
    )


class FakeProvider(LLMProvider):
    """In-memory provider for testing."""

    def __init__(self) -> None:
        self.complete_calls: list[LLMRequest] = []
        self.batch_calls: list[list[LLMRequest]] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.complete_calls.append(request)
        return _make_response(f"reply-to-{request.messages[0]['content']}")

    async def complete_batch(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        self.batch_calls.append(requests)
        return [_make_response(f"batch-{r.messages[0]['content']}") for r in requests]

    def supports_batch(self) -> bool:
        return True

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        batch: bool = False,
    ) -> float:
        return 0.0


# ── BatchingProvider Tests ───────────────────────────────────────────────────


class TestBatchingProvider:
    async def test_submit_delegates_to_batcher(self) -> None:
        """A single complete() call goes through the batcher and returns."""
        inner = FakeProvider()
        bp = BatchingProvider(inner, window_ms=50, max_batch_size=8)

        async with bp:
            result = await bp.complete(_make_request("test"))

        assert result.content == "batch-test"
        # The batcher should have called complete_batch on the inner provider
        assert len(inner.batch_calls) >= 1

    async def test_submit_many_coalesces(self) -> None:
        """Multiple concurrent submits should be coalesced into fewer batches."""
        inner = FakeProvider()
        bp = BatchingProvider(inner, window_ms=200, max_batch_size=64)

        async with bp:
            tasks = [asyncio.create_task(bp.complete(_make_request(f"q{i}"))) for i in range(10)]
            results = await asyncio.gather(*tasks)

        assert len(results) == 10
        # All 10 should have been processed, but in fewer than 10 batch calls
        total_requests = sum(len(batch) for batch in inner.batch_calls)
        assert total_requests == 10
        assert len(inner.batch_calls) < 10

    async def test_complete_batch_passthrough(self) -> None:
        """complete_batch() should delegate through the batcher."""
        inner = FakeProvider()
        bp = BatchingProvider(inner, window_ms=50, max_batch_size=64)

        async with bp:
            requests = [_make_request(f"b{i}") for i in range(3)]
            results = await bp.complete_batch(requests)

        assert len(results) == 3
        for i, r in enumerate(results):
            assert r.content == f"batch-b{i}"

    def test_inner_property(self) -> None:
        """The inner property should return the underlying LLMProvider."""
        inner = FakeProvider()
        bp = BatchingProvider(inner, window_ms=50, max_batch_size=8)
        assert bp.inner is inner

    def test_supports_batch_delegates(self) -> None:
        """supports_batch should delegate to inner provider."""
        inner = FakeProvider()
        bp = BatchingProvider(inner, window_ms=50)
        assert bp.supports_batch() is True

    def test_estimate_cost_delegates(self) -> None:
        """estimate_cost should delegate to inner provider."""
        inner = FakeProvider()
        bp = BatchingProvider(inner, window_ms=50)
        assert bp.estimate_cost(100, 100) == 0.0

    async def test_stats_tracked(self) -> None:
        """Stats should be updated after submitting requests."""
        inner = FakeProvider()
        bp = BatchingProvider(inner, window_ms=50, max_batch_size=8)

        async with bp:
            await bp.complete(_make_request("s1"))
            await bp.complete(_make_request("s2"))

        assert bp.stats.total_requests >= 2
        assert bp.stats.total_flushes >= 1


# ── ProviderRegistry Tests ───────────────────────────────────────────────────


class TestProviderRegistry:
    def _make_model_config(
        self,
        *,
        model_id: str = "test-model",
        provider: str = "openai",
        api_model: str = "gpt-4",
        batching_enabled: bool = False,
        window_ms: float = 100.0,
        max_batch_size: int = 64,
    ) -> ModelConfig:
        batching = (
            BatchingConfig(
                enabled=batching_enabled, window_ms=window_ms, max_batch_size=max_batch_size
            )
            if batching_enabled
            else None
        )
        return ModelConfig(
            id=model_id,
            provider=provider,
            api_model=api_model,
            batching=batching,
        )

    async def test_context_manager_lifecycle(self) -> None:
        """Registry should start and shut down cleanly as context manager."""
        registry = ProviderRegistry()
        async with registry:
            assert registry.running
        assert not registry.running

    def test_stats_empty_registry(self) -> None:
        """Empty registry should return empty stats."""
        registry = ProviderRegistry()
        assert registry.stats == {}

    def test_registry_deduplication_with_mock(self) -> None:
        """Same provider:api_model key should return the same provider instance."""
        registry = ProviderRegistry()

        mc1 = self._make_model_config(model_id="cfg-1", provider="openai", api_model="gpt-4")
        mc2 = self._make_model_config(model_id="cfg-2", provider="openai", api_model="gpt-4")

        fake = FakeProvider()
        with patch("consortium.providers.factory.create_provider", return_value=fake):
            p1 = registry.get(mc1)
            p2 = registry.get(mc2)

        # Same provider:api_model → same instance
        assert p1 is p2

    def test_registry_batching_wraps_provider(self) -> None:
        """When batching is enabled, registry should wrap with BatchingProvider."""
        registry = ProviderRegistry()
        mc = self._make_model_config(batching_enabled=True, window_ms=50, max_batch_size=16)

        fake = FakeProvider()
        with patch("consortium.providers.factory.create_provider", return_value=fake):
            provider = registry.get(mc)

        assert isinstance(provider, BatchingProvider)
        assert provider.inner is fake

    def test_registry_no_batching_returns_raw(self) -> None:
        """When batching is disabled, registry should return the raw provider."""
        registry = ProviderRegistry()
        mc = self._make_model_config(batching_enabled=False)

        fake = FakeProvider()
        with patch("consortium.providers.factory.create_provider", return_value=fake):
            provider = registry.get(mc)

        assert provider is fake
        assert not isinstance(provider, BatchingProvider)

    def test_registry_different_models_get_different_providers(self) -> None:
        """Different api_models should return different providers."""
        registry = ProviderRegistry()
        mc1 = self._make_model_config(api_model="gpt-4")
        mc2 = self._make_model_config(api_model="gpt-3.5-turbo")

        fake1 = FakeProvider()
        fake2 = FakeProvider()
        call_count = 0

        def mock_create(mc, **kwargs):
            nonlocal call_count
            call_count += 1
            return fake1 if call_count == 1 else fake2

        with patch("consortium.providers.factory.create_provider", side_effect=mock_create):
            p1 = registry.get(mc1)
            p2 = registry.get(mc2)

        assert p1 is not p2
