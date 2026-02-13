"""Integration test: ProviderRegistry → BatchingProvider → rate limiter → retries."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from async_batcher import TokenBucketRateLimiter
from consortium.config.models import BatchingConfig, ModelConfig, RateLimitConfig
from consortium.providers.base import LLMProvider, LLMRequest, LLMResponse
from consortium.providers.batching import BatchingProvider
from consortium.providers.registry import ProviderRegistry


# ── Helpers ──────────────────────────────────────────────────────────────────


def _req(content: str = "q") -> LLMRequest:
    return LLMRequest(
        system_prompt="",
        messages=[{"role": "user", "content": content}],
        model_config_id="test",
    )


def _resp(content: str = "a") -> LLMResponse:
    return LLMResponse(
        content=content,
        model="test",
        input_tokens=10,
        output_tokens=5,
        latency_ms=50.0,
        cost_usd=0.001,
        timestamp=datetime.now(UTC),
        request_id="r1",
    )


class FlakyProvider(LLMProvider):
    """Fails a configurable number of times before succeeding."""

    def __init__(self, failures: int = 0) -> None:
        self.calls = 0
        self._failures = failures

    async def complete(self, request: LLMRequest) -> LLMResponse:
        return _resp("single")

    async def complete_batch(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        self.calls += 1
        if self.calls <= self._failures:
            raise ConnectionError(f"transient failure #{self.calls}")
        return [_resp(f"ok-{i}") for i in range(len(requests))]

    def supports_batch(self) -> bool:
        return True

    def estimate_cost(self, input_tokens: int, output_tokens: int, *, batch: bool = False) -> float:
        return 0.0


# ── Tests ────────────────────────────────────────────────────────────────────


class TestRegistryRateLimitWiring:
    """Verify that ProviderRegistry creates rate limiters from config."""

    def test_batching_provider_gets_rate_limiter(self) -> None:
        """When rate_limits is set, BatchingProvider's batcher should have a rate limiter."""
        mc = ModelConfig(
            id="t1",
            provider="openai",
            api_model="gpt-4",
            rate_limits=RateLimitConfig(requests_per_minute=100, tokens_per_minute=500_000),
            batching=BatchingConfig(enabled=True, window_ms=50, max_batch_size=8),
        )
        registry = ProviderRegistry()
        fake = FlakyProvider()

        with patch("consortium.providers.factory.create_provider", return_value=fake):
            provider = registry.get(mc)

        assert isinstance(provider, BatchingProvider)
        assert provider._batcher._rate_limiter is not None
        assert isinstance(provider._batcher._rate_limiter, TokenBucketRateLimiter)

    def test_no_rate_limits_no_limiter(self) -> None:
        """When rate_limits has zero values, no rate limiter should be created."""
        mc = ModelConfig(
            id="t2",
            provider="openai",
            api_model="gpt-4",
            rate_limits=RateLimitConfig(requests_per_minute=0, tokens_per_minute=0),
            batching=BatchingConfig(enabled=True, window_ms=50, max_batch_size=8),
        )
        registry = ProviderRegistry()
        fake = FlakyProvider()

        with patch("consortium.providers.factory.create_provider", return_value=fake):
            provider = registry.get(mc)

        assert isinstance(provider, BatchingProvider)
        assert provider._batcher._rate_limiter is None

    def test_retry_config_propagated(self) -> None:
        """Max retries and backoff should be configured on the batcher."""
        mc = ModelConfig(
            id="t3",
            provider="openai",
            api_model="gpt-4",
            batching=BatchingConfig(enabled=True, window_ms=50, max_batch_size=8),
        )
        registry = ProviderRegistry()
        fake = FlakyProvider()

        with patch("consortium.providers.factory.create_provider", return_value=fake):
            provider = registry.get(mc)

        assert isinstance(provider, BatchingProvider)
        assert provider._batcher._max_retries == 10
        assert provider._batcher._retry_backoff_base == 1.0


class TestRetryThroughBatcher:
    """Verify retries work through the BatchingProvider stack."""

    async def test_transient_failure_retried(self) -> None:
        """A transient error should be retried and eventually succeed."""
        inner = FlakyProvider(failures=2)
        bp = BatchingProvider(
            inner,
            window_ms=20,
            max_batch_size=4,
            max_retries=5,
            retry_backoff_base=0.01,  # Fast backoff for tests
            retryable_exceptions=(ConnectionError,),
        )

        async with bp:
            result = await bp.complete(_req("retry-test"))

        assert result.content == "ok-0"
        # Should have called complete_batch 3 times (2 failures + 1 success)
        assert inner.calls == 3

    async def test_all_retries_exhausted(self) -> None:
        """When all retries are exhausted, the error should propagate."""
        inner = FlakyProvider(failures=100)  # Always fails
        bp = BatchingProvider(
            inner,
            window_ms=20,
            max_batch_size=4,
            max_retries=2,
            retry_backoff_base=0.01,
            retryable_exceptions=(ConnectionError,),
        )

        async with bp:
            with pytest.raises(ConnectionError):
                await bp.complete(_req("fatal"))


class TestRateLimitingBehavior:
    """Verify that rate limiting actually throttles requests."""

    async def test_rate_limited_requests_succeed(self) -> None:
        """Requests should succeed even when rate limited (they wait)."""
        inner = FlakyProvider(failures=0)
        limiter = TokenBucketRateLimiter(
            requests_per_minute=600,  # 10/sec
            tokens_per_minute=0,
        )
        bp = BatchingProvider(
            inner,
            window_ms=20,
            max_batch_size=4,
            rate_limiter=limiter,
        )

        async with bp:
            tasks = [asyncio.create_task(bp.complete(_req(f"r{i}"))) for i in range(5)]
            results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert all(r.content.startswith("ok-") for r in results)


class TestAutoDetectRateLimits:
    """Verify auto-detection of rate limits from response headers."""

    async def test_rate_limiter_updated_from_response_headers(self) -> None:
        """When a response has provider_rpm/tpm_limit, the limiter should update."""

        class HeaderProvider(LLMProvider):
            async def complete(self, request: LLMRequest) -> LLMResponse:
                return _resp()

            async def complete_batch(self, requests: list[LLMRequest]) -> list[LLMResponse]:
                return [
                    LLMResponse(
                        content="ok",
                        model="test",
                        input_tokens=10,
                        output_tokens=5,
                        latency_ms=50.0,
                        cost_usd=0.001,
                        timestamp=datetime.now(UTC),
                        request_id="r1",
                        provider_rpm_limit=1000,
                        provider_tpm_limit=200_000,
                    )
                    for _ in requests
                ]

            def supports_batch(self) -> bool:
                return True

            def estimate_cost(
                self, input_tokens: int, output_tokens: int, *, batch: bool = False
            ) -> float:
                return 0.0

        inner = HeaderProvider()
        limiter = TokenBucketRateLimiter(
            requests_per_minute=500,  # Initial seed from YAML
            tokens_per_minute=150_000,
        )
        bp = BatchingProvider(
            inner,
            window_ms=20,
            max_batch_size=4,
            rate_limiter=limiter,
        )

        async with bp:
            await bp.complete(_req("detect"))

        # Should have auto-updated from response headers
        cfg = limiter.config
        assert cfg.requests_per_minute == 1000
        assert cfg.tokens_per_minute == 200_000
        assert bp._limits_synced is True

    async def test_limits_synced_flag_prevents_repeat_updates(self) -> None:
        """Once _limits_synced is set, further responses should not update."""
        call_count = 0

        class ChangingProvider(LLMProvider):
            async def complete(self, request: LLMRequest) -> LLMResponse:
                return _resp()

            async def complete_batch(self, requests: list[LLMRequest]) -> list[LLMResponse]:
                nonlocal call_count
                call_count += 1
                # Second batch has different limits
                rpm = 1000 if call_count == 1 else 2000
                return [
                    LLMResponse(
                        content="ok",
                        model="test",
                        input_tokens=10,
                        output_tokens=5,
                        latency_ms=50.0,
                        cost_usd=0.001,
                        timestamp=datetime.now(UTC),
                        request_id=f"r{call_count}",
                        provider_rpm_limit=rpm,
                        provider_tpm_limit=100_000,
                    )
                    for _ in requests
                ]

            def supports_batch(self) -> bool:
                return True

            def estimate_cost(
                self, input_tokens: int, output_tokens: int, *, batch: bool = False
            ) -> float:
                return 0.0

        inner = ChangingProvider()
        limiter = TokenBucketRateLimiter(requests_per_minute=500)
        bp = BatchingProvider(inner, window_ms=20, max_batch_size=4, rate_limiter=limiter)

        async with bp:
            await bp.complete(_req("first"))
            # Force a second batch by submitting again
            await bp.complete(_req("second"))

        # Should still have the FIRST batch's limits (1000), not second (2000)
        assert limiter.config.requests_per_minute == 1000
