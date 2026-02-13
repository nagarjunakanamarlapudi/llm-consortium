"""Tests for TokenBucketRateLimiter."""

from __future__ import annotations

import asyncio
import time

import pytest

from async_batcher._rate_limiter import (
    RateLimiterConfig,
    RateLimiterStats,
    TokenBucketRateLimiter,
)


class TestRateLimiterConfig:
    """Test the RateLimiterConfig dataclass."""

    def test_defaults(self) -> None:
        cfg = RateLimiterConfig()
        assert cfg.requests_per_minute == 0
        assert cfg.tokens_per_minute == 0

    def test_custom_values(self) -> None:
        cfg = RateLimiterConfig(requests_per_minute=500, tokens_per_minute=150_000)
        assert cfg.requests_per_minute == 500
        assert cfg.tokens_per_minute == 150_000


class TestRateLimiterStats:
    """Test the RateLimiterStats dataclass."""

    def test_defaults(self) -> None:
        stats = RateLimiterStats()
        assert stats.total_acquires == 0
        assert stats.total_waits == 0
        assert stats.total_wait_ms == 0.0
        assert stats.peak_wait_ms == 0.0

    def test_avg_wait_ms_no_waits(self) -> None:
        stats = RateLimiterStats()
        assert stats.avg_wait_ms == 0.0  # no division by zero

    def test_avg_wait_ms(self) -> None:
        stats = RateLimiterStats(total_waits=4, total_wait_ms=200.0)
        assert stats.avg_wait_ms == 50.0


class TestTokenBucketRateLimiter:
    """Test the TokenBucketRateLimiter."""

    def test_disabled_when_no_limits(self) -> None:
        limiter = TokenBucketRateLimiter()
        assert not limiter.enabled

    def test_enabled_with_rpm(self) -> None:
        limiter = TokenBucketRateLimiter(requests_per_minute=100)
        assert limiter.enabled

    def test_enabled_with_tpm(self) -> None:
        limiter = TokenBucketRateLimiter(tokens_per_minute=50_000)
        assert limiter.enabled

    def test_config_property(self) -> None:
        limiter = TokenBucketRateLimiter(requests_per_minute=500, tokens_per_minute=150_000)
        cfg = limiter.config
        assert cfg.requests_per_minute == 500
        assert cfg.tokens_per_minute == 150_000

    @pytest.mark.asyncio
    async def test_acquire_no_limit_instant(self) -> None:
        """Disabled limiter should return immediately."""
        limiter = TokenBucketRateLimiter()
        start = time.monotonic()
        await limiter.acquire(estimated_tokens=100)
        elapsed = time.monotonic() - start
        assert elapsed < 0.1
        assert limiter.stats.total_acquires == 0  # disabled = no tracking

    @pytest.mark.asyncio
    async def test_acquire_within_capacity(self) -> None:
        """Requests within RPM capacity should not wait."""
        limiter = TokenBucketRateLimiter(requests_per_minute=600)  # 10/sec
        start = time.monotonic()
        for _ in range(5):
            await limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed < 0.1  # should be nearly instant
        assert limiter.stats.total_acquires == 5
        assert limiter.stats.total_waits == 0

    @pytest.mark.asyncio
    async def test_acquire_exceeds_rpm_waits(self) -> None:
        """Exceeding RPM should cause waiting."""
        # Very low RPM: 60 RPM = 1/sec
        limiter = TokenBucketRateLimiter(requests_per_minute=60)

        # Consume the initial bucket (starts full at 60 tokens)
        # We rapidly drain 60 tokens, then the 61st must wait
        for _ in range(60):
            await limiter.acquire()

        assert limiter.stats.total_waits == 0  # all within initial capacity

        # Next acquire must wait ~1 second for 1 token refill
        start = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - start

        assert elapsed >= 0.5  # should wait ~1s, allow margin
        assert limiter.stats.total_waits >= 1

    @pytest.mark.asyncio
    async def test_acquire_tpm_limiting(self) -> None:
        """TPM-based limiting should work independently of RPM."""
        # 600 TPM = 10 tokens/sec, no RPM limit
        limiter = TokenBucketRateLimiter(tokens_per_minute=600)

        # Consume all tokens at once
        await limiter.acquire(estimated_tokens=600)
        assert limiter.stats.total_waits == 0

        # Next request for just 10 tokens should wait ~1s
        start = time.monotonic()
        await limiter.acquire(estimated_tokens=10)
        elapsed = time.monotonic() - start

        assert elapsed >= 0.5  # should wait ~1s
        assert limiter.stats.total_waits >= 1

    @pytest.mark.asyncio
    async def test_acquire_both_limits(self) -> None:
        """When both RPM and TPM are set, the stricter one governs."""
        # 600 RPM (10/sec), 60 TPM (1 token/sec)
        # TPM is stricter: 1 token/sec means 1 request/sec even though RPM allows 10
        limiter = TokenBucketRateLimiter(
            requests_per_minute=600,
            tokens_per_minute=60,
        )

        # Consume initial TPM bucket
        await limiter.acquire(estimated_tokens=60)

        # RPM still has capacity, but TPM is exhausted
        start = time.monotonic()
        await limiter.acquire(estimated_tokens=1)
        elapsed = time.monotonic() - start

        assert elapsed >= 0.5  # TPM is the bottleneck

    @pytest.mark.asyncio
    async def test_concurrent_acquire(self) -> None:
        """Multiple concurrent acquires should be serialized properly."""
        limiter = TokenBucketRateLimiter(requests_per_minute=600)  # 10/sec

        async def do_acquire():
            await limiter.acquire()

        # Launch 10 concurrent acquires
        await asyncio.gather(*[do_acquire() for _ in range(10)])
        assert limiter.stats.total_acquires == 10

    def test_snapshot(self) -> None:
        """Snapshot should return current state."""
        limiter = TokenBucketRateLimiter(
            requests_per_minute=500,
            tokens_per_minute=150_000,
            name="test",
        )
        snap = limiter.snapshot()
        assert snap["rpm_limit"] == 500
        assert snap["tpm_limit"] == 150_000
        assert snap["rpm_available"] is not None
        assert snap["tpm_available"] is not None
        assert snap["stats"]["total_acquires"] == 0

    def test_snapshot_disabled(self) -> None:
        """Snapshot with no limits should have None for available."""
        limiter = TokenBucketRateLimiter()
        snap = limiter.snapshot()
        assert snap["rpm_limit"] == 0
        assert snap["rpm_available"] is None
        assert snap["tpm_available"] is None

    @pytest.mark.asyncio
    async def test_stats_peak_tracking(self) -> None:
        """Peak wait time should track the longest individual wait."""
        # 60 RPM = 1 request/sec
        limiter = TokenBucketRateLimiter(requests_per_minute=60)

        # Drain bucket
        for _ in range(60):
            await limiter.acquire()

        # Next must wait
        await limiter.acquire()

        assert limiter.stats.peak_wait_ms > 0
        assert limiter.stats.peak_wait_ms >= limiter.stats.avg_wait_ms


class TestUpdateLimits:
    """Test dynamic limit updates from provider response headers."""

    @pytest.mark.asyncio
    async def test_update_limits_no_change(self) -> None:
        """Updating with same values should be a no-op."""
        limiter = TokenBucketRateLimiter(requests_per_minute=500, tokens_per_minute=150_000)
        snap_before = limiter.snapshot()
        await limiter.update_limits(rpm=500, tpm=150_000)
        snap_after = limiter.snapshot()
        assert snap_before["rpm_limit"] == snap_after["rpm_limit"]
        assert snap_before["tpm_limit"] == snap_after["tpm_limit"]

    @pytest.mark.asyncio
    async def test_update_rpm_only(self) -> None:
        """Updating RPM should not affect TPM."""
        limiter = TokenBucketRateLimiter(requests_per_minute=500, tokens_per_minute=150_000)
        await limiter.update_limits(rpm=1000, tpm=0)  # tpm=0 means no change
        cfg = limiter.config
        assert cfg.requests_per_minute == 1000
        assert cfg.tokens_per_minute == 150_000  # unchanged

    @pytest.mark.asyncio
    async def test_update_tpm_only(self) -> None:
        """Updating TPM should not affect RPM."""
        limiter = TokenBucketRateLimiter(requests_per_minute=500, tokens_per_minute=150_000)
        await limiter.update_limits(rpm=0, tpm=300_000)
        cfg = limiter.config
        assert cfg.requests_per_minute == 500  # unchanged
        assert cfg.tokens_per_minute == 300_000

    @pytest.mark.asyncio
    async def test_update_scales_bucket_proportionally(self) -> None:
        """After consuming half the bucket, doubling limits should keep 50% ratio."""
        limiter = TokenBucketRateLimiter(requests_per_minute=100)
        # Consume 50 of 100 tokens (50%)
        for _ in range(50):
            await limiter.acquire()

        snap = limiter.snapshot()
        avail_before = snap["rpm_available"]
        assert avail_before <= 51  # ~50 remaining (allow small refill)

        # Double the RPM limit
        await limiter.update_limits(rpm=200)

        cfg = limiter.config
        assert cfg.requests_per_minute == 200
        snap_after = limiter.snapshot()
        # Should have ~50% of 200 = ~100 (proportional)
        assert snap_after["rpm_available"] <= 110

    @pytest.mark.asyncio
    async def test_update_from_disabled_to_enabled(self) -> None:
        """Starting with no limits and then setting them should enable throttling."""
        limiter = TokenBucketRateLimiter()
        assert not limiter.enabled

        await limiter.update_limits(rpm=60, tpm=10_000)
        assert limiter.enabled
        cfg = limiter.config
        assert cfg.requests_per_minute == 60
        assert cfg.tokens_per_minute == 10_000
