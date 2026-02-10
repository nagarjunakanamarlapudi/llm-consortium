"""Tests for the core Batcher class."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from async_batcher import Batcher, BatcherClosedError, BatchSizeError


# ── Helpers ──────────────────────────────────────────────────────────────────


async def echo_handler(batch: list[str]) -> list[str]:
    """Simple identity handler."""
    return batch


async def upper_handler(batch: list[str]) -> list[str]:
    """Handler that uppercases strings."""
    return [s.upper() for s in batch]


async def delayed_handler(batch: list[str]) -> list[str]:
    """Handler that simulates processing time."""
    await asyncio.sleep(0.05)
    return [f"processed:{s}" for s in batch]


async def failing_handler(batch: list[str]) -> list[str]:
    """Handler that always raises."""
    raise RuntimeError("handler failure")


async def wrong_count_handler(batch: list[str]) -> list[str]:
    """Handler that returns wrong number of responses."""
    return batch[:1]  # Always returns 1 item regardless of input


# ── Basic Functionality ──────────────────────────────────────────────────────


class TestBasicSubmit:
    async def test_single_submit(self) -> None:
        async with Batcher(handler=echo_handler, window_ms=50) as b:
            result = await b.submit("hello")
            assert result == "hello"

    async def test_single_submit_transforms(self) -> None:
        async with Batcher(handler=upper_handler, window_ms=50) as b:
            result = await b.submit("hello")
            assert result == "HELLO"

    async def test_submit_many(self) -> None:
        async with Batcher(handler=echo_handler, window_ms=50) as b:
            results = await b.submit_many(["a", "b", "c"])
            assert results == ["a", "b", "c"]

    async def test_submit_many_preserves_order(self) -> None:
        async with Batcher(handler=upper_handler, window_ms=50) as b:
            results = await b.submit_many(["alpha", "beta", "gamma"])
            assert results == ["ALPHA", "BETA", "GAMMA"]

    async def test_submit_many_empty_list(self) -> None:
        async with Batcher(handler=echo_handler, window_ms=50) as b:
            results = await b.submit_many([])
            assert results == []


# ── Batching Behavior ────────────────────────────────────────────────────────


class TestBatching:
    async def test_concurrent_submits_coalesce(self) -> None:
        """Multiple concurrent submits should be coalesced into fewer batches."""
        batches_seen: list[int] = []

        async def tracking_handler(batch: list[str]) -> list[str]:
            batches_seen.append(len(batch))
            return batch

        async with Batcher(
            handler=tracking_handler, window_ms=100, max_batch_size=64
        ) as b:
            tasks = [asyncio.create_task(b.submit(f"req-{i}")) for i in range(20)]
            results = await asyncio.gather(*tasks)

            assert len(results) == 20
            # Should be fewer than 20 handler calls (coalesced).
            assert len(batches_seen) < 20
            assert sum(batches_seen) == 20

    async def test_max_batch_size_triggers_flush(self) -> None:
        """Reaching max_batch_size should trigger immediate flush."""
        handler = AsyncMock(side_effect=lambda batch: batch)

        async with Batcher(handler=handler, window_ms=5000, max_batch_size=5) as b:
            # Submit exactly max_batch_size items.
            tasks = [asyncio.create_task(b.submit(f"item-{i}")) for i in range(5)]
            results = await asyncio.gather(*tasks)

            assert len(results) == 5
            # Handler should have been called (not waiting for 5s timeout).
            assert handler.call_count >= 1

    async def test_window_timeout_flushes(self) -> None:
        """Window timeout should flush even with fewer items than max_batch_size."""
        handler = AsyncMock(side_effect=lambda batch: batch)

        async with Batcher(handler=handler, window_ms=50, max_batch_size=1000) as b:
            result = await b.submit("solo-item")
            assert result == "solo-item"
            assert handler.call_count >= 1

    async def test_large_concurrent_batch(self) -> None:
        """100+ concurrent submits should produce reasonable batch counts."""
        batch_sizes: list[int] = []

        async def tracking_handler(batch: list[int]) -> list[int]:
            batch_sizes.append(len(batch))
            return [x * 2 for x in batch]

        async with Batcher(
            handler=tracking_handler, window_ms=100, max_batch_size=32
        ) as b:
            tasks = [asyncio.create_task(b.submit(i)) for i in range(100)]
            results = await asyncio.gather(*tasks)

            assert len(results) == 100
            assert results == [i * 2 for i in range(100)]
            # Far fewer than 100 handler calls.
            assert len(batch_sizes) <= 20
            assert sum(batch_sizes) == 100


# ── Error Handling ───────────────────────────────────────────────────────────


class TestErrorHandling:
    async def test_handler_exception_propagates(self) -> None:
        """Handler exception should propagate to all futures in the batch."""
        async with Batcher(handler=failing_handler, window_ms=50) as b:
            with pytest.raises(RuntimeError, match="handler failure"):
                await b.submit("anything")

    async def test_handler_exception_concurrent(self) -> None:
        """All concurrent futures should get the same handler exception."""
        async with Batcher(
            handler=failing_handler, window_ms=50, max_batch_size=10
        ) as b:
            tasks = [asyncio.create_task(b.submit(f"req-{i}")) for i in range(5)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                assert isinstance(r, RuntimeError)
                assert "handler failure" in str(r)

    async def test_wrong_response_count_raises_batch_size_error(self) -> None:
        """Handler returning wrong count should raise BatchSizeError."""
        async with Batcher(
            handler=wrong_count_handler, window_ms=50, max_batch_size=10
        ) as b:
            tasks = [asyncio.create_task(b.submit(f"req-{i}")) for i in range(3)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                assert isinstance(r, BatchSizeError)

    async def test_submit_after_close_raises(self) -> None:
        """Submitting after close should raise BatcherClosedError."""
        b = Batcher(handler=echo_handler, window_ms=50)
        async with b:
            pass  # Open and immediately close.

        with pytest.raises(BatcherClosedError):
            await b.submit("too-late")

    async def test_submit_many_after_close_raises(self) -> None:
        """submit_many after close should raise BatcherClosedError."""
        b = Batcher(handler=echo_handler, window_ms=50)
        async with b:
            pass

        with pytest.raises(BatcherClosedError):
            await b.submit_many(["a", "b"])

    async def test_submit_before_start_raises(self) -> None:
        """Submitting before entering context should raise."""
        b = Batcher(handler=echo_handler, window_ms=50)
        with pytest.raises(BatcherClosedError):
            await b.submit("too-early")


# ── Stats ────────────────────────────────────────────────────────────────────


class TestStats:
    async def test_stats_after_single_submit(self) -> None:
        async with Batcher(handler=echo_handler, window_ms=50) as b:
            await b.submit("x")
            s = b.stats
            assert s.total_requests == 1
            assert s.total_flushes == 1
            assert s.max_batch_size_seen == 1

    async def test_stats_after_concurrent_submits(self) -> None:
        async with Batcher(
            handler=echo_handler, window_ms=100, max_batch_size=64
        ) as b:
            tasks = [asyncio.create_task(b.submit(f"r{i}")) for i in range(10)]
            await asyncio.gather(*tasks)

            s = b.stats
            assert s.total_requests == 10
            assert s.total_flushes >= 1
            assert s.avg_batch_size > 0
            assert s.total_queue_wait_ms >= 0

    async def test_stats_batched_pct(self) -> None:
        """batched_pct should be > 0 when items are coalesced."""
        async with Batcher(
            handler=echo_handler, window_ms=100, max_batch_size=64
        ) as b:
            tasks = [asyncio.create_task(b.submit(f"r{i}")) for i in range(20)]
            await asyncio.gather(*tasks)

            s = b.stats
            # With 20 concurrent submits, most should be batched.
            assert s.total_requests == 20
            if s.total_flushes < 20:
                assert s.batched_pct > 0

    async def test_stats_handler_ms_tracked(self) -> None:
        async with Batcher(handler=delayed_handler, window_ms=50) as b:
            await b.submit("test")
            s = b.stats
            assert s.total_handler_ms > 0

    async def test_stats_zero_division_safe(self) -> None:
        """Stats properties should be safe with no data."""
        async with Batcher(handler=echo_handler, window_ms=50) as b:
            s = b.stats
            assert s.avg_batch_size == 0.0
            assert s.avg_queue_wait_ms == 0.0
            assert s.batched_pct == 0.0


# ── Context Manager ──────────────────────────────────────────────────────────


class TestContextManager:
    async def test_context_manager_drains_queue(self) -> None:
        """Exiting the context manager should flush remaining items."""
        results: list[str] = []

        async def collecting_handler(batch: list[str]) -> list[str]:
            results.extend(batch)
            return batch

        async with Batcher(
            handler=collecting_handler, window_ms=5000, max_batch_size=1000
        ) as b:
            # Submit but don't wait for flush — exit should drain.
            task = asyncio.create_task(b.submit("drain-me"))

        # After context exit, the task should be resolved.
        value = await task
        assert value == "drain-me"
        assert "drain-me" in results

    async def test_name_property(self) -> None:
        async with Batcher(handler=echo_handler, window_ms=50, name="test-batcher") as b:
            assert b.name == "test-batcher"
