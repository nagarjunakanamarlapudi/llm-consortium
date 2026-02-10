"""Tests for BatcherGroup — multi-batcher lifecycle management."""

from __future__ import annotations

import asyncio

import pytest

from async_batcher import Batcher, BatcherGroup


# ── Helpers ──────────────────────────────────────────────────────────────────


async def echo_handler(batch: list[str]) -> list[str]:
    return batch


async def double_handler(batch: list[int]) -> list[int]:
    return [x * 2 for x in batch]


# ── Registration ─────────────────────────────────────────────────────────────


class TestRegistration:
    def test_register_and_get(self) -> None:
        group = BatcherGroup()
        b = Batcher(handler=echo_handler, window_ms=50)
        group.register("echo", b)
        assert group.get("echo") is b

    def test_get_unknown_raises(self) -> None:
        group = BatcherGroup()
        with pytest.raises(KeyError, match="not found"):
            group.get("nonexistent")

    def test_duplicate_register_raises(self) -> None:
        group = BatcherGroup()
        b = Batcher(handler=echo_handler, window_ms=50)
        group.register("echo", b)
        with pytest.raises(ValueError, match="already registered"):
            group.register("echo", b)


# ── Lifecycle ────────────────────────────────────────────────────────────────


class TestLifecycle:
    async def test_start_and_shutdown(self) -> None:
        group = BatcherGroup()
        b1 = Batcher(handler=echo_handler, window_ms=50, name="b1")
        b2 = Batcher(handler=echo_handler, window_ms=50, name="b2")
        group.register("first", b1)
        group.register("second", b2)

        await group.start_all()
        # Both should be running.
        r1 = await b1.submit("hello")
        r2 = await b2.submit("world")
        assert r1 == "hello"
        assert r2 == "world"

        await group.shutdown_all()

    async def test_context_manager(self) -> None:
        group = BatcherGroup()
        b = Batcher(handler=echo_handler, window_ms=50)
        group.register("test", b)

        async with group:
            result = await b.submit("ctx")
            assert result == "ctx"

    async def test_multiple_batchers_concurrent(self) -> None:
        """Both batchers in a group should work concurrently."""
        group = BatcherGroup()
        b_echo = Batcher(handler=echo_handler, window_ms=50, name="echo")
        b_double = Batcher(handler=double_handler, window_ms=50, name="double")
        group.register("echo", b_echo)
        group.register("double", b_double)

        async with group:
            tasks = [
                asyncio.create_task(b_echo.submit("a")),
                asyncio.create_task(b_echo.submit("b")),
                asyncio.create_task(b_double.submit(5)),
                asyncio.create_task(b_double.submit(10)),
            ]
            results = await asyncio.gather(*tasks)
            assert results == ["a", "b", 10, 20]


# ── Stats ────────────────────────────────────────────────────────────────────


class TestGroupStats:
    async def test_stats_per_batcher(self) -> None:
        group = BatcherGroup()
        b1 = Batcher(handler=echo_handler, window_ms=50, name="alpha")
        b2 = Batcher(handler=echo_handler, window_ms=50, name="beta")
        group.register("alpha", b1)
        group.register("beta", b2)

        async with group:
            await b1.submit("x")
            await b2.submit("y")
            await b2.submit("z")

        stats = group.stats
        assert "alpha" in stats
        assert "beta" in stats
        assert stats["alpha"].total_requests == 1
        assert stats["beta"].total_requests == 2

    async def test_empty_group_stats(self) -> None:
        group = BatcherGroup()
        assert group.stats == {}
