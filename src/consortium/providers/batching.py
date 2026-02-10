"""BatchingProvider — wraps an LLMProvider with async-batcher for request coalescing."""

from __future__ import annotations

import structlog
from async_batcher import Batcher, BatcherStats

from consortium.providers.base import LLMProvider, LLMRequest, LLMResponse

logger = structlog.get_logger()


class BatchingProvider(LLMProvider):
    """Wraps an :class:`LLMProvider` with an async-batcher for request coalescing.

    This is a thin adapter: the library handles the queue/flush mechanics,
    and the inner provider's ``complete_batch()`` does the actual LLM call.

    Example::

        inner = AnthropicProvider(model_config)
        wrapped = BatchingProvider(inner, window_ms=50, max_batch_size=32)

        async with wrapped:  # starts the batcher
            response = await wrapped.complete(request)  # transparently batched
    """

    def __init__(
        self,
        inner: LLMProvider,
        *,
        window_ms: float = 100.0,
        max_batch_size: int = 64,
        name: str = "",
    ) -> None:
        self._inner = inner
        self._name = name
        self._batcher: Batcher[LLMRequest, LLMResponse] = Batcher(
            handler=inner.complete_batch,
            window_ms=window_ms,
            max_batch_size=max_batch_size,
            name=name,
            on_flush=self._on_flush,
        )

    def _on_flush(self, batch_size: int, handler_ms: float, queue_remaining: int) -> None:
        """Log each batcher flush for observability."""
        logger.debug(
            "batcher_flush",
            provider=self._name,
            batch_size=batch_size,
            handler_ms=round(handler_ms, 1),
            queue_remaining=queue_remaining,
            total_requests=self._batcher.stats.total_requests,
        )

    # ── LLMProvider interface ──────────────────────────────────────────

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Submit a single request via the batcher (transparently batched)."""
        return await self._batcher.submit(request)

    async def complete_batch(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        """Submit multiple requests via the batcher."""
        return await self._batcher.submit_many(requests)

    def estimate_cost(self, input_tokens: int, output_tokens: int, *, batch: bool = False) -> float:
        """Delegate cost estimation to the inner provider."""
        return self._inner.estimate_cost(input_tokens, output_tokens, batch=batch)

    def supports_batch(self) -> bool:
        """Delegate batch support check to the inner provider."""
        return self._inner.supports_batch()

    # ── Lifecycle ──────────────────────────────────────────────────────

    @property
    def stats(self) -> BatcherStats:
        """Current batching statistics."""
        return self._batcher.stats

    @property
    def inner(self) -> LLMProvider:
        """The unwrapped inner provider."""
        return self._inner

    @property
    def queue_depth(self) -> int:
        """Number of requests currently waiting in the batcher queue."""
        return self._batcher.queue_depth

    @property
    def window_ms(self) -> float:
        """Configured flush window in milliseconds."""
        return self._batcher.window_ms

    @property
    def max_batch_size(self) -> int:
        """Configured maximum batch size."""
        return self._batcher.max_batch_size

    @property
    def last_flush_time(self) -> float:
        """Monotonic timestamp of the most recent flush."""
        return self._batcher.last_flush_time

    @property
    def in_flight_flushes(self) -> int:
        """Number of handler calls currently in progress."""
        return self._batcher.in_flight_flushes

    @property
    def in_flight_requests(self) -> int:
        """Total requests across all in-flight handler calls."""
        return self._batcher.in_flight_requests

    async def start(self) -> None:
        """Start the batcher background flush loop."""
        await self._batcher.__aenter__()

    async def shutdown(self) -> None:
        """Shut down the batcher and flush any remaining items."""
        await self._batcher.__aexit__(None, None, None)

    async def __aenter__(self) -> BatchingProvider:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.shutdown()
