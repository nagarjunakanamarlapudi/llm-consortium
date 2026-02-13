"""Core Batcher implementation — transparent micro-batching for async workloads."""

from __future__ import annotations

import asyncio
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Awaitable, Callable, Generic, TypeVar

import structlog

from async_batcher._errors import BatcherClosedError, BatchSizeError
from async_batcher._stats import BatcherStats

if TYPE_CHECKING:
    from async_batcher._rate_limiter import TokenBucketRateLimiter

TReq = TypeVar("TReq")
TRes = TypeVar("TRes")

logger = structlog.get_logger(__name__)

# Default exceptions that should trigger a retry.
# Providers should raise these (or subclasses) for transient failures.
_DEFAULT_RETRYABLE = (
    ConnectionError,
    TimeoutError,
    asyncio.TimeoutError,
)


@dataclass(slots=True)
class _Pending(Generic[TReq, TRes]):
    """Internal queue entry pairing a request with its completion future."""

    request: TReq
    future: asyncio.Future[TRes]
    enqueued_at: float  # time.monotonic()


class Batcher(Generic[TReq, TRes]):
    """Transparent micro-batching for async workloads.

    Accumulates individual :meth:`submit` calls into batches, flushing when
    the time window expires or the queue reaches *max_batch_size*.  The
    *handler* function processes each batch.

    Type Parameters:
        TReq: Request type (any object).
        TRes: Response type (any object).

    Example::

        async def my_handler(batch: list[str]) -> list[str]:
            return [s.upper() for s in batch]

        async with Batcher(handler=my_handler, window_ms=50, max_batch_size=32) as b:
            result = await b.submit("hello")
            assert result == "HELLO"
    """

    def __init__(
        self,
        handler: Callable[[list[TReq]], Awaitable[list[TRes]]],
        *,
        window_ms: float = 100.0,
        max_batch_size: int = 64,
        name: str = "",
        on_flush: Callable[[int, float, int], None] | None = None,
        rate_limiter: TokenBucketRateLimiter | None = None,
        max_retries: int = 0,
        retry_backoff_base: float = 1.0,
        retryable_exceptions: tuple[type[BaseException], ...] | None = None,
    ) -> None:
        self._handler = handler
        self._window_ms = window_ms
        self._max_batch_size = max_batch_size
        self._name = name
        self._on_flush = on_flush

        # Rate limiting
        self._rate_limiter = rate_limiter

        # Retry configuration
        self._max_retries = max_retries
        self._retry_backoff_base = retry_backoff_base
        self._retryable_exceptions = retryable_exceptions or _DEFAULT_RETRYABLE

        self._queue: deque[_Pending[TReq, TRes]] = deque()
        self._stats = BatcherStats()
        self._running = False
        self._flush_event = asyncio.Event()
        self._flush_task: asyncio.Task[None] | None = None
        self._in_flight: set[asyncio.Task[None]] = set()
        self._in_flight_requests: int = 0
        self._last_flush_time: float = 0.0

        self._log = logger.bind(batcher=name or "default")

    # ── Public API ──────────────────────────────────────────────────────

    async def submit(self, request: TReq) -> TRes:
        """Submit a single request.  Blocks until the batch is flushed.

        Raises:
            BatcherClosedError: If the batcher has been shut down.
        """
        if not self._running:
            raise BatcherClosedError("Cannot submit to a closed Batcher")

        loop = asyncio.get_running_loop()
        future: asyncio.Future[TRes] = loop.create_future()
        self._queue.append(_Pending(request, future, time.monotonic()))

        if len(self._queue) >= self._max_batch_size:
            self._flush_event.set()

        return await future

    async def submit_many(self, requests: list[TReq]) -> list[TRes]:
        """Submit multiple requests.  All join the same flush cycle.

        Raises:
            BatcherClosedError: If the batcher has been shut down.
        """
        if not self._running:
            raise BatcherClosedError("Cannot submit to a closed Batcher")

        loop = asyncio.get_running_loop()
        futures: list[asyncio.Future[TRes]] = []

        for req in requests:
            fut: asyncio.Future[TRes] = loop.create_future()
            self._queue.append(_Pending(req, fut, time.monotonic()))
            futures.append(fut)

        if len(self._queue) >= self._max_batch_size:
            self._flush_event.set()

        return list(await asyncio.gather(*futures))

    @property
    def stats(self) -> BatcherStats:
        """Current cumulative statistics (read-only snapshot)."""
        return self._stats

    @property
    def name(self) -> str:
        """Human-readable name for this batcher."""
        return self._name

    @property
    def queue_depth(self) -> int:
        """Number of requests currently waiting in the queue."""
        return len(self._queue)

    @property
    def window_ms(self) -> float:
        """Configured flush window in milliseconds."""
        return self._window_ms

    @property
    def max_batch_size(self) -> int:
        """Configured maximum batch size."""
        return self._max_batch_size

    @property
    def last_flush_time(self) -> float:
        """Monotonic timestamp of the most recent flush (0.0 if none yet)."""
        return self._last_flush_time

    @property
    def in_flight_flushes(self) -> int:
        """Number of flush handler calls currently in progress."""
        return len(self._in_flight)

    @property
    def in_flight_requests(self) -> int:
        """Total requests across all in-flight handler calls."""
        return self._in_flight_requests

    # ── Lifecycle ───────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the batcher (sync version for use inside a running loop).

        Equivalent to ``await batcher.__aenter__()`` but callable from
        synchronous code that already has an active event loop.  This is
        used by :class:`BatcherGroup` to auto-start batchers registered
        after the group is already running.

        Safe to call multiple times — subsequent calls are no-ops.
        """
        if self._running:
            return
        self._running = True
        self._last_flush_time = time.monotonic()
        self._flush_event.clear()
        self._flush_task = asyncio.create_task(
            self._flush_loop(), name=f"batcher-flush-{self._name or id(self)}"
        )

    async def __aenter__(self) -> Batcher[TReq, TRes]:
        self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        # Yield control so any already-scheduled tasks (e.g. via
        # asyncio.create_task(b.submit(...))) can start and enqueue
        # their items before we mark the batcher as closed.
        await asyncio.sleep(0)

        self._running = False

        # Signal the loop to wake up and drain any remaining items.
        self._flush_event.set()

        if self._flush_task is not None:
            # Wait briefly for the flush loop to finish.
            try:
                await asyncio.wait_for(self._flush_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._flush_task.cancel()
                try:
                    await self._flush_task
                except asyncio.CancelledError:
                    pass
            self._flush_task = None

        # Wait for any in-flight flush tasks to complete.
        if self._in_flight:
            _, pending = await asyncio.wait(
                self._in_flight,
                timeout=30.0,
            )
            for t in pending:
                t.cancel()
            self._in_flight.clear()

    # ── Internals ───────────────────────────────────────────────────────

    async def _flush_loop(self) -> None:
        """Background loop: wait for timeout or event, then fire flushes.

        Flushes are fired as concurrent tasks so the loop can immediately
        start accumulating the next batch while the handler is in flight.
        """
        while self._running or len(self._queue) > 0:
            try:
                await asyncio.wait_for(
                    self._flush_event.wait(),
                    timeout=self._window_ms / 1000.0,
                )
            except asyncio.TimeoutError:
                pass  # Timeout expired — flush whatever is queued.

            self._flush_event.clear()

            # Fire flush as a concurrent task instead of awaiting it.
            if len(self._queue) > 0:
                task = asyncio.create_task(
                    self._do_flush(),
                    name=f"batcher-do-flush-{self._name or id(self)}",
                )
                self._in_flight.add(task)
                task.add_done_callback(self._in_flight.discard)

            if not self._running and len(self._queue) == 0:
                # Wait for any in-flight flushes before exiting.
                if self._in_flight:
                    await asyncio.gather(*self._in_flight, return_exceptions=True)
                break

    async def _acquire_rate_limit(self, batch_size: int) -> None:
        """Acquire rate limiter capacity for a batch of requests.

        Calls ``rate_limiter.acquire()`` once per request in the batch.
        Tracks wait statistics in ``BatcherStats``.
        """
        if self._rate_limiter is None or not self._rate_limiter.enabled:
            return

        start = time.monotonic()
        waits_before = self._rate_limiter.stats.total_waits

        for _ in range(batch_size):
            await self._rate_limiter.acquire()

        waits_after = self._rate_limiter.stats.total_waits
        new_waits = waits_after - waits_before

        if new_waits > 0:
            wait_ms = (time.monotonic() - start) * 1000.0
            self._stats.total_rate_limit_waits += new_waits
            self._stats.total_rate_limit_wait_ms += wait_ms

    async def _call_handler_with_retry(
        self,
        requests: list[TReq],
        batch_size: int,
    ) -> list[TRes]:
        """Call the handler with exponential backoff retry.

        Returns the handler's response list on success, or raises the
        last exception after all retries are exhausted.
        """
        last_exc: Exception | None = None
        attempts = 1 + self._max_retries  # first attempt + retries

        for attempt in range(attempts):
            try:
                return await self._handler(requests)
            except self._retryable_exceptions as exc:
                last_exc = exc
                if attempt < attempts - 1:
                    # Exponential backoff with full jitter
                    base_delay = self._retry_backoff_base * (2**attempt)
                    delay = random.uniform(0, base_delay)
                    self._stats.total_retries += 1

                    self._log.warning(
                        "batcher.retry",
                        attempt=attempt + 1,
                        max_retries=self._max_retries,
                        delay_s=round(delay, 2),
                        error=str(exc),
                        batch_size=batch_size,
                    )
                    await asyncio.sleep(delay)
                else:
                    # All retries exhausted
                    self._stats.total_exhausted += 1
                    self._log.error(
                        "batcher.retries_exhausted",
                        attempts=attempts,
                        error=str(exc),
                        batch_size=batch_size,
                    )
            except Exception:
                # Non-retryable exception — propagate immediately
                raise

        # Should not reach here but just in case
        if last_exc is not None:
            raise last_exc
        msg = "Unexpected state: no exception after retry exhaustion"
        raise RuntimeError(msg)

    async def _do_flush(self) -> None:
        """Drain up to *max_batch_size* entries and call the handler."""
        if len(self._queue) == 0:
            return

        # Drain
        n = min(len(self._queue), self._max_batch_size)
        batch: list[_Pending[TReq, TRes]] = [self._queue.popleft() for _ in range(n)]
        self._in_flight_requests += len(batch)

        # Measure queue wait times
        now = time.monotonic()
        queue_wait_ms = sum((now - p.enqueued_at) * 1000.0 for p in batch)

        # Acquire rate limiter capacity before calling handler
        try:
            await self._acquire_rate_limit(len(batch))
        except Exception as exc:
            # Rate limiter failure: reject all futures
            self._in_flight_requests -= len(batch)
            for p in batch:
                if not p.future.done():
                    p.future.set_exception(exc)
            return

        # Call handler (with retries if configured)
        requests = [p.request for p in batch]
        start_ns = time.perf_counter_ns()
        retried = False

        try:
            if self._max_retries > 0:
                retries_before = self._stats.total_retries
                responses = await self._call_handler_with_retry(requests, len(batch))
                if self._stats.total_retries > retries_before:
                    retried = True
                    self._stats.total_retry_successes += 1
            else:
                responses = await self._handler(requests)
        except Exception as exc:
            # Handler failure: reject all futures in this batch.
            self._in_flight_requests -= len(batch)
            self._stats.total_flushes += 1
            self._stats.total_requests += len(batch)
            self._stats.total_queue_wait_ms += queue_wait_ms
            for p in batch:
                if not p.future.done():
                    p.future.set_exception(exc)
            return

        handler_ms = (time.perf_counter_ns() - start_ns) / 1_000_000

        # Validate response count
        if len(responses) != len(batch):
            exc = BatchSizeError(
                f"Handler returned {len(responses)} responses, expected {len(batch)}"
            )
            self._in_flight_requests -= len(batch)
            self._stats.total_flushes += 1
            self._stats.total_requests += len(batch)
            self._stats.total_queue_wait_ms += queue_wait_ms
            for p in batch:
                if not p.future.done():
                    p.future.set_exception(exc)
            return

        # Success: update stats and resolve futures.
        self._in_flight_requests -= len(batch)
        self._stats.total_flushes += 1
        self._stats.total_requests += len(batch)
        self._stats.total_handler_ms += handler_ms
        self._stats.min_handler_ms = min(self._stats.min_handler_ms, handler_ms)
        self._stats.max_handler_ms = max(self._stats.max_handler_ms, handler_ms)
        self._stats.total_queue_wait_ms += queue_wait_ms

        if len(batch) > 1:
            self._stats.total_batched += len(batch)

        self._stats.max_batch_size_seen = max(self._stats.max_batch_size_seen, len(batch))
        self._last_flush_time = time.monotonic()

        if retried:
            self._log.info(
                "batcher.retry_succeeded",
                batch_size=len(batch),
                handler_ms=round(handler_ms, 1),
            )

        # Notify observer
        if self._on_flush is not None:
            self._on_flush(len(batch), handler_ms, len(self._queue))

        for p, response in zip(batch, responses):
            if not p.future.done():
                p.future.set_result(response)

        # If the queue still has items, wake the flush loop for another round.
        if len(self._queue) > 0:
            self._flush_event.set()
