"""Observable metrics for a Batcher instance."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BatcherStats:
    """Cumulative statistics for a single :class:`Batcher`.

    Updated atomically by the internal flush loop after each batch is
    processed.  All counters increase monotonically.
    """

    total_requests: int = 0
    """Number of individual requests processed (across all flushes)."""

    total_flushes: int = 0
    """Number of times the handler was called with a batch."""

    total_batched: int = 0
    """Requests that were part of a batch with size > 1."""

    max_batch_size_seen: int = 0
    """Largest batch observed so far."""

    total_handler_ms: float = 0.0
    """Cumulative wall-clock time spent inside the handler (milliseconds)."""

    min_handler_ms: float = field(default=float("inf"))
    """Shortest handler call duration (milliseconds)."""

    max_handler_ms: float = 0.0
    """Longest handler call duration (milliseconds)."""

    total_queue_wait_ms: float = 0.0
    """Cumulative time requests waited in the queue before being flushed (ms)."""

    @property
    def avg_batch_size(self) -> float:
        """Average number of requests per handler call."""
        if self.total_flushes == 0:
            return 0.0
        return self.total_requests / self.total_flushes

    @property
    def avg_handler_ms(self) -> float:
        """Average handler call duration (milliseconds)."""
        if self.total_flushes == 0:
            return 0.0
        return self.total_handler_ms / self.total_flushes

    @property
    def avg_queue_wait_ms(self) -> float:
        """Average queue wait time per request (milliseconds)."""
        if self.total_requests == 0:
            return 0.0
        return self.total_queue_wait_ms / self.total_requests

    @property
    def batched_pct(self) -> float:
        """Percentage of requests that were coalesced (batch size > 1)."""
        if self.total_requests == 0:
            return 0.0
        return (self.total_batched / self.total_requests) * 100.0
