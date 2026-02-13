"""async-batcher — Transparent micro-batching for async Python.

Public API::

    from async_batcher import Batcher, BatcherGroup, BatcherStats
    from async_batcher import TokenBucketRateLimiter, RateLimiterStats

    async with Batcher(handler=my_fn, window_ms=100, max_batch_size=64) as b:
        response = await b.submit(request)
"""

from async_batcher._batcher import Batcher
from async_batcher._errors import BatcherClosedError, BatcherError, BatchSizeError
from async_batcher._group import BatcherGroup
from async_batcher._rate_limiter import RateLimiterStats, TokenBucketRateLimiter
from async_batcher._stats import BatcherStats

__all__ = [
    "Batcher",
    "BatcherClosedError",
    "BatcherError",
    "BatcherGroup",
    "BatcherStats",
    "BatchSizeError",
    "RateLimiterStats",
    "TokenBucketRateLimiter",
]
