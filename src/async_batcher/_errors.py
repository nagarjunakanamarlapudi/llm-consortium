"""Exception types for the async-batcher library."""

from __future__ import annotations


class BatcherError(Exception):
    """Base exception for all batcher errors."""


class BatchSizeError(BatcherError):
    """Raised when the handler returns a different number of responses than requests.

    The handler function passed to ``Batcher`` **must** return exactly
    ``len(requests)`` responses.  When it doesn't, every future in the
    affected batch is rejected with this exception.
    """


class BatcherClosedError(BatcherError):
    """Raised when :meth:`submit` or :meth:`submit_many` is called after shutdown.

    Once the ``Batcher`` async context manager exits (or ``shutdown`` is
    called explicitly), no new requests can be enqueued.
    """
