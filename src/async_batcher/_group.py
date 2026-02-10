"""BatcherGroup — manage multiple Batcher instances with shared lifecycle."""

from __future__ import annotations

from typing import Any

from async_batcher._batcher import Batcher
from async_batcher._stats import BatcherStats


class BatcherGroup:
    """Manages multiple named :class:`Batcher` instances with shared lifecycle.

    Useful when an application has several independent batchers (e.g. one per
    LLM provider) that should be started and stopped together.

    Batchers registered **after** the group has been entered are
    automatically started on registration.

    Example::

        group = BatcherGroup()
        group.register("anthropic", anthropic_batcher)
        group.register("openai", openai_batcher)

        async with group:
            # Both batchers are running.
            # Late registrations are auto-started:
            group.register("ollama", ollama_batcher)  # started immediately
            ...

        # All batchers have been shut down.
    """

    def __init__(self) -> None:
        self._batchers: dict[str, Batcher[Any, Any]] = {}
        self._started = False

    def register(self, name: str, batcher: Batcher[Any, Any]) -> None:
        """Register a batcher under *name*.

        If the group has already been started (via ``__aenter__`` or
        ``start_all``), the batcher is started immediately by calling
        its ``start()`` method.

        Raises:
            ValueError: If *name* is already registered.
        """
        if name in self._batchers:
            msg = f"Batcher '{name}' is already registered"
            raise ValueError(msg)
        self._batchers[name] = batcher

        if self._started:
            batcher.start()

    def get(self, name: str) -> Batcher[Any, Any]:
        """Retrieve a registered batcher by *name*.

        Raises:
            KeyError: If *name* is not registered.
        """
        if name not in self._batchers:
            available = ", ".join(sorted(self._batchers)) or "(none)"
            msg = f"Batcher '{name}' not found. Registered: {available}"
            raise KeyError(msg)
        return self._batchers[name]

    async def start_all(self) -> None:
        """Start all registered batchers (call ``__aenter__``)."""
        self._started = True
        for batcher in self._batchers.values():
            await batcher.__aenter__()

    async def shutdown_all(self) -> None:
        """Shut down all registered batchers (call ``__aexit__``)."""
        self._started = False
        # Shut down in reverse registration order for clean teardown.
        for batcher in reversed(list(self._batchers.values())):
            await batcher.__aexit__(None, None, None)

    @property
    def stats(self) -> dict[str, BatcherStats]:
        """Return ``{name: stats}`` for every registered batcher."""
        return {name: b.stats for name, b in self._batchers.items()}

    # ── Async context manager ──────────────────────────────────────────

    async def __aenter__(self) -> BatcherGroup:
        await self.start_all()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.shutdown_all()
