"""ProviderRegistry — one BatchingProvider per (provider, api_model) pair."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from async_batcher import BatcherGroup, BatcherStats

from consortium.providers.base import LLMProvider
from consortium.providers.batching import BatchingProvider

if TYPE_CHECKING:
    from consortium.config.models import ModelConfig

logger = structlog.get_logger()


class ProviderRegistry:
    """Manages provider instances, optionally wrapping them with batchers.

    Each unique ``provider:api_model`` combination gets at most one provider
    instance.  When a model's :attr:`~ModelConfig.batching` is enabled, the
    raw provider is wrapped in a :class:`BatchingProvider`, and its batcher
    is registered with a shared :class:`BatcherGroup` for lifecycle
    management.

    Use as an async context manager to start/stop all batchers::

        async with ProviderRegistry() as registry:
            provider = registry.get(model_config)
            response = await provider.complete(request)
    """

    def __init__(self) -> None:
        self._providers: dict[str, LLMProvider] = {}
        self._group = BatcherGroup()
        self._running = False

    def get(self, model_config: ModelConfig) -> LLMProvider:
        """Get or create a provider for *model_config*.

        If batching is enabled on the model config, the returned provider is
        a :class:`BatchingProvider` wrapping the raw provider.  The same
        instance is returned for repeated calls with the same provider/model.
        """
        key = f"{model_config.provider}:{model_config.api_model}"

        if key in self._providers:
            return self._providers[key]

        from consortium.providers.factory import create_provider

        inner = create_provider(model_config)

        if model_config.batching and model_config.batching.enabled:
            wrapper = BatchingProvider(
                inner,
                window_ms=model_config.batching.window_ms,
                max_batch_size=model_config.batching.max_batch_size,
                name=key,
            )
            self._group.register(key, wrapper._batcher)
            self._providers[key] = wrapper
            logger.info(
                "batching_provider_created",
                provider=key,
                window_ms=model_config.batching.window_ms,
                max_batch_size=model_config.batching.max_batch_size,
            )
        else:
            self._providers[key] = inner
            logger.debug("provider_created", provider=key, batching="disabled")

        return self._providers[key]

    @property
    def stats(self) -> dict[str, BatcherStats]:
        """Batching stats for all providers with batching enabled."""
        return self._group.stats

    def batcher_live_state(self) -> dict[str, dict]:
        """Live state snapshot for all batching-enabled providers.

        Returns a dict keyed by provider name, each containing:
        - queue_depth: items currently waiting
        - window_ms: configured flush window
        - max_batch_size: configured batch limit
        - last_flush_time: monotonic timestamp of last flush
        - in_flight_flushes: number of handler calls currently in progress
        - stats: the BatcherStats object
        """
        state: dict[str, dict] = {}
        for key, provider in self._providers.items():
            if isinstance(provider, BatchingProvider):
                state[key] = {
                    "queue_depth": provider.queue_depth,
                    "window_ms": provider.window_ms,
                    "max_batch_size": provider.max_batch_size,
                    "last_flush_time": provider.last_flush_time,
                    "in_flight_flushes": provider.in_flight_flushes,
                    "in_flight_requests": provider.in_flight_requests,
                    "stats": provider.stats,
                }
        return state

    @property
    def running(self) -> bool:
        """Whether all batchers have been started."""
        return self._running

    async def __aenter__(self) -> ProviderRegistry:
        """Start all registered batchers."""
        await self._group.start_all()
        self._running = True
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Shut down all registered batchers."""
        await self._group.shutdown_all()
        self._running = False
