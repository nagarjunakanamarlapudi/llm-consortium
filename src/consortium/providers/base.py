"""Abstract LLM provider protocol and shared data types."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class LLMRequest:
    """Immutable request to an LLM."""

    system_prompt: str
    messages: list[dict[str, str]]  # [{"role": "user", "content": "..."}]
    model_config_id: str  # references a model config
    parameters: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)  # for tracing


@dataclass(frozen=True)
class LLMResponse:
    """Immutable response from an LLM."""

    content: str
    model: str  # actual model string returned by API
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float  # computed from model pricing config
    timestamp: datetime
    request_id: str  # provider's request ID
    cached_input_tokens: int = 0
    batch_id: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    provider_rpm_limit: int | None = None  # from response headers (OpenAI/Anthropic)
    provider_tpm_limit: int | None = None  # from response headers (OpenAI/Anthropic)


class LLMProvider(abc.ABC):
    """Protocol for all LLM providers."""

    @abc.abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a single real-time completion request."""
        ...

    @abc.abstractmethod
    async def complete_batch(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        """Submit requests via batch API. Blocks until all results are ready."""
        ...

    @abc.abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int, *, batch: bool = False) -> float:
        """Estimate cost in USD for given token counts."""
        ...

    @abc.abstractmethod
    def supports_batch(self) -> bool:
        """Whether this provider supports batch API."""
        ...
