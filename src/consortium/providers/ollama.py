"""Ollama LLM provider using the OpenAI-compatible API."""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import UTC, datetime

import structlog
from openai import AsyncOpenAI


from consortium.config.models import ModelConfig, OllamaConfig
from consortium.providers.base import LLMProvider, LLMRequest, LLMResponse

logger = structlog.get_logger(__name__)

_DEFAULT_OLLAMA = OllamaConfig()


class OllamaProvider(LLMProvider):
    """Provider for locally-hosted Ollama models via the OpenAI-compatible API."""

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._ollama_cfg = config.ollama or _DEFAULT_OLLAMA
        self._client = AsyncOpenAI(
            base_url=f"{self._ollama_cfg.host}/v1",
            api_key="ollama",  # Ollama ignores this but the SDK requires it
        )

    # ── public interface ────────────────────────────────────────────────

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a single real-time completion request to Ollama."""
        log = logger.bind(
            model=self._config.api_model,
            model_config_id=request.model_config_id,
        )
        log.debug("ollama.complete.start")

        messages = self._build_messages(request)
        params = self._build_params(request)

        start_ns = time.perf_counter_ns()
        response = await self._client.chat.completions.create(
            model=self._config.api_model,
            messages=messages,  # type: ignore[arg-type]
            **params,
        )
        latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000

        choice = response.choices[0]
        content = choice.message.content or ""
        usage = response.usage

        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        request_id = response.id or uuid.uuid4().hex

        log.debug(
            "ollama.complete.done",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=round(latency_ms, 1),
        )

        return LLMResponse(
            content=content,
            model=response.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=0.0,
            timestamp=datetime.now(UTC),
            request_id=request_id,
            metadata=request.metadata,
        )

    async def complete_batch(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        """Execute requests concurrently with bounded concurrency.

        Ollama does not have a native batch API, so this dispatches requests
        as concurrent async tasks gated by an ``asyncio.Semaphore`` whose
        limit comes from ``OllamaConfig.concurrency``.
        """
        if not requests:
            return []

        semaphore = asyncio.Semaphore(self._ollama_cfg.concurrency)

        async def _limited(req: LLMRequest) -> LLMResponse:
            async with semaphore:
                return await self.complete(req)

        logger.info(
            "ollama.complete_batch.start",
            count=len(requests),
            concurrency=self._ollama_cfg.concurrency,
            model=self._config.api_model,
        )

        results = await asyncio.gather(*[_limited(r) for r in requests])

        logger.info(
            "ollama.complete_batch.done",
            count=len(results),
        )

        return list(results)

    def estimate_cost(self, input_tokens: int, output_tokens: int, *, batch: bool = False) -> float:
        """Local inference is always free."""
        return 0.0

    def supports_batch(self) -> bool:
        """Ollama supports client-side concurrent batching."""
        return True

    # ── private helpers ─────────────────────────────────────────────────

    def _build_messages(self, request: LLMRequest) -> list[dict[str, str]]:
        """Prepend system prompt to the conversation messages."""
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.extend(request.messages)
        return messages

    def _build_params(self, request: LLMRequest) -> dict[str, object]:
        """Merge model defaults with per-request parameter overrides."""
        defaults = self._config.parameters
        params: dict[str, object] = {
            "temperature": defaults.temperature,
            "max_tokens": defaults.max_tokens,
            "top_p": defaults.top_p,
        }

        # Per-request overrides take precedence
        if request.parameters:
            for key in ("temperature", "max_tokens", "top_p"):
                if key in request.parameters:
                    params[key] = request.parameters[key]

        # Ollama-specific: keep_alive controls how long the model stays loaded
        if self._ollama_cfg.keep_alive:
            params["extra_body"] = {"keep_alive": self._ollama_cfg.keep_alive}

        return params
