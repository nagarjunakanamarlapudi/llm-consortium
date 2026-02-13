"""Anthropic LLM provider using the Messages API and Message Batches API."""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from datetime import UTC, datetime
from typing import Any

import structlog
from anthropic import AsyncAnthropic
from anthropic.types import Message, MessageParam


from consortium.config.models import ModelConfig
from consortium.providers.base import LLMProvider, LLMRequest, LLMResponse

logger = structlog.get_logger(__name__)


def _parse_int_header(headers: object, name: str) -> int | None:
    """Safely extract an integer header value, returning None on failure."""
    val = getattr(headers, "get", lambda *_: None)(name)
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


_TOKENS_PER_MILLION = 1_000_000

# Anthropic status codes / error types that warrant a retry.
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 529})

# Batch terminal states.
_BATCH_TERMINAL_STATES = frozenset({"ended", "canceled", "expired"})


_BATCH_POLL_INTERVAL_SECONDS = 30


def _is_retryable(exc: BaseException) -> bool:
    """Return True for transient Anthropic errors that should be retried."""
    from anthropic import APIStatusError, APIConnectionError, APITimeoutError

    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True
    if isinstance(exc, APIStatusError) and exc.status_code in _RETRYABLE_STATUS_CODES:
        return True
    return False


class AnthropicProvider(LLMProvider):
    """Anthropic provider backed by the Messages API (real-time) and Message Batches API."""

    def __init__(self, config: ModelConfig, *, api_key: str | None = None) -> None:
        self._config = config
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self._api_key:
            msg = (
                "Anthropic API key must be supplied via constructor argument "
                "or the ANTHROPIC_API_KEY environment variable."
            )
            raise ValueError(msg)
        self._client = AsyncAnthropic(api_key=self._api_key)
        self._log = logger.bind(provider="anthropic", model=config.api_model)

    # ── Public interface ──────────────────────────────────────────────────────

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a single real-time completion request via the Messages API."""
        messages = self._build_messages(request)
        system_blocks = self._build_system(request)
        params = self._merge_parameters(request)

        self._log.debug(
            "anthropic.complete.start",
            model=self._config.api_model,
            request_metadata=request.metadata,
        )

        start_ns = time.perf_counter_ns()

        # Anthropic API does not allow both temperature and top_p simultaneously.
        # Only include top_p if temperature is not set (or is None).
        create_kwargs: dict[str, Any] = {
            "model": self._config.api_model,
            "messages": messages,
            "system": system_blocks,
            "max_tokens": params["max_tokens"],
        }
        if params.get("temperature") is not None:
            create_kwargs["temperature"] = params["temperature"]
        elif params.get("top_p") is not None:
            create_kwargs["top_p"] = params["top_p"]

        raw_response = await self._client.messages.with_raw_response.create(**create_kwargs)
        latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000

        response: Message = raw_response.parse()

        # Extract rate limit headers
        rpm_limit = _parse_int_header(raw_response.headers, "anthropic-ratelimit-requests-limit")
        tpm_limit = _parse_int_header(raw_response.headers, "anthropic-ratelimit-tokens-limit")

        return self._message_to_response(
            response,
            latency_ms=latency_ms,
            provider_rpm_limit=rpm_limit,
            provider_tpm_limit=tpm_limit,
        )

    async def complete_batch(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        """Submit requests via the Anthropic Message Batches API.

        Blocks (async) until every result is ready, then returns responses
        in the same order as the input *requests*.
        """
        if not requests:
            return []

        if not self._config.supports_batch:
            self._log.warning(
                "anthropic.batch.unsupported",
                model=self._config.api_model,
                msg="Batch unsupported; falling back to concurrent real-time calls.",
            )
            return await self._fallback_concurrent(requests)

        # Build batch request items, keyed by a deterministic custom_id so we
        # can correlate results back to the original request order.
        batch_items = []
        for idx, req in enumerate(requests):
            custom_id = f"req-{idx}-{uuid.uuid4().hex[:8]}"
            messages = self._build_messages(req)
            system_blocks = self._build_system(req)
            params = self._merge_parameters(req)
            item_params: dict[str, Any] = {
                "model": self._config.api_model,
                "messages": messages,
                "system": system_blocks,
                "max_tokens": params["max_tokens"],
            }
            if params.get("temperature") is not None:
                item_params["temperature"] = params["temperature"]
            elif params.get("top_p") is not None:
                item_params["top_p"] = params["top_p"]

            batch_items.append(
                {
                    "custom_id": custom_id,
                    "params": item_params,
                }
            )

        self._log.info(
            "anthropic.batch.submit",
            count=len(batch_items),
            model=self._config.api_model,
        )

        # Submit the batch.
        batch = await self._client.messages.batches.create(requests=batch_items)
        batch_id = batch.id
        self._log.info("anthropic.batch.created", batch_id=batch_id)

        # Poll until the batch reaches a terminal state.
        batch = await self._poll_batch(batch_id)

        if batch.processing_status != "ended":
            msg = (
                f"Batch {batch_id} terminated with status "
                f"'{batch.processing_status}' instead of 'ended'."
            )
            raise RuntimeError(msg)

        # Collect results.
        results_by_custom_id: dict[str, Any] = {}
        result_stream = await self._client.messages.batches.results(batch_id)
        async for result in result_stream:
            results_by_custom_id[result.custom_id] = result

        # Map results back to the original request order.
        responses: list[LLMResponse] = []
        for item in batch_items:
            cid = item["custom_id"]
            result = results_by_custom_id.get(cid)
            if result is None:
                msg = f"Missing result for custom_id={cid} in batch {batch_id}."
                raise RuntimeError(msg)
            if result.result.type != "succeeded":
                error_info = getattr(result.result, "error", None)
                msg = (
                    f"Batch item {cid} did not succeed: "
                    f"type={result.result.type}, error={error_info}"
                )
                raise RuntimeError(msg)
            message: Message = result.result.message
            llm_response = self._message_to_response(
                message,
                latency_ms=0.0,  # Batch has no meaningful per-request latency.
                batch_id=batch_id,
            )
            responses.append(llm_response)

        self._log.info(
            "anthropic.batch.complete",
            batch_id=batch_id,
            count=len(responses),
        )
        return responses

    def estimate_cost(self, input_tokens: int, output_tokens: int, *, batch: bool = False) -> float:
        """Estimate cost in USD for given token counts."""
        pricing = self._config.pricing
        input_cost = (input_tokens / _TOKENS_PER_MILLION) * pricing.input
        output_cost = (output_tokens / _TOKENS_PER_MILLION) * pricing.output
        total = input_cost + output_cost
        if batch:
            total *= 1.0 - pricing.batch_discount
        return total

    def supports_batch(self) -> bool:  # noqa: D102
        return self._config.supports_batch

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_messages(self, request: LLMRequest) -> list[MessageParam]:
        """Convert the provider-agnostic message list to Anthropic MessageParam format."""
        return [{"role": msg["role"], "content": msg["content"]} for msg in request.messages]

    def _build_system(self, request: LLMRequest) -> list[dict[str, Any]]:
        """Build the system prompt blocks, with prompt caching when supported."""
        if not request.system_prompt:
            return []

        block: dict[str, Any] = {
            "type": "text",
            "text": request.system_prompt,
        }
        if self._config.supports_caching:
            block["cache_control"] = {"type": "ephemeral"}

        return [block]

    def _merge_parameters(self, request: LLMRequest) -> dict[str, Any]:
        """Merge per-request parameter overrides with model defaults."""
        defaults = self._config.parameters
        return {
            "temperature": request.parameters.get("temperature", defaults.temperature),
            "max_tokens": request.parameters.get("max_tokens", defaults.max_tokens),
            "top_p": request.parameters.get("top_p", defaults.top_p),
        }

    def _message_to_response(
        self,
        message: Message,
        *,
        latency_ms: float,
        batch_id: str | None = None,
        provider_rpm_limit: int | None = None,
        provider_tpm_limit: int | None = None,
    ) -> LLMResponse:
        """Convert an Anthropic Message object into our canonical LLMResponse."""
        content_parts = [block.text for block in message.content if block.type == "text"]
        content = "\n".join(content_parts)

        input_tokens = message.usage.input_tokens
        output_tokens = message.usage.output_tokens
        cached_input_tokens = getattr(message.usage, "cache_read_input_tokens", 0) or 0

        cost = self._compute_cost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            batch=batch_id is not None,
        )

        return LLMResponse(
            content=content,
            model=message.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            timestamp=datetime.now(UTC),
            request_id=message.id,
            cached_input_tokens=cached_input_tokens,
            batch_id=batch_id,
            provider_rpm_limit=provider_rpm_limit,
            provider_tpm_limit=provider_tpm_limit,
        )

    def _compute_cost(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int,
        batch: bool,
    ) -> float:
        """Compute the exact cost, accounting for cached tokens and batch discount."""
        pricing = self._config.pricing

        # Cached tokens are charged at the cached rate; remaining at the full input rate.
        non_cached_input = max(0, input_tokens - cached_input_tokens)
        input_cost = (non_cached_input / _TOKENS_PER_MILLION) * pricing.input
        cached_cost = (cached_input_tokens / _TOKENS_PER_MILLION) * pricing.cached_input
        output_cost = (output_tokens / _TOKENS_PER_MILLION) * pricing.output

        total = input_cost + cached_cost + output_cost
        if batch:
            total *= 1.0 - pricing.batch_discount
        return total

    async def _poll_batch(self, batch_id: str) -> Any:
        """Poll the batch until it reaches a terminal state."""
        while True:
            batch = await self._client.messages.batches.retrieve(batch_id)
            status = batch.processing_status
            self._log.debug(
                "anthropic.batch.poll",
                batch_id=batch_id,
                status=status,
                counts=getattr(batch, "request_counts", None),
            )
            if status in _BATCH_TERMINAL_STATES:
                return batch
            await asyncio.sleep(_BATCH_POLL_INTERVAL_SECONDS)

    async def _fallback_concurrent(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        """Execute requests concurrently via real-time API when batch is unavailable."""
        self._log.info(
            "anthropic.fallback_concurrent",
            count=len(requests),
        )
        tasks = [self.complete(req) for req in requests]
        return list(await asyncio.gather(*tasks))
