"""Google Gemini LLM provider using the google-genai SDK."""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from datetime import datetime, timezone

import structlog
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from consortium.config.models import ModelConfig

from .base import LLMProvider, LLMRequest, LLMResponse

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# ── Retry helpers ────────────────────────────────────────────────────────────

_MAX_ATTEMPTS = 6
_WAIT_MIN_SECONDS = 1
_WAIT_MAX_SECONDS = 60
_WAIT_JITTER_SECONDS = 2


def _is_retryable(exc: BaseException) -> bool:
    """Return True for transient / rate-limit errors worth retrying."""
    if isinstance(exc, genai_errors.ServerError):
        return True
    if isinstance(exc, genai_errors.ClientError):
        # 429 Too Many Requests is retryable
        return getattr(exc, "code", 0) == 429
    return False


# ── Provider ─────────────────────────────────────────────────────────────────


class GoogleProvider(LLMProvider):
    """Google Gemini provider backed by the ``google-genai`` SDK.

    Uses ``client.aio.models.generate_content`` for async completions.
    Batch is implemented as concurrent async calls because Google Gemini does
    not expose a native batch API comparable to Anthropic/OpenAI.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        api_key: str | None = None,
        vertexai: bool = False,
    ) -> None:
        self._config = config
        if vertexai:
            project = os.environ.get("GOOGLE_CLOUD_PROJECT")
            location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
            if not project:
                msg = (
                    "GOOGLE_CLOUD_PROJECT is required for Vertex AI. "
                    "Set the GOOGLE_CLOUD_PROJECT environment variable."
                )
                raise ValueError(msg)
            self._client = genai.Client(vertexai=True, project=project, location=location)
        else:
            api_key = api_key or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                msg = (
                    "Google API key is required. Pass api_key= or set the "
                    "GOOGLE_API_KEY environment variable."
                )
                raise ValueError(msg)
            self._client = genai.Client(api_key=api_key)

    # ── Public interface ─────────────────────────────────────────────────

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a single real-time completion request to Google Gemini."""
        return await self._call(request)

    async def complete_batch(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        """Execute requests concurrently (no native batch API for Gemini)."""
        if not requests:
            return []

        log = logger.bind(batch_size=len(requests), model=self._config.api_model)
        log.info("google.batch_start")

        tasks = [self._call(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        responses: list[LLMResponse] = []
        batch_id = uuid.uuid4().hex
        for idx, result in enumerate(results):
            if isinstance(result, BaseException):
                log.error(
                    "google.batch_item_failed",
                    index=idx,
                    error=str(result),
                )
                raise result
            responses.append(
                LLMResponse(
                    content=result.content,
                    model=result.model,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    latency_ms=result.latency_ms,
                    cost_usd=result.cost_usd,
                    timestamp=result.timestamp,
                    request_id=result.request_id,
                    cached_input_tokens=result.cached_input_tokens,
                    batch_id=batch_id,
                    metadata=result.metadata,
                )
            )

        log.info("google.batch_done", total=len(responses))
        return responses

    def estimate_cost(self, input_tokens: int, output_tokens: int, *, batch: bool = False) -> float:
        """Estimate cost in USD from the model's pricing config."""
        pricing = self._config.pricing
        per_million = 1_000_000.0

        input_cost = (input_tokens / per_million) * pricing.input
        output_cost = (output_tokens / per_million) * pricing.output

        total = input_cost + output_cost
        if batch:
            total *= 1.0 - pricing.batch_discount
        return total

    def supports_batch(self) -> bool:  # noqa: PLR6301
        """Gemini batch is emulated via concurrency, not a native API."""
        return self._config.supports_batch

    # ── Internal ─────────────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(_MAX_ATTEMPTS),
        wait=wait_exponential_jitter(
            initial=_WAIT_MIN_SECONDS,
            max=_WAIT_MAX_SECONDS,
            jitter=_WAIT_JITTER_SECONDS,
        ),
        reraise=True,
    )
    async def _call(self, request: LLMRequest) -> LLMResponse:
        """Low-level call with retry logic."""
        log = logger.bind(
            model=self._config.api_model,
            config_id=request.model_config_id,
        )
        log.debug("google.request_start")

        # Merge parameters: request-level overrides take precedence over config defaults.
        params = self._config.parameters
        temperature = request.parameters.get("temperature", params.temperature)
        max_tokens = request.parameters.get("max_tokens", params.max_tokens)
        top_p = request.parameters.get("top_p", params.top_p)

        # Build the contents list in google-genai format.
        contents = _build_contents(request.messages)

        config = genai_types.GenerateContentConfig(
            system_instruction=request.system_prompt or None,
            temperature=temperature,
            max_output_tokens=max_tokens,
            top_p=top_p,
        )

        t0 = time.monotonic()
        response = await self._client.aio.models.generate_content(
            model=self._config.api_model,
            contents=contents,
            config=config,
        )
        latency_ms = (time.monotonic() - t0) * 1000.0

        # Extract text from the response.
        text = response.text or ""

        # Extract token counts from usage_metadata.
        usage = response.usage_metadata
        input_tokens = (usage.prompt_token_count or 0) if usage else 0
        output_tokens = (usage.candidates_token_count or 0) if usage else 0
        cached_input_tokens = (usage.cached_content_token_count or 0) if usage else 0

        # Compute cost.
        cost_usd = self._compute_cost(input_tokens, output_tokens, cached_input_tokens)

        # Extract a request/response ID.
        request_id = response.response_id or uuid.uuid4().hex

        # Model version returned by the API.
        model_version = response.model_version or self._config.api_model

        log.debug(
            "google.request_done",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            latency_ms=round(latency_ms, 1),
            cost_usd=round(cost_usd, 6),
        )

        return LLMResponse(
            content=text,
            model=model_version,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            timestamp=datetime.now(tz=timezone.utc),
            request_id=request_id,
            cached_input_tokens=cached_input_tokens,
            metadata=request.metadata,
        )

    def _compute_cost(
        self, input_tokens: int, output_tokens: int, cached_input_tokens: int
    ) -> float:
        """Compute the actual cost using the pricing config.

        Cached input tokens are billed at the ``cached_input`` rate instead of
        the normal input rate.
        """
        pricing = self._config.pricing
        per_million = 1_000_000.0

        regular_input_tokens = input_tokens - cached_input_tokens
        input_cost = (regular_input_tokens / per_million) * pricing.input
        cached_cost = (cached_input_tokens / per_million) * pricing.cached_input
        output_cost = (output_tokens / per_million) * pricing.output

        return input_cost + cached_cost + output_cost


# ── Message helpers ──────────────────────────────────────────────────────────

# Mapping from the generic role names used in LLMRequest to the role names
# expected by the Gemini API.
_ROLE_MAP: dict[str, str] = {
    "user": "user",
    "assistant": "model",
    "system": "user",  # Gemini has no system role in contents; handled via system_instruction
}


def _build_contents(
    messages: list[dict[str, str]],
) -> list[genai_types.Content]:
    """Convert generic chat messages to google-genai Content objects."""
    contents: list[genai_types.Content] = []
    for msg in messages:
        role = _ROLE_MAP.get(msg["role"], "user")
        contents.append(
            genai_types.Content(
                role=role,
                parts=[genai_types.Part.from_text(text=msg["content"])],
            )
        )
    return contents
