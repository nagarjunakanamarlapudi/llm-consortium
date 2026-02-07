"""OpenAI LLM provider using the AsyncOpenAI SDK."""

from __future__ import annotations

import asyncio
import io
import json
import time
import uuid
from datetime import UTC, datetime

import structlog
from openai import AsyncOpenAI, APIConnectionError, APITimeoutError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from consortium.config.models import ModelConfig
from consortium.providers.base import LLMProvider, LLMRequest, LLMResponse

logger = structlog.get_logger(__name__)

_RETRYABLE_EXCEPTIONS = (APIConnectionError, APITimeoutError, RateLimitError)

_BATCH_TERMINAL_STATES = frozenset({"completed", "failed", "expired", "cancelled"})
_BATCH_POLL_INTERVAL_S = 30.0
_BATCH_POLL_MAX_S = 86_400.0  # 24 hours


class OpenAIProvider(LLMProvider):
    """OpenAI provider backed by the ``openai`` Python SDK.

    Supports both real-time Chat Completions and the asynchronous Batch API.
    All IO is async; the provider itself is stateless between calls.
    """

    def __init__(self, config: ModelConfig, *, api_key: str | None = None) -> None:
        self._config = config
        # When *api_key* is None the SDK reads OPENAI_API_KEY from the env.
        self._client = AsyncOpenAI(api_key=api_key)

    # ── public interface ─────────────────────────────────────────────────

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a single real-time Chat Completion request with retries."""
        return await self._complete_with_retry(request)

    async def complete_batch(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        """Submit requests via the OpenAI Batch API, poll until done, and
        return responses in the same order as *requests*.
        """
        if not requests:
            return []

        log = logger.bind(batch_size=len(requests))

        # 1. Build JSONL input --------------------------------------------------
        custom_id_to_index: dict[str, int] = {}
        lines: list[str] = []
        for idx, req in enumerate(requests):
            custom_id = f"req-{idx}-{uuid.uuid4().hex[:8]}"
            custom_id_to_index[custom_id] = idx
            body = self._build_chat_body(req)
            line = json.dumps(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                },
                separators=(",", ":"),
            )
            lines.append(line)

        jsonl_bytes = ("\n".join(lines) + "\n").encode()

        # 2. Upload file --------------------------------------------------------
        log.info("openai_batch.uploading_input_file", size_bytes=len(jsonl_bytes))
        input_file = await self._client.files.create(
            file=("batch_input.jsonl", io.BytesIO(jsonl_bytes)),
            purpose="batch",
        )

        # 3. Create batch -------------------------------------------------------
        log.info("openai_batch.creating", input_file_id=input_file.id)
        batch = await self._client.batches.create(
            input_file_id=input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        log.info("openai_batch.created", batch_id=batch.id)

        # 4. Poll until terminal state ------------------------------------------
        elapsed = 0.0
        while batch.status not in _BATCH_TERMINAL_STATES:
            await asyncio.sleep(_BATCH_POLL_INTERVAL_S)
            elapsed += _BATCH_POLL_INTERVAL_S
            batch = await self._client.batches.retrieve(batch.id)
            log.debug(
                "openai_batch.polling",
                batch_id=batch.id,
                status=batch.status,
                elapsed_s=elapsed,
            )
            if elapsed >= _BATCH_POLL_MAX_S:
                log.error("openai_batch.timeout", batch_id=batch.id)
                msg = f"Batch {batch.id} did not complete within {_BATCH_POLL_MAX_S}s"
                raise TimeoutError(msg)

        if batch.status != "completed":
            log.error(
                "openai_batch.terminal_failure",
                batch_id=batch.id,
                status=batch.status,
            )
            msg = f"Batch {batch.id} ended with status '{batch.status}'"
            raise RuntimeError(msg)

        # 5. Download results ---------------------------------------------------
        if batch.output_file_id is None:
            msg = f"Batch {batch.id} completed but has no output file"
            raise RuntimeError(msg)

        log.info("openai_batch.downloading_output", output_file_id=batch.output_file_id)
        output_bytes = await self._client.files.content(batch.output_file_id)
        raw_text = output_bytes.text

        # 6. Parse results and map back -----------------------------------------
        responses: list[LLMResponse | None] = [None] * len(requests)
        for line in raw_text.strip().splitlines():
            record = json.loads(line)
            custom_id: str = record["custom_id"]
            idx = custom_id_to_index[custom_id]

            resp_body = record.get("response", {})
            if resp_body.get("status_code") != 200:
                error_body = record.get("error", resp_body)
                log.error(
                    "openai_batch.item_error",
                    custom_id=custom_id,
                    error=error_body,
                )
                msg = f"Batch item {custom_id} failed: {error_body}"
                raise RuntimeError(msg)

            body = resp_body["body"]
            choice = body["choices"][0]
            usage = body.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            cached_input_tokens = usage.get("prompt_tokens_details", {}).get(
                "cached_tokens", 0
            )

            responses[idx] = LLMResponse(
                content=choice["message"]["content"],
                model=body.get("model", self._config.api_model),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=0.0,  # not meaningful for batch
                cost_usd=self._compute_cost(
                    input_tokens,
                    output_tokens,
                    cached_input_tokens=cached_input_tokens,
                    batch=True,
                ),
                timestamp=datetime.now(UTC),
                request_id=body.get("id", custom_id),
                cached_input_tokens=cached_input_tokens,
                batch_id=batch.id,
                metadata=requests[idx].metadata,
            )

        # Ensure every slot was filled.
        missing = [i for i, r in enumerate(responses) if r is None]
        if missing:
            msg = f"Batch {batch.id} missing results for request indices {missing}"
            raise RuntimeError(msg)

        return responses  # type: ignore[return-value]

    def estimate_cost(
        self, input_tokens: int, output_tokens: int, *, batch: bool = False
    ) -> float:
        return self._compute_cost(input_tokens, output_tokens, batch=batch)

    def supports_batch(self) -> bool:  # noqa: PLR6301
        return self._config.supports_batch

    # ── internals ────────────────────────────────────────────────────────

    def _build_chat_body(self, request: LLMRequest) -> dict:
        """Build the JSON body for a Chat Completions request."""
        merged_params = {
            "temperature": self._config.parameters.temperature,
            "max_tokens": self._config.parameters.max_tokens,
            "top_p": self._config.parameters.top_p,
            **request.parameters,
        }

        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.extend(request.messages)

        return {
            "model": self._config.api_model,
            "messages": messages,
            **merged_params,
        }

    @retry(
        retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def _complete_with_retry(self, request: LLMRequest) -> LLMResponse:
        """Execute a single Chat Completion call with tenacity retries."""
        log = logger.bind(
            model=self._config.api_model,
            config_id=request.model_config_id,
        )
        log.debug("openai.request_start")

        merged_params = {
            "temperature": self._config.parameters.temperature,
            "max_tokens": self._config.parameters.max_tokens,
            "top_p": self._config.parameters.top_p,
            **request.parameters,
        }

        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.extend(request.messages)

        t0 = time.perf_counter()
        response = await self._client.chat.completions.create(
            model=self._config.api_model,
            messages=messages,  # type: ignore[arg-type]
            **merged_params,
        )
        latency_ms = (time.perf_counter() - t0) * 1_000

        choice = response.choices[0]
        usage = response.usage

        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        cached_input_tokens = 0
        if usage and hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
            cached_input_tokens = getattr(
                usage.prompt_tokens_details, "cached_tokens", 0
            ) or 0

        cost = self._compute_cost(
            input_tokens,
            output_tokens,
            cached_input_tokens=cached_input_tokens,
        )

        log.info(
            "openai.request_complete",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            latency_ms=round(latency_ms, 1),
            cost_usd=round(cost, 6),
        )

        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            timestamp=datetime.now(UTC),
            request_id=response.id,
            cached_input_tokens=cached_input_tokens,
            metadata=request.metadata,
        )

    def _compute_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        *,
        cached_input_tokens: int = 0,
        batch: bool = False,
    ) -> float:
        """Compute cost in USD from :pyattr:`ModelConfig.pricing`.

        Pricing values in the config are **per 1 million tokens**.

        For batch requests the ``batch_discount`` fraction (e.g. 0.5 for 50 %
        off) is applied to both input and output costs.

        When ``cached_input_tokens`` is non-zero the cached portion is charged
        at the ``cached_input`` rate instead of the full ``input`` rate.
        """
        p = self._config.pricing

        billable_input = input_tokens - cached_input_tokens
        input_cost = (billable_input * p.input + cached_input_tokens * p.cached_input) / 1_000_000
        output_cost = (output_tokens * p.output) / 1_000_000
        total = input_cost + output_cost

        if batch and p.batch_discount > 0:
            total *= 1.0 - p.batch_discount

        return total
