"""Google Gemini LLM provider using the google-genai SDK."""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from datetime import datetime, timezone

import structlog
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types


from consortium.config.models import ModelConfig

from .base import LLMProvider, LLMRequest, LLMResponse

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


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
    When ``vertex_batch`` is configured and enabled, ``complete_batch()``
    uses the Vertex AI batch prediction API for cost savings and higher
    throughput.
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
        """Execute batch requests.

        - If ``vertex_batch.enabled``: submit as a Vertex batch prediction job
          (JSONL staged in GCS, cost savings and higher throughput).
        - Otherwise: fan-out to concurrent individual ``_call()`` calls.
        """
        if not requests:
            return []

        import datetime as _dt
        _dbg = open("/tmp/gemini_batch_debug.log", "a")
        vb = self._config.vertex_batch
        _dbg.write(f"\n[{_dt.datetime.now()}] complete_batch ENTRY: "
                    f"batch_size={len(requests)}, model={self._config.api_model}, "
                    f"has_vb={vb is not None}, "
                    f"vb_enabled={vb.enabled if vb else None}, "
                    f"vb_bucket={vb.gcs_bucket if vb else None}, "
                    f"will_use_vertex={bool(vb and vb.enabled and vb.gcs_bucket)}\n")
        _dbg.flush()
        logger.warning(
            "google.complete_batch_entry",
            batch_size=len(requests),
            model=self._config.api_model,
            has_vertex_batch=vb is not None,
            vb_enabled=vb.enabled if vb else None,
            vb_gcs_bucket=vb.gcs_bucket if vb else None,
            will_use_vertex=bool(vb and vb.enabled and vb.gcs_bucket),
        )
        if vb and vb.enabled and vb.gcs_bucket:
            _dbg.write(f"[{_dt.datetime.now()}] -> entering _batch_via_vertex\n")
            _dbg.flush()
            try:
                result = await self._batch_via_vertex(requests)
                _dbg.write(f"[{_dt.datetime.now()}] -> _batch_via_vertex returned {len(result)} results\n")
                _dbg.flush()
                _dbg.close()
                return result
            except Exception as e:
                _dbg.write(f"[{_dt.datetime.now()}] -> _batch_via_vertex FAILED: {e}\n")
                _dbg.flush()
                _dbg.close()
                logger.exception(
                    "vertex_batch.complete_batch_failed",
                    batch_size=len(requests),
                    gcs_bucket=vb.gcs_bucket,
                )
                raise
        _dbg.close()

        # Fallback: concurrent individual calls
        log = logger.bind(batch_size=len(requests), model=self._config.api_model)
        log.info("google.batch_start")

        tasks = [self._call(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        responses: list[LLMResponse] = []
        batch_id = uuid.uuid4().hex
        for idx, result in enumerate(results):
            if isinstance(result, BaseException):
                log.warning(
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

    def supports_batch(self) -> bool:
        """True when vertex batch or concurrent batch is available."""
        return self._config.supports_batch

    # ── Vertex AI Batch Prediction ───────────────────────────────────────

    async def _batch_via_vertex(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        """Submit requests via the Vertex AI batch prediction API.

        Uses Gemini-native JSONL format:
          Input:  ``{"request": {"contents": [...], "generationConfig": {...}}}``
          Output: ``{"status": "", "response": {"candidates": [...], "usageMetadata": {...}}}``
        """
        logger.warning(
            "google._batch_via_vertex_ENTERED",
            batch_size=len(requests),
            model=self._config.api_model,
        )
        from consortium.providers.vertex_batch_helpers import (
            _create_batch_job,
            _gcs_download,
            _gcs_list_blobs,
            _gcs_upload,
            _poll_batch_job,
        )

        vb = self._config.vertex_batch
        assert vb is not None  # noqa: S101

        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        location = vb.batch_location
        batch_id = uuid.uuid4().hex[:12]
        log = logger.bind(batch_id=batch_id, batch_size=len(requests))

        log.warning(
            "vertex_batch.step1_building_jsonl",
            batch_size=len(requests),
            location=location,
            model=self._config.api_model,
        )

        # 1. Build Gemini-format JSONL ──────────────────────────────────────
        lines: list[str] = []
        for req in requests:
            # Merge parameters
            params = self._config.parameters
            temperature = req.parameters.get("temperature", params.temperature)
            max_tokens = req.parameters.get("max_tokens", params.max_tokens)
            top_p = req.parameters.get("top_p", params.top_p)

            # Build contents in Gemini format
            contents = []
            for msg in req.messages:
                role = _ROLE_MAP.get(msg["role"], "user")
                contents.append({
                    "role": role,
                    "parts": [{"text": msg["content"]}],
                })

            request_body: dict = {
                "contents": contents,
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                    "topP": top_p,
                },
            }

            # Add system instruction if present
            if req.system_prompt:
                request_body["systemInstruction"] = {
                    "parts": [{"text": req.system_prompt}],
                }

            line = json.dumps(
                {"request": request_body},
                separators=(",", ":"),
            )
            lines.append(line)

        jsonl_bytes = ("\n".join(lines) + "\n").encode()
        log.warning(
            "vertex_batch.step1_jsonl_built",
            size_bytes=len(jsonl_bytes),
            requests=len(lines),
        )

        # 2. Upload JSONL to GCS ────────────────────────────────────────────
        gcs_prefix = f"vertex_batch/{batch_id}"
        input_blob = f"{gcs_prefix}/input.jsonl"
        log.warning(
            "vertex_batch.step2_uploading_gcs",
            bucket=vb.gcs_bucket,
            blob=input_blob,
        )
        input_uri = await asyncio.to_thread(
            _gcs_upload, vb.gcs_bucket, input_blob, jsonl_bytes,
        )
        output_uri = f"gs://{vb.gcs_bucket}/{gcs_prefix}/output/"
        log.warning(
            "vertex_batch.step2_gcs_uploaded",
            input_uri=input_uri,
            output_uri=output_uri,
        )

        # 3. Create batch prediction job ────────────────────────────────────
        log.warning(
            "vertex_batch.step3_creating_job",
            project=project,
            location=location,
            model=self._config.api_model,
        )
        job = await _create_batch_job(
            project=project,
            location=location,
            model_path=f"publishers/google/models/{self._config.api_model}",
            input_uri=input_uri,
            output_uri=output_uri,
            display_name=f"consortium-gemini-{batch_id}",
        )
        job_name = job["name"]
        log = log.bind(job_name=job_name)
        log.warning("vertex_batch.step3_job_created", job_name=job_name)

        # 4. Poll until completion ──────────────────────────────────────────
        final_job = await _poll_batch_job(
            job_name=job_name,
            poll_interval_s=vb.poll_interval_s,
            poll_timeout_s=vb.poll_timeout_s,
        )

        state = final_job.get("state", "")
        if state != "JOB_STATE_SUCCEEDED":
            error = final_job.get("error", {})
            log.error("vertex_batch.step4_job_FAILED", state=state, error=error)
            msg = f"Batch job {job_name} ended with state '{state}': {error}"
            raise RuntimeError(msg)

        log.warning("vertex_batch.step4_job_succeeded")

        # 5. Download and parse Gemini-format results ───────────────────────
        output_blobs = await asyncio.to_thread(
            _gcs_list_blobs, vb.gcs_bucket, f"{gcs_prefix}/output/",
        )
        jsonl_blobs = [b for b in output_blobs if b.endswith(".jsonl")]

        if not jsonl_blobs:
            msg = f"No output JSONL found in gs://{vb.gcs_bucket}/{gcs_prefix}/output/"
            raise RuntimeError(msg)

        # Gemini batch output preserves request order (one output per input line).
        # But outputs may be split across multiple blobs, so we collect all records.
        all_records: list[dict] = []
        for blob_name in jsonl_blobs:
            raw = await asyncio.to_thread(_gcs_download, vb.gcs_bucket, blob_name)
            for line in raw.decode().strip().splitlines():
                all_records.append(json.loads(line))

        responses: list[LLMResponse] = []
        pricing = self._config.pricing
        per_million = 1_000_000.0

        for idx, record in enumerate(all_records):
            status = record.get("status", "")
            resp = record.get("response", {})
            candidates = resp.get("candidates", [])
            usage = resp.get("usageMetadata", {})

            if status or not candidates:
                log.error(
                    "vertex_batch.item_error",
                    index=idx,
                    status=status,
                )
                # Surface the error as an empty response rather than silently skipping
                responses.append(
                    LLMResponse(
                        content="",
                        model=self._config.api_model,
                        input_tokens=0,
                        output_tokens=0,
                        latency_ms=0.0,
                        cost_usd=0.0,
                        timestamp=datetime.now(tz=timezone.utc),
                        request_id=uuid.uuid4().hex,
                        batch_id=batch_id,
                        metadata=requests[idx].metadata if idx < len(requests) else {},
                    )
                )
                continue

            # Extract text from first candidate
            candidate = candidates[0]
            content_parts = candidate.get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in content_parts)

            input_tokens = usage.get("promptTokenCount", 0)
            output_tokens = usage.get("candidatesTokenCount", 0)
            cached_input_tokens = usage.get("cachedContentTokenCount", 0)

            # Compute cost
            regular_input = input_tokens - cached_input_tokens
            cost_usd = (
                (regular_input / per_million) * pricing.input
                + (cached_input_tokens / per_million) * pricing.cached_input
                + (output_tokens / per_million) * pricing.output
            )

            model_version = resp.get("modelVersion", self._config.api_model)

            responses.append(
                LLMResponse(
                    content=text,
                    model=model_version,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=0.0,  # not meaningful for batch
                    cost_usd=cost_usd,
                    timestamp=datetime.now(tz=timezone.utc),
                    request_id=uuid.uuid4().hex,
                    cached_input_tokens=cached_input_tokens,
                    batch_id=batch_id,
                    metadata=requests[idx].metadata if idx < len(requests) else {},
                )
            )

        if len(responses) != len(requests):
            log.warning(
                "vertex_batch.response_count_mismatch",
                expected=len(requests),
                actual=len(responses),
            )

        parsed = sum(1 for r in responses if r.content)
        log.warning("vertex_batch.step5_results_parsed", parsed=parsed, total=len(responses))
        return responses

    # ── Internal ─────────────────────────────────────────────────────────

    async def _call(self, request: LLMRequest) -> LLMResponse:
        """Low-level call to Google Gemini API."""
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

