"""Google Gemini LLM provider using the google-genai SDK."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
import uuid
from collections import deque
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


_NON_ASCII_RE = re.compile(r"[^\x00-\x7f]+")


def _fingerprint_request(request_body: dict) -> str:
    """Stable fingerprint from Gemini request text content.

    Only hashes text content (not generationConfig) to avoid serialization
    differences (e.g. ``1.0`` vs ``1``).

    Non-ASCII characters are stripped before hashing because Vertex AI batch
    prediction corrupts multi-byte UTF-8 sequences (e.g. emoji) in the
    embedded request within output records, replacing them with U+FFFD
    replacement characters.  Stripping non-ASCII ensures the fingerprint
    is identical for both the original input and the (potentially mangled)
    output copy.
    """
    parts: list[str] = []
    for content in request_body.get("contents", []):
        for part in content.get("parts", []):
            if isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
    si = request_body.get("systemInstruction", {})
    for part in si.get("parts", []):
        if isinstance(part, dict) and "text" in part:
            parts.append(part["text"])
    raw = "||".join(parts)
    normalized = _NON_ASCII_RE.sub("", raw)
    return hashlib.sha256(normalized.encode()).hexdigest()[:20]


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

        vb = self._config.vertex_batch
        logger.info(
            "google.complete_batch_entry",
            batch_size=len(requests),
            model=self._config.api_model,
            will_use_vertex=bool(vb and vb.enabled and vb.gcs_bucket),
        )
        if vb and vb.enabled and vb.gcs_bucket:
            try:
                return await self._batch_via_vertex(requests)
            except Exception:
                logger.exception(
                    "vertex_batch.complete_batch_failed",
                    batch_size=len(requests),
                    gcs_bucket=vb.gcs_bucket,
                )
                raise

        # Fallback: concurrent individual calls
        log = logger.bind(batch_size=len(requests), model=self._config.api_model)
        log.info("google.batch_start")

        tasks = [self._call(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        responses: list[LLMResponse] = []
        batch_id = uuid.uuid4().hex
        errors = 0
        for idx, result in enumerate(results):
            if isinstance(result, BaseException):
                errors += 1
                log.error(
                    "google.batch_item_failed",
                    index=idx,
                    error=str(result),
                )
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
                        metadata=requests[idx].metadata,
                    )
                )
            else:
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

        if errors:
            log.warning("google.batch_done_with_errors", errors=errors, total=len(responses))
        else:
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

        log.info(
            "vertex_batch.step1_building_jsonl",
            location=location,
            model=self._config.api_model,
        )

        # 1. Build Gemini-format JSONL ──────────────────────────────────────
        lines: list[str] = []
        fp_to_indices: dict[str, deque[int]] = {}
        for idx, req in enumerate(requests):
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

            # Build fingerprint for output→input matching (Vertex AI does not
            # preserve request order for Gemini batch predictions).
            fp = _fingerprint_request(request_body)
            fp_to_indices.setdefault(fp, deque()).append(idx)

            line = json.dumps(
                {"request": request_body},
                separators=(",", ":"),
            )
            lines.append(line)

        jsonl_bytes = ("\n".join(lines) + "\n").encode()
        log.info(
            "vertex_batch.step1_jsonl_built",
            size_bytes=len(jsonl_bytes),
            requests=len(lines),
        )

        # 2. Upload JSONL to GCS ────────────────────────────────────────────
        gcs_prefix = f"vertex_batch/{batch_id}"
        input_blob = f"{gcs_prefix}/input.jsonl"
        log.info(
            "vertex_batch.step2_uploading_gcs",
            bucket=vb.gcs_bucket,
            blob=input_blob,
        )
        input_uri = await asyncio.to_thread(
            _gcs_upload, vb.gcs_bucket, input_blob, jsonl_bytes,
        )
        output_uri = f"gs://{vb.gcs_bucket}/{gcs_prefix}/output/"
        log.info(
            "vertex_batch.step2_gcs_uploaded",
            input_uri=input_uri,
            output_uri=output_uri,
        )

        # 3. Create batch prediction job ────────────────────────────────────
        log.info(
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
        log.info("vertex_batch.step3_job_created", job_name=job_name)

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

        log.info("vertex_batch.step4_job_succeeded")

        # 5. Download and parse Gemini-format results ───────────────────────
        output_blobs = await asyncio.to_thread(
            _gcs_list_blobs, vb.gcs_bucket, f"{gcs_prefix}/output/",
        )
        jsonl_blobs = [b for b in output_blobs if b.endswith(".jsonl")]

        if not jsonl_blobs:
            msg = f"No output JSONL found in gs://{vb.gcs_bucket}/{gcs_prefix}/output/"
            raise RuntimeError(msg)

        # Vertex AI does NOT preserve request order for Gemini batch — match
        # via embedded request fingerprint.  Outputs may be split across
        # multiple blobs, so we collect all records first.
        all_records: list[dict] = []
        for blob_name in jsonl_blobs:
            raw = await asyncio.to_thread(_gcs_download, vb.gcs_bucket, blob_name)
            for line in raw.decode().strip().splitlines():
                all_records.append(json.loads(line))

        # Pre-allocate response slots so results land at the correct index.
        responses: list[LLMResponse | None] = [None] * len(requests)

        for record in all_records:
            # Match this output back to its input via fingerprint.
            embedded_req = record.get("request", {})
            fp = _fingerprint_request(embedded_req)

            idx_deque = fp_to_indices.get(fp)
            if not idx_deque:
                log.error("vertex_batch.unmatched_output", fingerprint=fp)
                continue

            idx = idx_deque.popleft()

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
                responses[idx] = LLMResponse(
                    content="",
                    model=self._config.api_model,
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=0.0,
                    cost_usd=0.0,
                    timestamp=datetime.now(tz=timezone.utc),
                    request_id=uuid.uuid4().hex,
                    batch_id=batch_id,
                    metadata=requests[idx].metadata,
                )
                continue

            # Extract text from first candidate
            candidate = candidates[0]
            content_parts = candidate.get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in content_parts)

            input_tokens = usage.get("promptTokenCount", 0)
            output_tokens = usage.get("candidatesTokenCount", 0)
            cached_input_tokens = usage.get("cachedContentTokenCount", 0)

            cost_usd = self._compute_cost(
                input_tokens, output_tokens, cached_input_tokens, batch=True,
            )

            model_version = resp.get("modelVersion", self._config.api_model)

            responses[idx] = LLMResponse(
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
                metadata=requests[idx].metadata,
            )

        # Verify all input requests received a response.
        missing = [i for i, r in enumerate(responses) if r is None]
        if missing:
            log.error("vertex_batch.missing_results", count=len(missing), indices=missing[:10])
            raise RuntimeError(
                f"Batch {batch_id} missing results for {len(missing)}/{len(responses)} requests"
            )

        parsed = sum(1 for r in responses if r.content)
        log.info("vertex_batch.step5_results_parsed", parsed=parsed, total=len(responses))
        return responses  # type: ignore[return-value]

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
        self,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int,
        *,
        batch: bool = False,
    ) -> float:
        """Compute the actual cost using the pricing config.

        Cached input tokens are billed at the ``cached_input`` rate instead of
        the normal input rate.  When *batch* is True the ``batch_discount``
        fraction is applied.
        """
        pricing = self._config.pricing
        per_million = 1_000_000.0

        regular_input_tokens = input_tokens - cached_input_tokens
        input_cost = (regular_input_tokens / per_million) * pricing.input
        cached_cost = (cached_input_tokens / per_million) * pricing.cached_input
        output_cost = (output_tokens / per_million) * pricing.output

        total = input_cost + cached_cost + output_cost
        if batch and pricing.batch_discount > 0:
            total *= 1.0 - pricing.batch_discount
        return total


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

