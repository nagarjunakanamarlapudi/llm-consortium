"""Google Vertex AI provider for OpenAI-compatible models (e.g. GPT-OSS).

Vertex AI serves certain third-party models via an OpenAI-compatible
``/chat/completions`` endpoint.  This provider reuses the OpenAI SDK
(``AsyncOpenAI``) with:

- **base_url** pointing at the Vertex endpoint
- **api_key** obtained via Application Default Credentials (``google-auth``)

Batch prediction is supported via the Vertex AI ``batchPredictionJobs``
REST API.  When ``vertex_batch.enabled`` is ``True`` in the model config,
``complete_batch()`` uploads JSONL to GCS and creates a batch job.

Environment variables:
    GOOGLE_CLOUD_PROJECT  – GCP project ID  (required)
    GOOGLE_CLOUD_LOCATION – region, e.g. ``us-east5`` (default: ``us-central1``)
    VERTEX_OPENAI_ENDPOINT – full endpoint hostname override (optional)
    VERTEX_BATCH_GCS_BUCKET – GCS bucket for batch JSONL staging (optional)
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import UTC, datetime

import google.auth
import google.auth.transport.requests
import structlog
from openai import AsyncOpenAI

from consortium.config.models import ModelConfig
from consortium.providers.base import LLMRequest, LLMResponse

from .openai import OpenAIProvider

logger = structlog.get_logger(__name__)

# ── Authentication ────────────────────────────────────────────────────────────

# Cached credentials — refreshed automatically by google-auth when expired.
_credentials: google.auth.credentials.Credentials | None = None


def _get_credentials() -> google.auth.credentials.Credentials:
    """Get GCP credentials using Application Default Credentials."""
    global _credentials  # noqa: PLW0603

    if _credentials is None:
        _credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )

    # Refresh if expired or not yet fetched
    _credentials.refresh(google.auth.transport.requests.Request())
    return _credentials


def _get_access_token() -> str:
    """Get a GCP access token using Application Default Credentials."""
    creds = _get_credentials()
    if not creds.token:
        msg = "Failed to obtain GCP access token via Application Default Credentials"
        raise RuntimeError(msg)
    return creds.token


# ── Endpoint URL ──────────────────────────────────────────────────────────────


def _build_base_url() -> str:
    """Build the Vertex AI OpenAI-compatible endpoint URL."""
    override = os.environ.get("VERTEX_OPENAI_ENDPOINT")
    if override:
        return override.rstrip("/")

    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    if not project:
        msg = (
            "GOOGLE_CLOUD_PROJECT is required for Vertex OpenAI provider. "
            "Set the GOOGLE_CLOUD_PROJECT environment variable."
        )
        raise ValueError(msg)

    # For 'global' region, Vertex uses the bare hostname.
    if location == "global":
        host = "aiplatform.googleapis.com"
    else:
        host = f"{location}-aiplatform.googleapis.com"

    return (
        f"https://{host}/v1"
        f"/projects/{project}/locations/{location}/endpoints/openapi"
    )


# ── Batch prediction helpers ─────────────────────────────────────────────────

_BATCH_TERMINAL_STATES = frozenset({
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_PARTIALLY_SUCCEEDED",
    "JOB_STATE_EXPIRED",
})


def _gcs_upload(bucket: str, blob_path: str, data: bytes) -> str:
    """Upload bytes to GCS and return the gs:// URI."""
    from google.cloud import storage

    client = storage.Client()
    bucket_obj = client.bucket(bucket)
    blob = bucket_obj.blob(blob_path)
    blob.upload_from_string(data, content_type="application/jsonl")
    uri = f"gs://{bucket}/{blob_path}"
    logger.info("vertex_batch.gcs_uploaded", uri=uri, size_bytes=len(data))
    return uri


def _gcs_download(bucket: str, blob_path: str) -> bytes:
    """Download bytes from a GCS path."""
    from google.cloud import storage

    client = storage.Client()
    bucket_obj = client.bucket(bucket)
    blob = bucket_obj.blob(blob_path)
    return blob.download_as_bytes()


def _gcs_list_blobs(bucket: str, prefix: str) -> list[str]:
    """List blob names under a GCS prefix."""
    from google.cloud import storage

    client = storage.Client()
    bucket_obj = client.bucket(bucket)
    return [blob.name for blob in bucket_obj.list_blobs(prefix=prefix)]


async def _create_batch_job(
    project: str,
    location: str,
    model: str,
    input_uri: str,
    output_uri: str,
    display_name: str,
) -> dict:
    """Create a Vertex AI batch prediction job via REST API."""
    import aiohttp

    creds = _get_credentials()

    if location == "global":
        host = "aiplatform.googleapis.com"
    else:
        host = f"{location}-aiplatform.googleapis.com"

    url = (
        f"https://{host}/v1"
        f"/projects/{project}/locations/{location}/batchPredictionJobs"
    )

    body = {
        "displayName": display_name,
        "model": f"publishers/openai/models/{model.removeprefix('openai/')}",
        "inputConfig": {
            "instancesFormat": "jsonl",
            "gcsSource": {"uris": [input_uri]},
        },
        "outputConfig": {
            "predictionsFormat": "jsonl",
            "gcsDestination": {"outputUriPrefix": output_uri},
        },
    }

    headers = {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=body, headers=headers) as resp:
            resp_json = await resp.json()
            if resp.status != 200:
                msg = f"Batch job creation failed ({resp.status}): {resp_json}"
                raise RuntimeError(msg)
            logger.info(
                "vertex_batch.job_created",
                job_name=resp_json.get("name"),
                display_name=display_name,
            )
            return resp_json


async def _poll_batch_job(
    job_name: str,
    poll_interval_s: float,
    poll_timeout_s: float,
) -> dict:
    """Poll a batch prediction job until it reaches a terminal state."""
    import aiohttp

    creds = _get_credentials()

    if "global" in job_name:
        host = "aiplatform.googleapis.com"
    else:
        # Extract location from job_name: projects/X/locations/Y/batchPredictionJobs/Z
        parts = job_name.split("/")
        loc_idx = parts.index("locations") + 1
        location = parts[loc_idx]
        host = f"{location}-aiplatform.googleapis.com"

    url = f"https://{host}/v1/{job_name}"
    headers = {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json",
    }

    elapsed = 0.0
    log = logger.bind(job_name=job_name)

    while True:
        # Refresh credentials before each poll
        _get_credentials()
        headers["Authorization"] = f"Bearer {_credentials.token}"  # type: ignore[union-attr]

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as resp:
                job = await resp.json()

        state = job.get("state", "JOB_STATE_UNSPECIFIED")
        log.warning("vertex_batch.polling", state=state, elapsed_s=elapsed)

        if state in _BATCH_TERMINAL_STATES:
            return job

        if elapsed >= poll_timeout_s:
            msg = f"Batch job {job_name} did not complete within {poll_timeout_s}s"
            raise TimeoutError(msg)

        await asyncio.sleep(poll_interval_s)
        elapsed += poll_interval_s


# ── Provider ──────────────────────────────────────────────────────────────────


class VertexOpenAIProvider(OpenAIProvider):
    """Vertex AI provider for OpenAI-compatible models (GPT-OSS, etc.).

    Inherits all logic from :class:`OpenAIProvider`; only the client
    construction differs (custom base_url + GCP bearer token via ADC).

    When ``vertex_batch`` is configured and enabled, ``complete_batch()``
    uses the Vertex AI batch prediction API for 50% cost savings.
    """

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        base_url = _build_base_url()
        token = _get_access_token()
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=token,
        )
        logger.info(
            "vertex_openai.client_created",
            base_url=base_url,
            model=config.api_model,
        )

    async def complete_batch(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        """Execute batch requests.

        - If ``vertex_batch.enabled``: submit as a Vertex batch prediction job
          (JSONL staged in GCS, 50% cost savings).
        - Otherwise: fan-out to concurrent individual ``_complete_impl()`` calls.
        """
        if not requests:
            return []

        vb = self._config.vertex_batch
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
        tasks = [self._complete_impl(r) for r in requests]
        return list(await asyncio.gather(*tasks))

    async def _batch_via_vertex(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        """Submit requests via the Vertex AI batch prediction API."""
        vb = self._config.vertex_batch
        assert vb is not None  # noqa: S101

        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        location = vb.batch_location  # batch API location (often us-central1)
        batch_id = uuid.uuid4().hex[:12]
        log = logger.bind(batch_id=batch_id, batch_size=len(requests))

        log.warning(
            "vertex_batch.step1_building_jsonl",
            batch_size=len(requests),
            location=location,
            model=self._config.api_model,
        )

        # 1. Build OpenAI-format JSONL ──────────────────────────────────────
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
            model=self._config.api_model,
            input_uri=input_uri,
            output_uri=output_uri,
            display_name=f"consortium-{batch_id}",
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

        # 5. Download and parse results ─────────────────────────────────────
        output_blobs = await asyncio.to_thread(
            _gcs_list_blobs, vb.gcs_bucket, f"{gcs_prefix}/output/",
        )
        jsonl_blobs = [b for b in output_blobs if b.endswith(".jsonl")]

        if not jsonl_blobs:
            msg = f"No output JSONL found in gs://{vb.gcs_bucket}/{gcs_prefix}/output/"
            raise RuntimeError(msg)

        responses: list[LLMResponse | None] = [None] * len(requests)
        for blob_name in jsonl_blobs:
            raw = await asyncio.to_thread(_gcs_download, vb.gcs_bucket, blob_name)
            for line in raw.decode().strip().splitlines():
                record = json.loads(line)
                custom_id = record["custom_id"]
                if custom_id not in custom_id_to_index:
                    continue
                idx = custom_id_to_index[custom_id]

                # Vertex format: error="" means success; choices/usage live
                # directly under "response" (no "body" wrapper, no status_code)
                error_field = record.get("error", "")
                resp_body = record.get("response", {})

                # The response may be nested under "body" (OpenAI format)
                # or directly at top level (Vertex format) — handle both.
                body = resp_body.get("body", resp_body)

                choices = body.get("choices", [])

                if error_field or not choices:
                    log.error(
                        "vertex_batch.item_error",
                        custom_id=custom_id,
                        error=error_field,
                        status_code=resp_body.get("status_code"),
                    )
                    # Don't raise — mark as None, handle missing at the end
                    continue

                choice = choices[0]
                msg = choice.get("message", {})
                # Reasoning models may put output in reasoning_content
                content = msg.get("content") or msg.get("reasoning_content") or ""

                usage = body.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
                cached_input_tokens = (
                    usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                )

                responses[idx] = LLMResponse(
                    content=content,
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
                    batch_id=batch_id,
                    metadata=requests[idx].metadata,
                )

        missing = [i for i, r in enumerate(responses) if r is None]
        if missing:
            log.warning(
                "vertex_batch.missing_results",
                count=len(missing),
                total=len(responses),
                indices=missing[:10],
            )
            msg = f"Batch {batch_id} missing results for {len(missing)}/{len(responses)} requests"
            raise RuntimeError(msg)

        parsed = sum(1 for r in responses if r is not None)
        log.warning("vertex_batch.step5_results_parsed", parsed=parsed, total=len(responses))
        return responses  # type: ignore[return-value]

    def supports_batch(self) -> bool:
        """Vertex OpenAI endpoint supports batch when configured."""
        vb = self._config.vertex_batch
        return bool(vb and vb.enabled and vb.gcs_bucket)
