"""Shared helpers for Vertex AI batch prediction (GCS I/O, job creation, polling).

These utilities are used by both :class:`VertexOpenAIProvider` (GPT-OSS models)
and :class:`GoogleProvider` (Gemini models) to stage JSONL in GCS, create
batch prediction jobs, and poll for completion.

Environment variables:
    GOOGLE_CLOUD_PROJECT  – GCP project ID  (required)
    GOOGLE_CLOUD_LOCATION – region, e.g. ``us-east5`` (default: ``us-central1``)
"""

from __future__ import annotations

import asyncio

import google.auth
import google.auth.transport.requests
import structlog

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


# ── GCS I/O ──────────────────────────────────────────────────────────────────

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


# ── Batch job lifecycle ──────────────────────────────────────────────────────

_BATCH_TERMINAL_STATES = frozenset({
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_PARTIALLY_SUCCEEDED",
    "JOB_STATE_EXPIRED",
})


async def _create_batch_job(
    project: str,
    location: str,
    model_path: str,
    input_uri: str,
    output_uri: str,
    display_name: str,
) -> dict:
    """Create a Vertex AI batch prediction job via REST API.

    Args:
        project: GCP project ID.
        location: Region (e.g. ``us-central1``).
        model_path: Full model resource path, e.g.
            ``publishers/google/models/gemini-2.5-pro``,
            ``publishers/openai/models/gpt-oss-120b-maas``, or
            ``publishers/meta/models/llama-4-maverick-17b-128e-instruct-maas``.
        input_uri: GCS URI of the input JSONL file.
        output_uri: GCS URI prefix for output.
        display_name: Human-readable job name.
    """
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
        "model": model_path,
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
