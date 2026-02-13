# Rate Limiting & Centralized Retries

> **Status:** Fully implemented — Phases 1–7 complete.

This document describes the rate limiting, retry, and auto-detection systems
built into the LLM Consortium request pipeline.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Configuration (YAML)](#2-configuration-yaml)
3. [Token Bucket Rate Limiter](#3-token-bucket-rate-limiter)
4. [Centralized Retries](#4-centralized-retries)
5. [Auto-Detection from Response Headers](#5-auto-detection-from-response-headers)
6. [Provider-Specific Behavior](#6-provider-specific-behavior)
7. [Observability & Debugging](#7-observability--debugging)
8. [Testing](#8-testing)

---

## 1. Architecture Overview

All LLM requests flow through a three-layer pipeline:

```
LLMRequest
   │
   ▼
┌──────────────────────────────────┐
│  BatchingProvider                │  ← wraps everything
│  ┌───────────────────────────┐   │
│  │  async-batcher (Batcher)  │   │
│  │  ┌─────────────────────┐  │   │
│  │  │ TokenBucketRateLimiter│  │   │  ← proactive throttling
│  │  └─────────────────────┘  │   │
│  │  ┌─────────────────────┐  │   │
│  │  │ Retry logic (10x)   │  │   │  ← reactive error handling
│  │  └─────────────────────┘  │   │
│  └───────────────────────────┘   │
│        │                         │
│        ▼                         │
│  ┌───────────────┐               │
│  │ Inner Provider │  (OpenAI,    │  ← actual API call
│  │  .complete_batch()  Anthropic │
│  └───────────────┘    etc.)      │
└──────────────────────────────────┘
   │
   ▼
LLMResponse (with optional rate limit headers)
   │
   ▼
BatchingProvider._maybe_update_rate_limits()
   │
   ▼
TokenBucketRateLimiter.update_limits()  ← dynamic adjustment
```

**Key design decisions:**

| Decision | Rationale |
|----------|-----------|
| Rate limiting in batcher, not providers | Single point of control; providers stay simple |
| No per-provider `@retry` decorators | Retries handled centrally; eliminates double-retry bugs |
| Auto-detect only on first response | Avoids churn from fluctuating header values |
| YAML config as seed, headers as override | Ensures throttling from the very first request |

---

## 2. Configuration (YAML)

Rate limits are configured per model in `configs/models/<provider>.yaml`:

```yaml
model:
  id: "gpt-4.1"
  provider: "openai"
  api_model: "gpt-4.1-2025-04-14"
  # ...

  batching:
    enabled: true
    window_ms: 10000
    max_batch_size: 8

  rate_limits:
    requests_per_minute: 500      # RPM — 0 means unlimited
    tokens_per_minute: 2000000    # TPM — 0 means unlimited
```

### Config Model

Defined in `src/consortium/config/models.py`:

```python
class RateLimitConfig(BaseModel):
    requests_per_minute: int = 0   # 0 = no limit
    tokens_per_minute: int = 0     # 0 = no limit
```

### How Limits Are Wired

`ProviderRegistry.get(model_config)` reads `model_config.rate_limits` and
creates a `TokenBucketRateLimiter` if either RPM or TPM is > 0. This limiter
is passed to the `BatchingProvider`, which passes it to the `Batcher`.

```
ModelConfig.rate_limits → ProviderRegistry → TokenBucketRateLimiter → Batcher
```

---

## 3. Token Bucket Rate Limiter

**File:** `src/async_batcher/_rate_limiter.py`

The `TokenBucketRateLimiter` uses a dual-bucket algorithm:

| Bucket | Tracks | Refill Rate | Initial Capacity |
|--------|--------|-------------|------------------|
| Request bucket | RPM | `rpm / 60` tokens/sec | Full (= RPM) |
| Token bucket | TPM | `tpm / 60` tokens/sec | Full (= TPM) |

### How `acquire()` Works

1. Refill both buckets based on elapsed time
2. Check if request bucket has ≥ 1 token AND token bucket has ≥ `estimated_tokens`
3. If yes → consume and return immediately
4. If no → calculate wait time, release lock, sleep, retry

```python
# Every LLM request goes through this:
await rate_limiter.acquire(estimated_tokens=request_token_count)
```

### Proportional Bucket Scaling

When limits are updated (e.g., from response headers), the bucket is
**proportionally scaled** — if you had 50% capacity remaining and the limit
doubles, you'll have 50% of the new limit:

```
Before: RPM=500, bucket=250 (50% full)
Update: RPM=1000
After:  RPM=1000, bucket=500 (still 50% full)
```

---

## 4. Centralized Retries

**File:** `src/async_batcher/_batcher.py`

The `Batcher` retries failed batch handler calls with exponential backoff:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_retries` | 10 | Maximum retry attempts per batch |
| `retry_backoff_base` | 1.0 s | Base for exponential backoff (`base * 2^attempt`) |
| `retryable_exceptions` | `(Exception,)` | Which exceptions trigger a retry |

### What This Replaces

Previously, each provider had its own `@retry` decorator (via `tenacity`).
This caused **double-retry cascading** — the batcher and provider would both
retry independently. Now:

- ✅ Retries happen once, at the batcher level
- ✅ All providers share consistent retry behavior
- ✅ Rate limiting and retries are coordinated (no thundering herd on 429)
- ❌ Per-provider `@retry` removed from OpenAI, Anthropic, Google, Ollama

---

## 5. Auto-Detection from Response Headers

### How It Works

```
1. YAML seed → rate limiter initialized (e.g., RPM=500, TPM=2M)
2. First API call → provider returns LLMResponse with header-extracted limits
3. BatchingProvider._maybe_update_rate_limits() → updates limiter
4. All subsequent calls use the provider-reported limits
```

### Provider Header Mapping

| Provider | RPM Header | TPM Header |
|----------|-----------|-----------|
| **OpenAI** | `x-ratelimit-limit-requests` | `x-ratelimit-limit-tokens` |
| **Anthropic** | `anthropic-ratelimit-requests-limit` | `anthropic-ratelimit-tokens-limit` |
| Google | — (no headers) | — (uses YAML only) |
| Vertex AI | — (no headers) | — (uses YAML only) |
| Ollama | — (local, no limits) | — |

### Implementation Details

#### Provider Side (OpenAI Example)

```python
# Uses .with_raw_response to access HTTP headers
raw_response = await self._client.chat.completions.with_raw_response.create(
    model=self._config.api_model,
    messages=messages,
    **params,
)
response = raw_response.parse()

# Parse rate limit headers
rpm_limit = _parse_int_header(raw_response.headers, "x-ratelimit-limit-requests")
tpm_limit = _parse_int_header(raw_response.headers, "x-ratelimit-limit-tokens")

return LLMResponse(
    # ... response fields ...
    provider_rpm_limit=rpm_limit,  # NEW
    provider_tpm_limit=tpm_limit,  # NEW
)
```

#### LLMResponse Fields

```python
@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    # ... other fields ...
    provider_rpm_limit: int | None = None  # from response headers
    provider_tpm_limit: int | None = None  # from response headers
```

#### BatchingProvider Auto-Update

```python
# In BatchingProvider:
async def _maybe_update_rate_limits(self, responses: list[LLMResponse]) -> None:
    if self._limits_synced:       # Only update once
        return
    for resp in responses:
        if resp.provider_rpm_limit or resp.provider_tpm_limit:
            await rate_limiter.update_limits(
                rpm=resp.provider_rpm_limit or 0,
                tpm=resp.provider_tpm_limit or 0,
            )
            self._limits_synced = True
            return
```

### The `_limits_synced` Flag

The flag ensures we only update the rate limiter **once** (from the first
response that includes headers). This prevents:

- Churn from slightly different header values across requests
- Unnecessary lock acquisition on every response
- Confusion from mid-run limit changes

---

## 6. Provider-Specific Behavior

### OpenAI & Anthropic: Auto-Detect

These providers use `.with_raw_response` to capture HTTP response headers.
The detected limits reflect your **actual account-level quotas**, which may
differ from the public API docs.

**Example scenario:**
- YAML config seeds RPM=500 (conservative guess)
- First API response header says RPM=10,000 (your org's actual limit)
- Rate limiter auto-updates to RPM=10,000
- Throughput immediately increases 20×

### Google, Vertex AI, Ollama: YAML-Only Fallback

These providers don't expose rate limit headers. Throttling relies entirely on:

1. **YAML seed** — proactive throttling from configured values
2. **429 retries** — reactive backoff when limits are exceeded (batcher retry)

To tune these, adjust `rate_limits` in their YAML configs:

```yaml
# configs/models/google_gemini3.yaml
rate_limits:
  requests_per_minute: 60       # Google's free tier
  tokens_per_minute: 1000000
```

---

## 7. Observability & Debugging

### Structured Log Events

| Event | Level | Fields | When |
|-------|-------|--------|------|
| `rate_limiter.waited` | DEBUG | `wait_ms`, `tokens` | Request waited for capacity |
| `rate_limiter.limits_updated` | INFO | `old_rpm`, `new_rpm`, `old_tpm`, `new_tpm` | Limits changed via auto-detect |
| `rate_limits.auto_detected` | INFO | `provider`, `rpm`, `tpm` | BatchingProvider applied new limits |
| `batcher_flush` | DEBUG | `batch_size`, `handler_ms` | Batch sent to provider |

### Rate Limiter Snapshot

```python
limiter = TokenBucketRateLimiter(requests_per_minute=500)
snap = limiter.snapshot()
# {
#   "rpm_limit": 500,
#   "tpm_limit": 0,
#   "rpm_available": 487.3,    # current bucket level
#   "tpm_available": None,     # disabled (no TPM set)
#   "stats": {
#     "total_acquires": 13,
#     "total_waits": 0,
#     "total_wait_ms": 0.0,
#     "peak_wait_ms": 0.0
#   }
# }
```

### Stats Tracking

The `RateLimiterStats` dataclass tracks:

| Stat | Description |
|------|-------------|
| `total_acquires` | Total successful acquire() calls |
| `total_waits` | How many acquires had to wait |
| `total_wait_ms` | Cumulative wait time |
| `peak_wait_ms` | Longest single wait |
| `avg_wait_ms` | Average wait (computed property) |

---

## 8. Testing

### Unit Tests

**File:** `tests/unit/test_rate_limiter.py`

| Test | What It Verifies |
|------|-----------------|
| `test_disabled_when_no_limits` | RPM=0, TPM=0 → limiter disabled |
| `test_acquire_within_capacity` | Requests within limits don't wait |
| `test_acquire_exceeds_rpm_waits` | Exceeding RPM causes blocking |
| `test_acquire_tpm_limiting` | TPM limiting works independently |
| `test_acquire_both_limits` | Stricter limit governs |
| `test_concurrent_acquire` | Thread safety under contention |
| `test_update_limits_no_change` | Same values → no-op |
| `test_update_rpm_only` | RPM update doesn't affect TPM |
| `test_update_tpm_only` | TPM update doesn't affect RPM |
| `test_update_scales_bucket_proportionally` | 50% → 50% after doubling |
| `test_update_from_disabled_to_enabled` | Disabled → enabled on first update |

### Integration Tests

**File:** `tests/unit/test_rate_limit_integration.py`

| Test | What It Verifies |
|------|-----------------|
| `test_batching_provider_gets_rate_limiter` | Registry wires limiter from config |
| `test_no_rate_limits_no_limiter` | RPM=0 → no limiter created |
| `test_retry_config_propagated` | Retry settings flow to batcher |
| `test_transient_failure_retried` | ConnectionError retried & succeeds |
| `test_all_retries_exhausted` | Max retries → error propagated |
| `test_rate_limited_requests_succeed` | Rate-limited requests eventually succeed |
| `test_rate_limiter_updated_from_response_headers` | Auto-detect updates limiter |
| `test_limits_synced_flag_prevents_repeat_updates` | Only first response updates |

### Running Tests

```bash
# Run all rate limiting tests
pytest tests/unit/test_rate_limiter.py tests/unit/test_rate_limit_integration.py -v

# Run just the auto-detect tests
pytest tests/unit/test_rate_limit_integration.py -k "AutoDetect" -v
```
