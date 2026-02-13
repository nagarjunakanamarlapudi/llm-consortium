# Rate Limiting Guide

## Overview

Vertex AI enforces per-model RPM (requests per minute) quotas on every GCP project. This applies to **all models** available through Vertex AI Model Garden -- Google Gemini, Anthropic Claude, Meta Llama, Mistral, DeepSeek, and others. If you exceed your quota, API calls return `429 ResourceExhausted` errors.

This CLI includes a proactive rate limiter that prevents you from hitting those limits in the first place.

The rate limiter works in two layers:
1. **Proactive gate** -- A token bucket per model blocks outgoing requests to stay within RPM limits
2. **Smart retry** -- If a 429 still occurs, the client retries with exponential backoff + jitter

## How Vertex AI RPM Quotas Work

Every GCP project gets a set of per-model RPM quotas. These quotas are:

- **Per project** -- shared across all applications and service accounts in the project
- **Per model** -- each base model has its own independent limit
- **Per region** -- quotas are allocated per region (e.g. `us-central1`) for some models, global for others
- **Per API type** -- Google models use `GenerateContent`, partner models (Claude, Mistral) use `OnlinePrediction`/`rawPredict`
- **Not per user** -- if two people share a project, they share the quota

### Quota Endpoints by Provider

Vertex AI tracks quotas differently depending on the model provider:

| Provider | Models | Quota API Endpoint |
|----------|--------|--------------------|
| **Google** | Gemini 2.x, 3.x, Gemma | `GenerateContentRequestsPerMinutePerProjectPerRegionPerBaseModel` |
| **Google (global)** | Same as above | `GlobalGenerateContentRequestsPerMinutePerProjectPerBaseModel` |
| **Anthropic** | Claude Opus, Sonnet, Haiku | `GlobalOnlinePredictionRequestsPerMinutePerProjectPerBaseModel` |
| **Meta** | Llama 3.x (MaaS) | `GenerateContentRequestsPerMinutePerProjectPerRegionPerBaseModel` |
| **DeepSeek** | DeepSeek R1 (MaaS) | `GenerateContentRequestsPerMinutePerProjectPerRegionPerBaseModel` |
| **Mistral** | Mistral Large, Nemo (MaaS) | `GenerateContentRequestsPerMinutePerProjectPerRegionPerBaseModel` |
| **Google Health** | MedLM | `GenerateContentRequestsPerMinutePerProjectPerRegionPerBaseModel` |

### Typical RPM Quotas (Free/Low-Tier Projects)

#### Google Gemini Models

| Model | RPM | Notes |
|-------|-----|-------|
| gemini-2.5-flash | 15 | Recommended default |
| gemini-2.5-pro | 5 | More capable, lower quota |
| gemini-2.0-flash / 2.0-flash-001 | 15 | Stable GA model |
| gemini-3-flash-preview | 10 | Preview |
| gemini-3-pro-preview | 5 | Preview |
| gemini-1.5-flash | 1 | Legacy |
| gemini-1.5-pro | 1 | Legacy |
| gemini-experimental | 2 | Experimental |

#### Partner Models (Claude, Llama, Mistral, DeepSeek)

Partner models start with **0 RPM quota** until you explicitly enable them in the Vertex AI Model Garden. Unlike Gemini models which get a default quota automatically, partner models require:

1. Enabling the specific model in Model Garden (accepting partner terms)
2. In some cases, contacting Google Cloud Sales for quota provisioning
3. A billing account with payment method on file

Once enabled, the Cloud Quotas API will report actual RPM values. Run `vertex-gemini-cli quota fetch` after enablement to pick up the real limits.

**Typical quotas after enablement** (varies by project and billing tier):

| Provider | Example Models | Typical RPM | Notes |
|----------|---------------|-------------|-------|
| **Anthropic** | claude-sonnet-4-5, claude-opus-4-6, claude-haiku-4-5 | Varies | Requires Model Garden enablement; quota shown in Console under `OnlinePrediction` |
| **Meta** | llama-3.3-70b-instruct-maas, llama3-405b-instruct-maas | 60-100 | MaaS models; uses `GenerateContent` API |
| **DeepSeek** | deepseek-r1-0528-maas | 600 | MaaS model |
| **Mistral** | mistral-large-2411-maas, mistral-nemo-2407-maas | 60 | MaaS model |
| **Google Health** | MedLM-Large-1.5 | 60 | Requires separate enablement |

**Important:** If the Cloud Console shows `0` RPM for a partner model and says "Contact our Sales Team", the model has not been provisioned for your project. You must complete the enablement process first.

Paid-tier projects with usage history can request higher quotas.

## Prerequisites

### 1. GCP Project with Billing Enabled

Rate limits are tied to your GCP project. You need an active project with billing enabled.

```bash
# Check your current project
gcloud config get-value project

# Verify billing is enabled
gcloud billing projects describe YOUR_PROJECT_ID
```

If billing is not enabled, visit: https://console.cloud.google.com/billing

**Why billing is required:**
- Vertex AI API requires a billing-enabled project, even for free-tier usage
- Partner models (Claude, Llama, Mistral) require billing to be active
- Quota increases are only available to billing-enabled projects
- Some models (Claude, Llama MaaS) are pay-per-use with no free tier

### 2. Vertex AI API Enabled

```bash
gcloud services enable aiplatform.googleapis.com --project=YOUR_PROJECT_ID
```

### 3. Authentication (Application Default Credentials)

The quota fetch command uses ADC to authenticate with Google's APIs.

```bash
gcloud auth application-default login
```

### 4. Cloud Quotas API Enabled (for automatic quota discovery)

To automatically fetch your project's RPM limits, enable the Cloud Quotas API:

```bash
gcloud services enable cloudquotas.googleapis.com --project=YOUR_PROJECT_ID
```

Without this API, the CLI falls back to conservative default RPM values.

### 5. Partner Model Access (for Anthropic, Meta, Mistral)

Partner models require additional setup beyond just enabling the Vertex AI API:

#### Anthropic Claude on Vertex AI

1. Go to https://console.cloud.google.com/vertex-ai/model-garden
2. Search for "Claude" and select the model you want
3. Click **Enable** to accept the terms and enable the model
4. The model will use the `rawPredict` / `streamRawPredict` endpoint

```bash
# Verify Claude is accessible (example for Claude Sonnet)
curl -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  "https://us-east5-aiplatform.googleapis.com/v1/projects/YOUR_PROJECT_ID/locations/us-east5/publishers/anthropic/models/claude-sonnet-4-5:rawPredict" \
  -d '{"anthropic_version":"vertex-2023-10-16","max_tokens":10,"messages":[{"role":"user","content":"Hi"}]}'
```

**Note:** Claude models on Vertex AI are typically available in `us-east5` and `europe-west1`, not `us-central1`.

#### Meta Llama (MaaS) on Vertex AI

1. Go to Vertex AI Model Garden
2. Search for "Llama" and select the model
3. Click **Enable** -- MaaS models are served by Google's infrastructure
4. These use the standard `GenerateContent` API

#### Mistral on Vertex AI

1. Go to Vertex AI Model Garden
2. Search for "Mistral" and select the model
3. Click **Enable** to accept terms

## Fetching Your RPM Quotas

### Automatic (Recommended)

```bash
vertex-gemini-cli quota fetch
```

This command:
1. Authenticates using your Application Default Credentials
2. Calls the Cloud Quotas API for three quota endpoints:
   - `GenerateContentRequestsPerMinute` (regional) -- Gemini, Llama, DeepSeek, Mistral
   - `GlobalGenerateContentRequestsPerMinute` -- same models, global view
   - `GlobalOnlinePredictionRequestsPerMinute` -- Claude and other partner models
3. Merges API results with built-in defaults (API values take precedence)
4. Writes the result to `rate_limits.yaml`

If the Cloud Quotas API is not enabled or returns an error, the command still writes defaults to the file.

### Manual via Google Cloud Console

1. Go to https://console.cloud.google.com/iam-admin/quotas
2. Filter by **Service**: `Vertex AI API`
3. Search for `GenerateContent` (for Gemini, Llama, DeepSeek, Mistral models)
4. Search for `OnlinePrediction` (for Claude and partner models)
5. Look for entries with `per base model` in the name
6. The table shows per-model limits for your region

### Manual via gcloud CLI

```bash
# Requires gcloud alpha component
gcloud components install alpha

# List GenerateContent quotas (Gemini, Llama, DeepSeek, Mistral)
gcloud alpha services quota list \
  --service=aiplatform.googleapis.com \
  --project=YOUR_PROJECT_ID \
  --filter="quotaId:GenerateContentRequestsPerMinute"

# List OnlinePrediction quotas (Claude, partner models)
gcloud alpha services quota list \
  --service=aiplatform.googleapis.com \
  --project=YOUR_PROJECT_ID \
  --filter="quotaId:OnlinePredictionRequestsPerMinute"
```

### View Current Limits

```bash
# Show what limits the CLI is currently using
vertex-gemini-cli quota show
```

## Configuration

Rate limiting is configured in `config.yaml`:

```yaml
rate_limits:
  enabled: true                       # Set to false to disable rate limiting
  rate_limits_file: rate_limits.yaml   # File storing per-model RPM limits
  default_rpm: 5                      # Fallback RPM for unknown models
```

### Disabling Rate Limiting

Set `enabled: false` in `config.yaml`:

```yaml
rate_limits:
  enabled: false
```

Or via environment variable:

```bash
RATE_LIMIT_ENABLED=false vertex-gemini-cli realtime predict "test"
```

### Editing Limits Manually

You can edit `rate_limits.yaml` directly:

```yaml
rate_limits:
  gemini-2.5-flash:
    rpm: 30                           # increase if you have higher quota
  anthropic-claude-sonnet-4:
    rpm: 120                          # adjust to match your project's quota
  llama-3.3-70b-instruct-maas:
    rpm: 200
```

Run `vertex-gemini-cli quota fetch` to reset to API-reported values.

## How to Increase Your Quotas

### Step 1: Ensure Billing Is Active

Quota increases require an active billing account linked to your project.

```bash
# List your billing accounts
gcloud billing accounts list

# Link billing account to project
gcloud billing projects link YOUR_PROJECT_ID \
  --billing-account=YOUR_BILLING_ACCOUNT_ID
```

If you don't have a billing account:
1. Go to https://console.cloud.google.com/billing
2. Click **Create Account**
3. Add a payment method (credit card or invoice for enterprise)
4. Link it to your project

### Step 2: Build Usage History

Google requires usage history before granting quota increases. New projects start with low limits. Use the models at the default quota levels for a period before requesting increases.

### Step 3: Request via Console

1. Go to https://console.cloud.google.com/iam-admin/quotas
2. Filter for `Vertex AI API` and the relevant quota metric
3. Select the quota for your model and region
4. Click **Edit Quotas**
5. Enter the new limit and a justification
6. Submit the request

### Step 4: Request via gcloud

```bash
# Request a quota increase (example: gemini-2.5-flash to 100 RPM in us-central1)
gcloud alpha services quota update \
  --service=aiplatform.googleapis.com \
  --project=YOUR_PROJECT_ID \
  --consumer=projects/YOUR_PROJECT_ID \
  --metric=aiplatform.googleapis.com/generate_content_requests_per_minute_per_project_per_base_model \
  --unit=1/min/{project} \
  --dimensions=base_model=gemini-2.5-flash,region=us-central1 \
  --value=100
```

### Step 5: After Quota Increase Is Approved

Once Google approves the increase, re-fetch your limits:

```bash
vertex-gemini-cli quota fetch
```

The updated RPM values will be written to `rate_limits.yaml` and the rate limiter will use them immediately.

### Quota Increase Tips

- **Provide a clear justification** -- explain your use case and expected traffic
- **Request reasonable increases** -- doubling your current limit is more likely to be approved than 100x
- **Enterprise accounts** get faster approvals and higher ceilings
- **Partner model quotas** (Claude, Mistral) may require contacting the partner through Google's Model Garden or support channels
- **MaaS model quotas** (Llama, DeepSeek served by Google) follow the same process as Gemini

## How the Rate Limiter Works

### Token Bucket Algorithm

Each model gets its own token bucket:
- Bucket capacity = RPM limit (e.g. 15 tokens for gemini-2.5-flash)
- Tokens refill at `RPM / 60` per second (e.g. 0.25 tokens/sec for 15 RPM)
- Each API call consumes 1 token
- If no tokens are available, the caller blocks until one refills

This smooths out request patterns. Instead of bursting 15 requests and then waiting 60 seconds, requests are spaced evenly.

### Smart Retry

If a 429 still occurs (e.g. due to other consumers sharing the project quota), the client:
1. Checks if the error is retryable (429, 503, 500, 504 = yes; 400, 403, 404 = no)
2. Parses `Retry-After` from gRPC metadata if available
3. Otherwise calculates delay: `retry_delay * 2^(attempt-1)`, capped at 60s
4. Adds random jitter: `uniform(0, delay * 0.5)` to prevent thundering herd
5. Retries up to `max_retries` times (default: 3)

### Request Flow

```
vertex-gemini-cli realtime predict "prompt"
  -> client.generate_content()
    -> [1] Rate limiter: acquire token (blocks if bucket empty)
    -> [2] API call: model.generate_content(prompt)
    -> [3] On 429/503: smart retry with backoff + jitter
    -> [4] On 400/404: fail immediately (non-retryable)
  -> Return response
```

## Troubleshooting

### "Rate limiter timeout: could not acquire token"

The rate limiter waited 120 seconds and couldn't get a token. This means your RPM is very low or many requests are queued.

**Fix**: Check if your RPM is correct (`vertex-gemini-cli quota show`), or reduce concurrency.

### "403 Forbidden" on `quota fetch`

The Cloud Quotas API needs a quota project header and proper authentication.

```bash
# Make sure the API is enabled
gcloud services enable cloudquotas.googleapis.com --project=YOUR_PROJECT_ID

# Re-authenticate
gcloud auth application-default login
```

### "429 ResourceExhausted" during load tests

Your load test QPS exceeds the RPM quota. Reduce QPS:

```bash
# For 15 RPM, max safe QPS is 15/60 = 0.25
vertex-gemini-cli loadtest realtime --qps 0.2 --max-requests 10
```

### Claude/partner models show 0 RPM

If the Cloud Console shows 0 RPM for Claude or other partner models, the model hasn't been provisioned for your project. This is not a bug -- partner models require explicit enablement:

1. Go to **Vertex AI Model Garden** in the Cloud Console
2. Find and **Enable** the specific model (this accepts the partner's terms)
3. If the Console says "Contact our Sales Team", you need to go through Google Cloud Sales to get quota
4. After enablement, re-run `vertex-gemini-cli quota fetch` to pick up the real limits

The CLI does **not** include default RPM values for partner models because their quota starts at 0 until provisioned. If you know your actual quota, you can edit `rate_limits.yaml` manually.

### Rate limits file not found

If `rate_limits.yaml` doesn't exist, the CLI uses built-in defaults. To create it:

```bash
vertex-gemini-cli quota fetch
```

### Partner model returns "Model not found"

Partner models are region-specific:
- **Claude**: Available in `us-east5`, `europe-west1` (not `us-central1`)
- **Llama MaaS**: Available in `us-central1`
- **Mistral MaaS**: Available in `us-central1`

Ensure your `config.yaml` `gcp.location` matches the model's supported region.

## Quick Setup Checklist

- [ ] GCP project created
- [ ] Billing account linked: `gcloud billing projects link ...`
- [ ] Vertex AI API enabled: `gcloud services enable aiplatform.googleapis.com`
- [ ] Cloud Quotas API enabled: `gcloud services enable cloudquotas.googleapis.com`
- [ ] Authenticated: `gcloud auth application-default login`
- [ ] (If using Claude) Model enabled in Model Garden and correct region set
- [ ] (If using Llama/Mistral) Model enabled in Model Garden
- [ ] Fetched quotas: `vertex-gemini-cli quota fetch`
- [ ] Verified limits: `vertex-gemini-cli quota show`
- [ ] Tested: `vertex-gemini-cli realtime predict "What is 2+2?"`
