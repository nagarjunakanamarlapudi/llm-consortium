# Batch Implementation Audit — Findings

## Executive Summary

The batch implementation **exists in code** but has **three bottlenecks** that prevent it from delivering the throughput Ollama's server-side continuous batching can offer. The biggest issue: you're feeding Ollama **2 requests at a time** when it could handle 50–200 simultaneously.

---

## Finding 1: Ollama concurrency is 2 — this is the main bottleneck

**File:** `configs/models/ollama_gptoss.yaml`
```yaml
ollama:
  concurrency: 2    # ← THIS IS THE PROBLEM
```

**File:** `src/consortium/providers/ollama.py` lines 97-100
```python
async def complete_batch(self, requests: list[LLMRequest]) -> list[LLMResponse]:
    semaphore = asyncio.Semaphore(self._ollama_cfg.concurrency)  # = 2
    async def _limited(req: LLMRequest) -> LLMResponse:
        async with semaphore:
            return await self.complete(req)
    results = await asyncio.gather(*[_limited(r) for r in requests])
```

**What happens:** Even when `_evaluate_batch()` passes 50 requests to `complete_batch()`, the semaphore only lets **2 through at a time**. The other 48 queue up client-side. Ollama never sees more than 2 concurrent requests, so its server-side continuous batching (llama.cpp's batch scheduler) has almost nothing to batch.

**Why this matters for your 120B model:** Ollama uses llama.cpp which supports continuous batching — when multiple requests arrive simultaneously, they share the GPU's attention computation. With batch size 1-2, the GPU does full forward passes for each request separately. With batch size 50+, the same forward pass serves all 50 requests' next tokens, dramatically improving throughput.

**This explains your timing:** 1 request = 13 min, 200 requests = 20 min. The 200 requests are being processed ~2 at a time, but Ollama's internal batching of those 2 gives a small speedup. If you sent all 50 (or more) at once, Ollama could batch them properly on the GPU and the total wall-clock time would be much closer to 13-15 minutes for all 200.

---

## Finding 2: Design generation "batch" mode doesn't batch at all

**File:** `src/consortium/batch/runner.py` lines 207-220
```python
# For batch mode, we submit generation steps across all pending runs,
# then review steps, etc. This requires the orchestrators to support
# step-wise execution, which is a deeper refactor.
#
# For now, we use a simpler approach: group runs by model provider
# and submit concurrent individual calls with controlled parallelism.

from consortium.orchestrator.engine import OrchestratorEngine
engine = OrchestratorEngine(...)
semaphore = asyncio.Semaphore(self.config.experiment.limits.max_concurrent_runs)

async def _run_one(vid, tid, rep):
    async with semaphore:                      # max_concurrent_runs = 5
        await engine.run(vid, tid, rep, ...)   # Each run is multi-step sequential

await asyncio.gather(*[_run_one(...) for ... in pending])
```

**What happens:** `consortium run batch` is identical to `consortium run experiment` — it runs complete multi-step orchestrations concurrently (up to 5), NOT batch API calls. The `BatchCollector` class exists but is **never used** by `BatchExperimentRunner`.

**Impact for design generation:** Each run has sequential steps (generate → review → revise). With `max_concurrent_runs=5`, you get 5 runs progressing simultaneously, each making individual LLM calls. Since each call goes through Ollama with concurrency 2, you're limited to 2 GPU inference requests at any moment.

---

## Finding 3: Eval & coherence batch paths work correctly — but starved by concurrency

**File:** `src/consortium/evaluation/pipeline.py`

The `_evaluate_batch()` and `_coherence_batch()` methods are **correctly implemented**:

1. ✅ Build all requests upfront (designs × runs_per_design)
2. ✅ Chunk by `batch_size: 50`
3. ✅ Call `provider.complete_batch(chunk)` per chunk
4. ✅ Parse responses and persist

But `complete_batch()` → Ollama → `Semaphore(2)` means each chunk of 50 is processed 2 at a time. The pipeline does the right thing; the provider throttles it.

---

## Recommended Fixes

### Fix 1 (Critical): Increase Ollama concurrency

**File:** `configs/models/ollama_gptoss.yaml`
```yaml
ollama:
  concurrency: 50   # Was: 2. Let Ollama's server-side batching work.
```

Start with 50 and tune based on GPU VRAM. Watch `nvidia-smi` — if you see OOM errors, reduce. For a 120B model you may need to stay lower (10-20) depending on available VRAM after model loading.

**You should also set Ollama's server-side parallel slots:**
```bash
OLLAMA_NUM_PARALLEL=50 ollama serve
```
By default Ollama only allows 1-4 parallel requests. This env var tells it to accept more concurrent requests for its internal batch scheduler.

### Fix 2 (Important): Increase batch_size for eval

**File:** `configs/evaluator.yaml`
```yaml
batch:
  enabled: true
  batch_size: 200   # Was: 50. Send all requests in one chunk.
```

With 200 eval requests, a batch_size of 50 creates 4 sequential chunks. Each chunk waits for all 50 to complete before the next 50 start. With batch_size=200, all requests enter the Ollama queue at once, maximizing GPU utilization.

### Fix 3 (Nice-to-have): Increase max_concurrent_runs for design generation

**File:** `configs/experiment.yaml`
```yaml
limits:
  max_concurrent_runs: 20   # Was: 5 (implicit default)
```

Since design runs are individual engine.run() calls, more concurrency means more LLM requests in flight at once, giving Ollama more to batch.

### Fix 4 (Future): Wire up BatchCollector for design generation

The `BatchCollector` class in `batch/runner.py` has the right architecture but is never used. The TODO comment in `BatchExperimentRunner.run_batch_experiment()` acknowledges this:

> "This requires the orchestrators to support step-wise execution, which is a deeper refactor."

This would be the proper solution: collect all generation-step prompts across runs → submit as one batch → collect all review-step prompts → submit as one batch. But it requires refactoring the variant orchestrators to support step-at-a-time execution rather than full-pipeline execution.

---

## Quick Validation

To confirm the diagnosis, check the Ollama server logs when running a batch:

```bash
# Terminal 1: Watch Ollama logs
OLLAMA_DEBUG=1 OLLAMA_NUM_PARALLEL=50 ollama serve 2>&1 | grep -E "batch|parallel|slot"

# Terminal 2: Run eval
consortium run evaluate --force
```

You should see log lines showing how many requests Ollama is processing simultaneously. With `concurrency: 2`, you'll see at most 2 active slots. After the fix, you should see many more.

---

## Expected Impact

| Scenario | Before (concurrency=2) | After (concurrency=50) |
|---|---|---|
| 1 eval request | ~1-2 min | ~1-2 min (same) |
| 200 eval requests | ~20 min (2 at a time) | ~3-5 min (batched on GPU) |
| 200 coherence requests | ~20 min | ~3-5 min |
| Full experiment (design gen) | Limited by 2 concurrent | Limited by GPU VRAM |

The exact speedup depends on how much VRAM is available after loading the 120B model. Continuous batching shares the KV cache, so each additional concurrent request adds only ~2-4MB of KV cache per token position.
