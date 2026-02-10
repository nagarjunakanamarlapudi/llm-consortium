# Request Batching Layer — Design Document (v2)

## Problem

The consortium supports heterogeneous LLM backends — a single
experiment might use Claude for design generation, OpenAI for reviews,
and Ollama for evaluation. Each backend has different throughput
characteristics, rate limits, and native batch APIs. Today, LLM calls
from agents, eval, and coherence pipelines arrive individually at each
provider, missing the opportunity for backend-aware batching.

## Solution: Per-Provider Request Coalescing

Introduce a **`BatchingProvider`** decorator that wraps each real
provider instance with its own independent queue and flush loop.
Requests from any module (orchestration, eval, coherence) targeting
the same provider instance coalesce automatically. Different providers
run their batchers independently with backend-specific strategies.

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                        Application Layer                         │
 │  v1.designer   v2.leader   v2.reviewers   Eval Pipeline  Coher. │
 └───────┬────────────┬────────────┬──────────────┬───────────┬────┘
         │            │            │              │           │
    .complete()  .complete()  .complete()   .complete_batch() │
         │            │            │              │           │
         ▼            ▼            ▼              ▼           ▼
 ┌───────────────────────────────────────────────────────────────────┐
 │                      ProviderRegistry                             │
 │                                                                   │
 │  ┌─────────────────────┐  ┌──────────────────┐  ┌──────────────┐ │
 │  │ BatchingProvider     │  │ BatchingProvider  │  │ Batching...  │ │
 │  │ key: "anthropic:     │  │ key: "openai:    │  │ key: "ollama:│ │
 │  │  claude-sonnet-4-5"  │  │  gpt-4.1"        │  │  gpt-oss"   │ │
 │  │                      │  │                  │  │              │ │
 │  │ strategy: rate_limit │  │ strategy: native │  │ strategy:    │ │
 │  │ window: 50ms         │  │ window: 200ms    │  │  gpu_coalesce│ │
 │  │ max_batch: 20        │  │ max_batch: 100   │  │ window: 150ms│ │
 │  │ queue: [■■■■]        │  │ queue: [■■]      │  │ max_batch: 64│ │
 │  │                      │  │                  │  │ queue: [■■■] │ │
 │  └──────────┬───────────┘  └────────┬─────────┘  └──────┬───────┘ │
 └─────────────┼──────────────────────┼────────────────────┼─────────┘
               ▼                      ▼                    ▼
      ┌─────────────────┐   ┌─────────────────┐   ┌──────────────────┐
      │ AnthropicProvider│   │ OpenAIProvider   │   │ OllamaProvider   │
      │ Messages API     │   │ Chat Completions │   │ llama.cpp server │
      └─────────────────┘   └─────────────────┘   └──────────────────┘
```

## Per-Provider Batching Strategies

Each provider type has fundamentally different optimal batching:

| Provider | Backend reality | Optimal strategy | `window_ms` | `max_batch` |
|---|---|---|---|---|
| **Ollama** | llama.cpp continuous batching — more concurrent requests = one GPU forward pass serves all | GPU coalescing: maximize in-flight requests | 100–200 | 64 |
| **Anthropic** | Native Message Batches API (50% cost), plus real-time Messages API with rate limits | Dual-mode: use native batch API for large volumes, rate-limited concurrent for real-time | 50–100 | 20 |
| **OpenAI** | Native Batch API (async, 24h window, 50% cost), plus rate-limited Chat Completions | Dual-mode: same as Anthropic | 50–100 | 20 |
| **Google** | No batch API; rate-limited Gemini API | Rate-limit-aware concurrent dispatch | 50 | 10 |

The `BatchingProvider` doesn't need to know these details — it just
coalesces and forwards to `inner.complete_batch()`. Each real provider's
`complete_batch()` already implements the right dispatch strategy
(Ollama uses semaphored concurrency, Anthropic uses the Batches API,
OpenAI uses their Batch API, Google uses semaphored concurrency).

## Detailed Design

### 1. Registry Key: `provider:model`

The registry creates **one `BatchingProvider` per unique
(provider_type, api_model) pair**. This means:

- All agents using `anthropic:claude-sonnet-4-5-20250929` share one batcher
- All agents using `openai:gpt-4.1` share a different batcher
- All agents using `ollama:gpt-oss-120b` share a third batcher
- Two different Anthropic models (`claude-sonnet` vs `claude-opus`) get separate batchers (different rate limits)

```python
def _registry_key(config: ModelConfig) -> str:
    """Unique key per provider backend + model combination."""
    return f"{config.provider}:{config.api_model}"
```

### 2. `BatchingProvider` — unchanged from v1

The decorator is provider-agnostic. It queues requests and flushes
them to `inner.complete_batch()`. The inner provider handles
backend-specific dispatch.

```python
class BatchingProvider(LLMProvider):
    """Transparent request coalescing wrapper around any LLMProvider.

    One instance per provider:model combination. All agents/pipelines
    targeting the same backend share a single queue.
    """

    def __init__(
        self,
        inner: LLMProvider,
        *,
        window_ms: float = 100.0,
        max_batch_size: int = 64,
        name: str = "",            # for logging: "anthropic:claude-sonnet-4-5"
    ) -> None:
        self._inner = inner
        self._window_ms = window_ms
        self._max_batch_size = max_batch_size
        self._name = name
        self._queue: deque[_PendingRequest] = deque()
        self._flush_event = asyncio.Event()
        self._flush_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._stats = BatchingStats()
        self._closed = False

    # complete(), complete_batch(), _flush_loop(), _dispatch_batch()
    # ... identical to v1 design — pure queue/flush mechanics ...

    # Delegate non-batching methods to inner
    def estimate_cost(self, input_tokens, output_tokens, *, batch=False):
        return self._inner.estimate_cost(input_tokens, output_tokens, batch=batch)

    def supports_batch(self):
        return self._inner.supports_batch()
```

### 3. `ProviderRegistry` — the shared backbone

```python
# src/consortium/providers/registry.py

class ProviderRegistry:
    """One BatchingProvider per (provider_type, api_model) pair.

    All modules — orchestration, evaluation, coherence — obtain providers
    from this registry, ensuring requests to the same backend coalesce.

    Usage:
        registry = ProviderRegistry()
        async with registry:
            provider = registry.get(model_config)
            response = await provider.complete(request)
    """

    def __init__(self) -> None:
        self._providers: dict[str, LLMProvider] = {}
        self._batchers: list[BatchingProvider] = []

    def get(self, model_config: ModelConfig) -> LLMProvider:
        """Get or create a (possibly batching-wrapped) provider for this model.

        Multiple ModelConfigs with the same provider:api_model share one
        instance. E.g., if v1.designer and v5.leader both use
        anthropic:claude-sonnet-4-5, they get the same BatchingProvider.
        """
        key = _registry_key(model_config)
        if key not in self._providers:
            # Create the raw provider (Ollama/Anthropic/OpenAI/Google)
            inner = _create_raw_provider(model_config)

            # Wrap with batching if configured
            if model_config.batching and model_config.batching.enabled:
                batcher = BatchingProvider(
                    inner,
                    window_ms=model_config.batching.window_ms,
                    max_batch_size=model_config.batching.max_batch_size,
                    name=key,
                )
                self._batchers.append(batcher)
                self._providers[key] = batcher
            else:
                self._providers[key] = inner

            logger.info("provider_registered", key=key,
                        batching=model_config.batching.enabled
                        if model_config.batching else False)

        return self._providers[key]

    async def start_all(self) -> None:
        """Start all batcher flush loops."""
        for b in self._batchers:
            await b.start()
        logger.info("batching_started", count=len(self._batchers),
                    names=[b._name for b in self._batchers])

    async def shutdown_all(self) -> None:
        """Flush remaining queues and stop all batchers."""
        for b in self._batchers:
            await b.shutdown()
        logger.info("batching_shutdown",
                    stats={b._name: b.stats for b in self._batchers})

    async def __aenter__(self):
        await self.start_all()
        return self

    async def __aexit__(self, *exc):
        await self.shutdown_all()

    @property
    def stats(self) -> dict[str, BatchingStats]:
        return {b._name: b.stats for b in self._batchers}
```

### 4. Configuration: Per-Model Batching

Each model config carries its own batching settings, since optimal
parameters vary by backend:

```yaml
# configs/models/anthropic_sonnet.yaml
id: anthropic_sonnet
provider: anthropic
api_model: claude-sonnet-4-5-20250929
batching:
  enabled: true
  window_ms: 50        # Cloud API — don't wait too long
  max_batch_size: 20   # Anthropic rate limits

# configs/models/openai_gpt4.yaml
id: openai_gpt4
provider: openai
api_model: gpt-4.1
batching:
  enabled: true
  window_ms: 50
  max_batch_size: 20

# configs/models/ollama_gptoss.yaml
id: ollama_gptoss
provider: ollama
api_model: gpt-oss-120b
batching:
  enabled: true
  window_ms: 150       # GPU coalescing — wait longer to collect more
  max_batch_size: 64   # GPU can handle many concurrent
ollama:
  host: http://localhost:11434
  concurrency: 64      # Must match max_batch_size for full GPU utilization
  keep_alive: 30m
```

### 5. How Requests Flow: A Mixed-Provider Experiment

Consider this experiment config:

```yaml
variants:
  v2:
    agents:
      leader:
        model: anthropic_sonnet     # Claude for design
      reviewers:
        model: openai_gpt4          # OpenAI for review
        count: 3

evaluator:
  model: ollama_gptoss              # Local LLM for eval (free)

coherence:
  model: ollama_gptoss              # Same local LLM
```

With 5 repetitions running concurrently:

```
 v2-rep0.leader ──────┐
 v2-rep1.leader ──────┤
 v2-rep2.leader ──────┼──→ BatchingProvider("anthropic:claude-sonnet-4-5")
 v2-rep3.leader ──────┤    queue: [■■■■■]  → flush 5 to Anthropic API
 v2-rep4.leader ──────┘

 v2-rep0.reviewer₀ ──┐
 v2-rep0.reviewer₁ ──┤
 v2-rep0.reviewer₂ ──┤
 v2-rep1.reviewer₀ ──┼──→ BatchingProvider("openai:gpt-4.1")
 v2-rep1.reviewer₁ ──┤    queue: [■■■■■■■■■■■■■■■]  → flush 15 to OpenAI API
 v2-rep1.reviewer₂ ──┤
 ...                  ┘

 eval: 200 requests ──┐
 coherence: 200 req ──┼──→ BatchingProvider("ollama:gpt-oss-120b")
                      ┘    queue: [■■■...■■■]  → flush 64 at a time to Ollama
                           GPU processes all 64 in one forward pass
```

Each batcher runs its own independent flush loop:
- Anthropic batcher: flushes every 50ms or at 20 requests
- OpenAI batcher: flushes every 50ms or at 20 requests
- Ollama batcher: flushes every 150ms or at 64 requests

### 6. Integration Points

#### 6a. Agent Instantiation (orchestrator/factory.py)

Replace the local `_provider_cache` with the shared registry:

```python
def instantiate_agents(
    variant_config: VariantConfig,
    full_config: FullConfig,
    renderer: PromptRenderer,
    limits: LimitsConfig | None = None,
    *,
    registry: ProviderRegistry | None = None,
) -> dict[str, BaseAgent | list[BaseAgent]]:
    # ...

    def _get_provider(model_id: str) -> LLMProvider:
        model_cfg = full_config.get_model(model_id)
        if registry is not None:
            return registry.get(model_cfg)
        # Fallback: create unbatched provider (backwards compat)
        return create_provider(model_cfg)

    # ... rest unchanged ...
```

#### 6b. Evaluation Pipeline

```python
class EvaluationPipeline:
    def __init__(
        self,
        config: FullConfig,
        database: Database,
        prompts_dir: str | Path,
        *,
        registry: ProviderRegistry | None = None,
    ) -> None:
        # ...
        model_config = config.get_model(self.evaluator_config.model)
        if registry is not None:
            self.provider = registry.get(model_config)
        else:
            self.provider = create_provider(model_config)
```

Existing `_evaluate_batch()` and `_coherence_batch()` call
`self.provider.complete_batch(chunk)`. When `self.provider` is a
`BatchingProvider`, those requests join the shared queue and coalesce
with any concurrent orchestration requests to the same backend.

#### 6c. Top-Level Runner (batch/runner.py)

```python
class BatchExperimentRunner:
    async def run_batch_experiment(self, ...):
        registry = ProviderRegistry()

        async with registry:
            # All orchestration runs share the registry
            tasks = []
            for variant, task, rep in pending_runs:
                engine = OrchestratorEngine(
                    config=self.config,
                    database=self.database,
                    prompts_dir=self.prompts_dir,
                    registry=registry,
                )
                tasks.append(engine.run(variant, task, rep))

            # Concurrent execution — requests coalesce per provider
            await asyncio.gather(*tasks)

            # Eval and coherence also use the same registry
            eval_pipeline = EvaluationPipeline(
                ..., registry=registry
            )
            await eval_pipeline.run()

        # Registry shutdown flushes all queues, logs stats
```

### 7. Flush Loop Details

Each `BatchingProvider` instance runs its own `_flush_loop` as an
`asyncio.Task`. Since they're all on the same event loop, there's no
thread contention — just standard async coordination:

```
Event Loop
  ├─ Task: anthropic_batcher._flush_loop()    ← wakes every 50ms
  ├─ Task: openai_batcher._flush_loop()       ← wakes every 50ms
  ├─ Task: ollama_batcher._flush_loop()       ← wakes every 150ms
  ├─ Task: engine.run(v2, task1, rep0)        ← enqueues to anthropic + openai
  ├─ Task: engine.run(v2, task1, rep1)        ← enqueues to anthropic + openai
  ├─ Task: eval_pipeline._evaluate_batch()    ← enqueues to ollama
  └─ Task: eval_pipeline._coherence_batch()   ← enqueues to ollama
```

### 8. Error Isolation

Provider-level batching gives natural error isolation:

- If Anthropic has a rate limit spike, only the Anthropic batcher's
  futures get delayed/rejected. OpenAI and Ollama continue unaffected.
- If Ollama OOMs on a large batch, only eval/coherence futures fail.
  Design generation on Claude continues.
- Each batcher can implement provider-specific back-pressure:

```python
async def _dispatch_batch(self, batch: list[_PendingRequest]) -> None:
    try:
        responses = await self._inner.complete_batch(requests)
        for pending, response in zip(batch, responses):
            pending.future.set_result(response)
    except RateLimitError as exc:
        # Re-enqueue with exponential backoff (provider-specific)
        logger.warning("rate_limited", provider=self._name,
                       batch_size=len(batch), retry_after=exc.retry_after)
        await asyncio.sleep(exc.retry_after or 1.0)
        async with self._lock:
            self._queue.extendleft(reversed(batch))  # re-enqueue at front
    except Exception as exc:
        for pending in batch:
            if not pending.future.cancelled():
                pending.future.set_exception(exc)
```

### 9. Observability: Per-Provider Stats

End-of-experiment stats are per-provider:

```
batching_stats
  anthropic:claude-sonnet-4-5:
    total_requests=40   flushes=8    avg_batch=5.0   max_batch=5
    avg_wait_ms=32.1    batched_pct=100%

  openai:gpt-4.1:
    total_requests=120  flushes=12   avg_batch=10.0  max_batch=15
    avg_wait_ms=28.4    batched_pct=100%

  ollama:gpt-oss-120b:
    total_requests=400  flushes=7    avg_batch=57.1  max_batch=64
    avg_wait_ms=94.2    batched_pct=98.5%
```

This directly answers: "Is my Ollama GPU being saturated?" "Am I
hitting Anthropic rate limits?" "Is OpenAI the bottleneck?"

## What Changes, What Doesn't

### New files

| File | Purpose |
|---|---|
| `providers/batching.py` | `BatchingProvider`, `_PendingRequest`, `BatchingStats` |
| `providers/registry.py` | `ProviderRegistry` — one batcher per provider:model |

### Modified files

| File | Change |
|---|---|
| `config/models.py` | Add `BatchingConfig` to `ModelConfig` |
| `providers/factory.py` | Extract `_create_raw_provider()` for registry use |
| `orchestrator/factory.py` | Accept `registry` param, use instead of local cache |
| `orchestrator/engine.py` | Pass registry to `instantiate_agents()` |
| `evaluation/pipeline.py` | Accept optional `registry` param |
| `batch/runner.py` | Create `ProviderRegistry`, pass to all modules |
| Model YAML configs | Add `batching:` section |

### Unchanged files

| File | Why |
|---|---|
| `providers/ollama.py` | Inner provider — untouched |
| `providers/anthropic.py` | Inner provider — untouched |
| `providers/openai.py` | Inner provider — untouched |
| `providers/google.py` | Inner provider — untouched |
| `orchestrator/variants/v1–v8` | Call `provider.complete()` as before |
| `agents/*.py` | `_call_llm()` unchanged |

## Implementation Order

1. **`BatchingConfig`** in `config/models.py`
2. **`BatchingProvider`** in `providers/batching.py` + unit tests
3. **`ProviderRegistry`** in `providers/registry.py`
4. **`_create_raw_provider()`** refactor in `providers/factory.py`
5. **Wire registry** into `orchestrator/factory.py` + `engine.py`
6. **Wire registry** into `evaluation/pipeline.py`
7. **Wire registry** into `batch/runner.py` as lifecycle owner
8. **Model YAML configs** — add `batching:` per model
9. **Integration test** — mixed-provider experiment with stats validation
10. **Tuning** — benchmark `window_ms` and `max_batch_size` per provider

## Testing Strategy

### Unit: Coalescing works per-provider

```python
async def test_per_provider_isolation():
    """Two batchers run independently."""
    fake_anthropic = FakeProvider(name="anthropic")
    fake_ollama = FakeProvider(name="ollama")

    async with BatchingProvider(fake_anthropic, window_ms=50) as bp_a, \
               BatchingProvider(fake_ollama, window_ms=50) as bp_o:

        # Fire requests to both concurrently
        a_futures = [bp_a.complete(make_request(i)) for i in range(5)]
        o_futures = [bp_o.complete(make_request(i)) for i in range(10)]
        await asyncio.gather(*a_futures, *o_futures)

    # Each provider got exactly one batch, with only its own requests
    assert len(fake_anthropic.calls) == 1
    assert len(fake_anthropic.calls[0]) == 5
    assert len(fake_ollama.calls) == 1
    assert len(fake_ollama.calls[0]) == 10
```

### Unit: Registry deduplicates

```python
async def test_registry_shares_provider():
    """Same model config → same BatchingProvider instance."""
    registry = ProviderRegistry()
    p1 = registry.get(anthropic_sonnet_config)
    p2 = registry.get(anthropic_sonnet_config)
    assert p1 is p2  # Same instance

    p3 = registry.get(openai_gpt4_config)
    assert p3 is not p1  # Different provider → different instance
```

### Integration: Cross-module coalescing

```python
async def test_orchestrator_and_eval_share_batcher():
    """Orchestration + eval requests to same model coalesce."""
    registry = ProviderRegistry()
    async with registry:
        provider = registry.get(ollama_config)

        # Simulate: orchestrator makes 5 calls, eval makes 50
        orch = [provider.complete(r) for r in orch_requests]
        eval_ = provider.complete_batch(eval_requests)

        await asyncio.gather(*orch, eval_)

        stats = registry.stats["ollama:gpt-oss-120b"]
        assert stats.total_requests == 55
        # Should have been batched, not 55 individual calls
        assert stats.total_flushes < 10
```
