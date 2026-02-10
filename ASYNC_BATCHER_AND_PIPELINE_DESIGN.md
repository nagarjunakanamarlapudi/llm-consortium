# Design Document: `async-batcher` Library + Streaming Pipeline

## Part 1: `async-batcher` — Standalone Open-Source Library

### Why a Library

The request coalescing pattern is universally useful:
- LLM applications (batch GPU inference, rate-limited APIs)
- Database write batching (group INSERTs for throughput)
- HTTP client batching (coalesce API calls)
- Message queue producers (batch publish)

There is no good Python library for this. The closest are:
- **aiobatch** — unmaintained, Python 3.7
- **batch-processor** — synchronous only
- ML serving systems (vLLM, Triton) — baked into C++ runtimes

### Library Identity

```
Name:     async-batcher
Tagline:  Transparent micro-batching for async Python
License:  MIT
Python:   ≥ 3.11 (for TaskGroup)
Deps:     None (pure asyncio)
```

### Core API

The library exposes ONE primitive: `Batcher[TReq, TRes]`.

```python
from async_batcher import Batcher

# Consumer provides the batch handler
async def my_handler(batch: list[MyRequest]) -> list[MyResponse]:
    """Process a batch of requests. Called by the batcher on flush."""
    return await my_backend.process_many(batch)

# Create and use
async with Batcher(handler=my_handler, window_ms=100, max_batch=64) as b:
    # From anywhere — coroutines, tasks, modules:
    response = await b.submit(request)       # single request, returns single response
    responses = await b.submit_many(reqs)    # bulk submit, returns list
```

That's it. One class, two methods. Everything else is internal.

### Type Signature

```python
from typing import TypeVar, Generic, Callable, Awaitable

TReq = TypeVar("TReq")
TRes = TypeVar("TRes")

class Batcher(Generic[TReq, TRes]):
    """Transparent micro-batching for async workloads.

    Accumulates individual submit() calls into batches, flushing when
    the time window expires or the queue reaches max_batch_size.
    The handler function processes each batch.

    Type Parameters:
        TReq: Request type (any object)
        TRes: Response type (any object)
    """

    def __init__(
        self,
        handler: Callable[[list[TReq]], Awaitable[list[TRes]]],
        *,
        window_ms: float = 100.0,
        max_batch_size: int = 64,
        name: str = "",
    ) -> None: ...

    async def submit(self, request: TReq) -> TRes:
        """Submit a single request. Blocks until the batch is flushed."""
        ...

    async def submit_many(self, requests: list[TReq]) -> list[TRes]:
        """Submit multiple requests. All join the same flush cycle."""
        ...

    @property
    def stats(self) -> BatcherStats: ...

    async def __aenter__(self) -> Self: ...
    async def __aexit__(self, *exc) -> None: ...
```

### Internal Architecture

```
                     ┌─────────────────────────────────────────┐
                     │              Batcher[TReq, TRes]         │
                     │                                          │
  submit(req) ──────►│  _queue: deque[_Pending[TReq, TRes]]    │
  submit_many(reqs)─►│                                          │
                     │  _flush_loop (background Task):          │
                     │    sleep(window_ms) OR event.set()       │
                     │    drain queue                           │
                     │    call handler(batch)                   │
                     │    resolve futures                       │
                     │                                          │
                     │  Triggers:                               │
                     │    • len(queue) >= max_batch_size         │
                     │    • window_ms timeout                    │
                     └──────────────────┬───────────────────────┘
                                        │
                                        ▼
                              handler(list[TReq]) → list[TRes]
                              (provided by consumer)
```

### `_Pending` — internal queue entry

```python
@dataclass(slots=True)
class _Pending(Generic[TReq, TRes]):
    request: TReq
    future: asyncio.Future[TRes]
    enqueued_at: float  # time.monotonic()
```

### `BatcherStats` — observable metrics

```python
@dataclass
class BatcherStats:
    total_requests: int = 0
    total_flushes: int = 0
    total_batched: int = 0         # requests in batches > 1
    max_batch_size_seen: int = 0
    total_handler_ms: float = 0.0  # time spent in handler()
    total_queue_wait_ms: float = 0.0

    @property
    def avg_batch_size(self) -> float: ...

    @property
    def avg_queue_wait_ms(self) -> float: ...

    @property
    def batched_pct(self) -> float: ...
```

### Error Handling

```python
# If handler raises, ALL futures in that batch get the exception
try:
    responses = await self._handler(requests)
except Exception as exc:
    for pending in batch:
        pending.future.set_exception(exc)
    return

# handler MUST return exactly len(requests) responses
if len(responses) != len(requests):
    exc = BatchSizeError(f"Handler returned {len(responses)}, expected {len(requests)}")
    for pending in batch:
        pending.future.set_exception(exc)
    return

# Success: resolve each future
for pending, response in zip(batch, responses):
    pending.future.set_result(response)
```

### Exceptions

```python
class BatcherError(Exception): ...
class BatchSizeError(BatcherError): ...    # handler returned wrong count
class BatcherClosedError(BatcherError): ... # submit after shutdown
```

### Advanced: BatcherGroup

For managing multiple batchers with shared lifecycle:

```python
class BatcherGroup:
    """Manages multiple named Batcher instances with shared lifecycle."""

    def register(self, name: str, batcher: Batcher) -> None: ...
    def get(self, name: str) -> Batcher: ...

    async def start_all(self) -> None: ...
    async def shutdown_all(self) -> None: ...

    @property
    def stats(self) -> dict[str, BatcherStats]: ...

    async def __aenter__(self) -> Self: ...
    async def __aexit__(self, *exc) -> None: ...
```

### Package Structure

```
async-batcher/
├── pyproject.toml
├── LICENSE               # MIT
├── README.md
├── src/
│   └── async_batcher/
│       ├── __init__.py   # exports: Batcher, BatcherGroup, BatcherStats
│       ├── _batcher.py   # Batcher implementation
│       ├── _group.py     # BatcherGroup
│       ├── _stats.py     # BatcherStats
│       └── _errors.py    # BatcherError, BatchSizeError, BatcherClosedError
└── tests/
    ├── test_batcher.py
    ├── test_group.py
    ├── test_errors.py
    └── test_concurrency.py
```

### `pyproject.toml`

```toml
[project]
name = "async-batcher"
version = "0.1.0"
description = "Transparent micro-batching for async Python"
requires-python = ">=3.11"
license = "MIT"
dependencies = []  # Zero dependencies

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-asyncio>=0.24", "ruff", "mypy"]
```

### README Example

```python
import asyncio
from async_batcher import Batcher

# Your batch handler — called with accumulated requests
async def call_llm(prompts: list[str]) -> list[str]:
    # This could be Ollama, OpenAI, a database, anything
    return [f"Response to: {p}" for p in prompts]

async def main():
    async with Batcher(handler=call_llm, window_ms=50, max_batch=32) as b:
        # Launch many concurrent tasks — they automatically batch
        tasks = [b.submit(f"Question {i}") for i in range(100)]
        results = await asyncio.gather(*tasks)
        # call_llm was called ~4 times with ~25 items each,
        # NOT 100 individual calls
        print(f"Got {len(results)} results in ~{b.stats.total_flushes} batches")

asyncio.run(main())
```

---

## Part 2: Consortium Integration — `ProviderRegistry`

The consortium uses `async-batcher` by wrapping each provider's
`complete_batch()` as a handler:

### `BatchingProvider` — thin adapter

```python
# src/consortium/providers/batching.py

from async_batcher import Batcher
from consortium.providers.base import LLMProvider, LLMRequest, LLMResponse


class BatchingProvider(LLMProvider):
    """Wraps an LLMProvider with an async-batcher for request coalescing.

    This is a thin adapter: the library does the queue/flush work,
    the inner provider's complete_batch() does the actual LLM call.
    """

    def __init__(
        self,
        inner: LLMProvider,
        *,
        window_ms: float = 100.0,
        max_batch_size: int = 64,
        name: str = "",
    ) -> None:
        self._inner = inner
        self._batcher: Batcher[LLMRequest, LLMResponse] = Batcher(
            handler=inner.complete_batch,  # <-- That's it. Library calls this.
            window_ms=window_ms,
            max_batch_size=max_batch_size,
            name=name,
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        return await self._batcher.submit(request)

    async def complete_batch(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        return await self._batcher.submit_many(requests)

    # Delegate everything else
    def estimate_cost(self, input_tokens, output_tokens, *, batch=False):
        return self._inner.estimate_cost(input_tokens, output_tokens, batch=batch)

    def supports_batch(self):
        return self._inner.supports_batch()

    @property
    def stats(self):
        return self._batcher.stats

    async def start(self):
        await self._batcher.__aenter__()

    async def shutdown(self):
        await self._batcher.__aexit__(None, None, None)
```

Notice how thin this is — 30 lines. The library does all the work.
`handler=inner.complete_batch` is the entire integration point.

### `ProviderRegistry` — uses `BatcherGroup` internally

```python
# src/consortium/providers/registry.py

from async_batcher import BatcherGroup

class ProviderRegistry:
    """One BatchingProvider per (provider, api_model) pair.

    Uses async-batcher's BatcherGroup for lifecycle management.
    """

    def __init__(self) -> None:
        self._providers: dict[str, LLMProvider] = {}
        self._group = BatcherGroup()

    def get(self, model_config: ModelConfig) -> LLMProvider:
        key = f"{model_config.provider}:{model_config.api_model}"
        if key not in self._providers:
            inner = _create_raw_provider(model_config)

            if model_config.batching and model_config.batching.enabled:
                wrapper = BatchingProvider(
                    inner,
                    window_ms=model_config.batching.window_ms,
                    max_batch_size=model_config.batching.max_batch_size,
                    name=key,
                )
                self._group.register(key, wrapper._batcher)
                self._providers[key] = wrapper
            else:
                self._providers[key] = inner

        return self._providers[key]

    async def __aenter__(self):
        await self._group.start_all()
        return self

    async def __aexit__(self, *exc):
        await self._group.shutdown_all()

    @property
    def stats(self):
        return self._group.stats
```

---

## Part 3: `consortium run full-experiment` — Streaming Pipeline

### The Problem with Sequential Stages

Today's workflow:

```
Step 1: consortium run experiment     ← waits for ALL 320 runs
Step 2: consortium run evaluate       ← waits for ALL 320 evals
Step 3: consortium run coherence      ← waits for ALL 320 checks
Step 4: consortium analyze doctor     ← then analyze
```

Total wall-clock = sum of all four stages. No overlap.

### The Streaming Pipeline

```
consortium run full-experiment
```

Runs are pipelined: as each design generation completes, its eval and
coherence start immediately. No run waits for others.

```
Time ────────────────────────────────────────────────────────────────►

Run 1:  [████ generate ████][██ eval ██]
                             [█ coher █]

Run 2:  [████████ generate ████████][██ eval ██]
                                     [█ coher █]

Run 3:  [████ generate ████][██ eval ██]
                             [█ coher █]
                                                    ┌─────────────┐
Run 4:  [████████████ generate ████████████][██ eval ██]           │
                                             [█ coher █]          │
                                                                  │
                              All complete ──────────────────────►  │ analyze
                                                                  │
                                                    └─────────────┘
```

Key properties:
- Generation, eval, and coherence requests **coalesce in shared batchers**
- A run's eval and coherence start the moment its generation finishes
- Eval and coherence for the same run run **in parallel**
- Analysis is a **barrier** — waits for everything
- If generation uses Claude and eval uses Ollama, their batchers are independent

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    FullExperimentRunner                        │
│                                                               │
│  ProviderRegistry (shared across all stages)                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │
│  │ Batcher: │  │ Batcher: │  │ Batcher: │                   │
│  │ anthropic│  │ openai   │  │ ollama   │                   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                   │
│       │              │              │                         │
│  ┌────┴──────────────┴──────────────┴─────────────────────┐  │
│  │                 Task Pool                               │  │
│  │                                                         │  │
│  │  ┌─────────────────────────────────────────────────┐   │  │
│  │  │ Per-run pipeline (asyncio.Task per run):         │   │  │
│  │  │                                                   │   │  │
│  │  │  1. generate(variant, task, rep)                  │   │  │
│  │  │       │                                           │   │  │
│  │  │       ▼                                           │   │  │
│  │  │  2. asyncio.gather(                               │   │  │
│  │  │       evaluate(run_id),                           │   │  │
│  │  │       coherence(run_id)                           │   │  │
│  │  │     )                                             │   │  │
│  │  │       │                                           │   │  │
│  │  │       ▼                                           │   │  │
│  │  │  3. mark_complete(run_id)                         │   │  │
│  │  └─────────────────────────────────────────────────┘   │  │
│  │                                                         │  │
│  │  × 320 concurrent tasks (throttled by semaphore)        │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                               │
│  Barrier: await asyncio.gather(*all_run_tasks)               │
│                                                               │
│  4. analyze()  ← runs only after all tasks complete           │
└──────────────────────────────────────────────────────────────┘
```

### Core Implementation

```python
# src/consortium/pipeline/full_experiment.py

class FullExperimentRunner:
    """Streaming pipeline: generate → eval+coherence → analyze.

    Each run flows through stages independently. All stages share
    the same ProviderRegistry for optimal request batching.
    """

    def __init__(
        self,
        config: FullConfig,
        database: Database,
        prompts_dir: str | Path,
    ) -> None:
        self.config = config
        self.database = database
        self.prompts_dir = Path(prompts_dir)
        self.registry = ProviderRegistry()

    async def run(
        self,
        *,
        variants: list[str] | None = None,
        tasks: list[str] | None = None,
        repetitions: int | None = None,
        force: bool = False,
        skip_analysis: bool = False,
    ) -> FullExperimentStats:
        """Execute the complete experiment pipeline.

        For each (variant, task, rep):
            1. Design generation (orchestrator)
            2. Evaluation + coherence (in parallel)
        Then:
            3. Statistical analysis (after all runs complete)

        Returns:
            Aggregate statistics across all stages.
        """
        variant_ids = variants or list(self.config.experiment.variants)
        task_ids = tasks or list(self.config.experiment.tasks)
        n_reps = repetitions or self.config.experiment.repetitions

        run_matrix = [
            (vid, tid, rep)
            for vid in variant_ids
            for tid in task_ids
            for rep in range(n_reps)
        ]

        # Determine what needs running
        pending = self._get_pending_runs(run_matrix, force=force)

        log = logger.bind(
            total=len(run_matrix),
            pending_gen=len(pending["generate"]),
            pending_eval=len(pending["evaluate"]),
            pending_coherence=len(pending["coherence"]),
        )
        log.info("full_experiment_start")

        stats = FullExperimentStats(total_runs=len(run_matrix))
        semaphore = asyncio.Semaphore(
            self.config.experiment.limits.max_concurrent_runs
        )

        async with self.registry:
            # Create shared pipeline components
            engine = OrchestratorEngine(
                self.config, self.database, self.prompts_dir,
                registry=self.registry,
            )
            eval_pipeline = EvaluationPipeline(
                self.config, self.database, self.prompts_dir,
                registry=self.registry,
            )

            # Launch per-run pipelines concurrently
            run_tasks = []
            for vid, tid, rep in run_matrix:
                task = asyncio.create_task(
                    self._run_single_pipeline(
                        engine, eval_pipeline, semaphore, stats,
                        vid, tid, rep, pending, force,
                    )
                )
                run_tasks.append(task)

            # Barrier: wait for all runs to complete all stages
            await asyncio.gather(*run_tasks, return_exceptions=True)

            # Log batching stats per provider
            for name, batcher_stats in self.registry.stats.items():
                log.info("provider_stats", provider=name, **vars(batcher_stats))

        # Stage 4: Analysis (only after everything is done)
        if not skip_analysis:
            log.info("analysis_start")
            await self._run_analysis(variant_ids, task_ids)
            stats.analysis_complete = True

        log.info("full_experiment_complete", **vars(stats))
        return stats

    async def _run_single_pipeline(
        self,
        engine: OrchestratorEngine,
        eval_pipeline: EvaluationPipeline,
        semaphore: asyncio.Semaphore,
        stats: FullExperimentStats,
        variant_id: str,
        task_id: str,
        rep: int,
        pending: dict[str, set],
        force: bool,
    ) -> None:
        """Execute the full pipeline for one (variant, task, rep).

        Stages:
            1. Generation (if needed)
            2. Eval + Coherence in parallel (if needed)
        """
        run_key = (variant_id, task_id, rep)
        log = logger.bind(variant=variant_id, task=task_id, rep=rep)

        try:
            # Stage 1: Generation
            if run_key in pending["generate"]:
                async with semaphore:
                    log.info("stage_generate_start")
                    design = await engine.run(
                        variant_id, task_id, rep, resume=True, force=force,
                    )
                    stats.generation_completed += 1
                    log.info("stage_generate_done", design_id=design.design_id)

            # Stage 2: Eval + Coherence (parallel, immediately after generation)
            run_id = self._get_run_id(variant_id, task_id, rep)
            if run_id is None:
                log.warning("no_run_id_found_skipping_eval")
                return

            eval_coherence_tasks = []

            if run_key in pending["evaluate"]:
                eval_coherence_tasks.append(
                    self._evaluate_run(eval_pipeline, run_id, log, stats)
                )

            if run_key in pending["coherence"]:
                eval_coherence_tasks.append(
                    self._coherence_run(eval_pipeline, run_id, log, stats)
                )

            if eval_coherence_tasks:
                # These share the batcher with generation requests
                # from OTHER runs that are still in-flight
                await asyncio.gather(*eval_coherence_tasks)

        except Exception as e:
            stats.failed += 1
            log.error("pipeline_failed", error=str(e))

    async def _evaluate_run(self, pipeline, run_id, log, stats):
        """Evaluate a single run's final design."""
        log.info("stage_eval_start")
        await pipeline.evaluate(run_id=run_id)
        stats.evaluation_completed += 1
        log.info("stage_eval_done")

    async def _coherence_run(self, pipeline, run_id, log, stats):
        """Run coherence checks on a single run's final design."""
        log.info("stage_coherence_start")
        await pipeline.run_all_coherence_checks(run_id=run_id)
        stats.coherence_completed += 1
        log.info("stage_coherence_done")

    def _get_pending_runs(self, run_matrix, *, force):
        """Determine which runs need which stages."""
        pending = {
            "generate": set(),
            "evaluate": set(),
            "coherence": set(),
        }

        for vid, tid, rep in run_matrix:
            run_key = (vid, tid, rep)

            # Check generation status
            status = self._get_run_status(vid, tid, rep)
            if force or status != "completed":
                pending["generate"].add(run_key)
                # If generation needed, eval and coherence also needed
                pending["evaluate"].add(run_key)
                pending["coherence"].add(run_key)
                continue

            # Generation done — check eval and coherence
            run_id = self._get_run_id(vid, tid, rep)
            if not self._has_evaluation(run_id) or force:
                pending["evaluate"].add(run_key)
            if not self._has_coherence(run_id) or force:
                pending["coherence"].add(run_key)

        return pending

    async def _run_analysis(self, variant_ids, task_ids):
        """Run statistical analysis after all runs complete."""
        from consortium.analysis.framework import AnalysisFramework
        framework = AnalysisFramework(self.config, self.database)
        framework.run_full_analysis(variant_ids, task_ids)
```

### Stats Dataclass

```python
@dataclass
class FullExperimentStats:
    total_runs: int = 0
    generation_completed: int = 0
    evaluation_completed: int = 0
    coherence_completed: int = 0
    failed: int = 0
    analysis_complete: bool = False

    @property
    def all_stages_complete(self) -> bool:
        return (
            self.generation_completed + self.failed == self.total_runs
            and self.evaluation_completed + self.failed == self.total_runs
            and self.coherence_completed + self.failed == self.total_runs
        )
```

### CLI Command

```python
# src/consortium/cli/run.py

@app.command("full-experiment")
def full_experiment(
    config_dir: Path = typer.Option(_DEFAULT_CONFIGS, "--configs", "-c"),
    database: Path = typer.Option(_DEFAULT_DB, "--database", "-d"),
    prompts_dir: Path = typer.Option(_DEFAULT_PROMPTS, "--prompts", "-p"),
    variants: Optional[str] = typer.Option(None, "--variants"),
    tasks: Optional[str] = typer.Option(None, "--tasks"),
    repetitions: Optional[int] = typer.Option(None, "--reps"),
    force: bool = typer.Option(False, "--force"),
    skip_analysis: bool = typer.Option(False, "--skip-analysis"),
) -> None:
    """Run the full pipeline: generate → evaluate + coherence → analyze.

    Runs stream through stages as they complete — no waiting for all
    generation to finish before starting evaluation. All stages share
    batching providers for maximum throughput.
    """
    from consortium.pipeline.full_experiment import FullExperimentRunner

    config, db = _load_config_and_db(config_dir, database)
    runner = FullExperimentRunner(config, db, prompts_dir)

    variant_list = variants.split(",") if variants else None
    task_list = tasks.split(",") if tasks else None

    try:
        stats = asyncio.run(
            runner.run(
                variants=variant_list,
                tasks=task_list,
                repetitions=repetitions,
                force=force,
                skip_analysis=skip_analysis,
            )
        )

        # Rich output
        table = Table(title="Full Experiment Results")
        table.add_column("Stage", style="cyan")
        table.add_column("Completed", justify="right")
        table.add_column("Status")

        table.add_row("Generation", str(stats.generation_completed),
                      "✓" if stats.generation_completed == stats.total_runs else "…")
        table.add_row("Evaluation", str(stats.evaluation_completed),
                      "✓" if stats.evaluation_completed == stats.total_runs else "…")
        table.add_row("Coherence", str(stats.coherence_completed),
                      "✓" if stats.coherence_completed == stats.total_runs else "…")
        table.add_row("Analysis", "—",
                      "✓" if stats.analysis_complete else "skipped")

        if stats.failed > 0:
            table.add_row("Failed", f"[red]{stats.failed}[/red]", "⚠")

        console.print(table)

    except Exception as e:
        console.print(f"[red]Pipeline failed: {e}[/red]")
        raise typer.Exit(code=1)
    finally:
        db.close()
```

---

## Part 4: How It All Connects

### The Flow

```
User runs: consortium run full-experiment --variants v1,v2 --tasks t1 --reps 5

1. FullExperimentRunner creates ProviderRegistry
2. Registry creates BatchingProviders (one per provider:model):
     - BatchingProvider("anthropic:claude-sonnet-4-5")  ← wraps AnthropicProvider
     - BatchingProvider("openai:gpt-4.1")               ← wraps OpenAIProvider
     - BatchingProvider("ollama:gpt-oss-120b")           ← wraps OllamaProvider
   Each uses async-batcher's Batcher internally

3. 10 concurrent run-pipeline tasks launch:
     v1-t1-rep0, v1-t1-rep1, ..., v2-t1-rep4

4. As runs execute:
     - v1-t1-rep0's designer calls provider.complete()
       → BatchingProvider.complete() → batcher.submit()
       → queued in Anthropic batcher

     - v2-t1-rep0's reviewers call provider.complete()
       → queued in OpenAI batcher

     - Anthropic batcher flushes every 50ms:
       sends 5 requests → AnthropicProvider.complete_batch()

     - v1-t1-rep0 generation finishes
       → IMMEDIATELY starts eval + coherence for v1-t1-rep0
       → eval calls provider.complete_batch()
       → queued in Ollama batcher (alongside still-running generation evals)

5. Ollama batcher flushes every 150ms:
     - Batch includes eval from v1-t1-rep0 + coherence from v1-t1-rep0
       + eval from v1-t1-rep1 (which also just finished)
     - GPU processes all in one forward pass

6. All 10 runs complete all stages

7. Analysis runs (barrier — all must be done)

8. Stats printed with per-provider batching metrics
```

### Dependency Graph

```
async-batcher (library, zero deps)
    │
    │  pip install async-batcher
    │
    ▼
consortium (application)
    ├── providers/batching.py     ← uses async_batcher.Batcher
    ├── providers/registry.py     ← uses async_batcher.BatcherGroup
    └── pipeline/full_experiment.py  ← uses registry across stages
```

---

## Part 5: Implementation Plan

### Phase 1: `async-batcher` library (standalone repo)

| Step | Files | Est. |
|---|---|---|
| 1. Scaffold repo + pyproject.toml | repo setup | 30m |
| 2. `Batcher` core (queue, flush loop, futures) | `_batcher.py` | 2h |
| 3. `BatcherStats` | `_stats.py` | 30m |
| 4. `BatcherGroup` | `_group.py` | 1h |
| 5. Error types | `_errors.py` | 15m |
| 6. Tests (coalescing, timing, errors, concurrency) | `tests/` | 2h |
| 7. README + docs | `README.md` | 1h |
| 8. Publish to PyPI | CI/CD | 30m |

### Phase 2: Consortium integration

| Step | Files | Est. |
|---|---|---|
| 1. `BatchingConfig` in config/models.py | config | 30m |
| 2. `BatchingProvider` adapter | providers/batching.py | 30m |
| 3. `ProviderRegistry` | providers/registry.py | 1h |
| 4. Wire into orchestrator factory + engine | 2 files | 1h |
| 5. Wire into evaluation pipeline | 1 file | 30m |
| 6. Model YAML configs | configs/ | 15m |
| 7. Integration tests | tests/ | 1h |

### Phase 3: `full-experiment` pipeline

| Step | Files | Est. |
|---|---|---|
| 1. `FullExperimentRunner` | pipeline/full_experiment.py | 2h |
| 2. CLI command | cli/run.py | 30m |
| 3. `_get_pending_runs` logic | full_experiment.py | 1h |
| 4. Rich progress display | cli/run.py | 1h |
| 5. End-to-end test | tests/ | 1h |

---

## Open Questions

1. **Library name**: `async-batcher` vs `microbatch` vs `batch-queue`?
   - `async-batcher` is descriptive and available on PyPI
   - `microbatch` is catchier but might conflict

2. **Monorepo or separate repo for the library?**
   - Separate repo = clean OSS, independent versioning
   - Monorepo with workspace = easier development iteration
   - Recommendation: **separate repo**, pin version in consortium

3. **Should the library support partial failures?**
   - Current: handler raises → all futures rejected
   - Enhanced: handler returns `list[TRes | Exception]`
   - Recommendation: start simple, add `PartialBatchResult` later

4. **Backpressure in the pipeline?**
   - If generation is fast and eval is slow, runs pile up waiting for eval
   - Could add a pipeline-level semaphore per stage
   - Recommendation: semaphore on generation is sufficient for v1 —
     eval/coherence are I/O-bound and the batcher naturally throttles

5. **Progress display?**
   - Rich live display with per-stage progress bars would be ideal
   - `[Gen: 8/10] [Eval: 5/10] [Coher: 4/10] [Analyze: waiting]`
   - Add in Phase 3, step 4
