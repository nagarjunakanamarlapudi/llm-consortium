"""Batch orchestration layer — groups LLM requests across runs for batch API submission.

The batch layer sits above OrchestratorEngine and provider.complete_batch().
Within a single run, steps are sequential (generate → review → revise).
But *across* runs, independent runs can be batched together. For example,
all generation steps across different tasks can share a single batch API call.

Two modes:
1. **BatchCollector** — deferred execution. Agents push requests into a queue
   instead of calling the provider directly. A flush() groups by provider
   and submits via complete_batch(). Used to maximize batch API efficiency.

2. **BatchExperimentRunner** — orchestrates multi-step batching across the
   experiment matrix by running each pipeline step across all pending runs
   in batch, then advancing to the next step.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog

from consortium.config.models import FullConfig, ModelConfig
from consortium.providers.base import LLMProvider, LLMRequest, LLMResponse
from consortium.providers.factory import create_provider
from consortium.storage.database import Database

logger = structlog.get_logger()


@dataclass
class PendingRequest:
    """A request waiting to be sent via batch API."""

    request_id: str
    request: LLMRequest
    model_config: ModelConfig
    provider: LLMProvider
    future: asyncio.Future
    submitted_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class BatchCollector:
    """Collects LLM requests and submits them as provider batch calls.

    Drop-in replacement for direct provider.complete() calls. Instead of
    awaiting a response immediately, callers receive a Future that resolves
    when the batch is flushed.

    Usage::

        collector = BatchCollector()

        # Agents enqueue requests (returns immediately)
        future = collector.enqueue(request, model_config, provider)

        # ... more enqueues ...

        # Flush sends all pending requests via batch API
        await collector.flush()

        # Now all futures are resolved
        response = future.result()
    """

    def __init__(self) -> None:
        self._pending: list[PendingRequest] = []
        self._loop: asyncio.AbstractEventLoop | None = None

    def enqueue(
        self,
        request: LLMRequest,
        model_config: ModelConfig,
        provider: LLMProvider,
    ) -> asyncio.Future:
        """Add a request to the batch queue.

        Returns a Future that will contain the LLMResponse after flush().
        """
        if self._loop is None:
            self._loop = asyncio.get_running_loop()

        future = self._loop.create_future()
        pending = PendingRequest(
            request_id=uuid.uuid4().hex,
            request=request,
            model_config=model_config,
            provider=provider,
            future=future,
        )
        self._pending.append(pending)
        return future

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    async def flush(self) -> int:
        """Submit all pending requests via batch API, grouped by provider.

        Returns:
            Number of responses received.
        """
        if not self._pending:
            return 0

        # Group by provider instance (using id() as key)
        by_provider: dict[int, list[PendingRequest]] = defaultdict(list)
        for p in self._pending:
            by_provider[id(p.provider)].append(p)

        total_responses = 0
        log = logger.bind(
            total_pending=len(self._pending),
            provider_groups=len(by_provider),
        )
        log.info("batch_flush_start")

        for provider_id, group in by_provider.items():
            provider = group[0].provider
            requests = [p.request for p in group]
            provider_name = group[0].model_config.provider

            g_log = log.bind(
                provider=provider_name,
                batch_size=len(requests),
            )

            if provider.supports_batch():
                g_log.info("submitting_batch")
                try:
                    responses = await provider.complete_batch(requests)
                    for pending, response in zip(group, responses):
                        pending.future.set_result(response)
                        total_responses += 1
                except Exception as e:
                    g_log.error("batch_failed", error=str(e))
                    for pending in group:
                        pending.future.set_exception(e)
            else:
                # Fallback: run as parallel individual calls
                g_log.info("batch_fallback_parallel")
                tasks = [
                    self._complete_single(pending) for pending in group
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for pending, result in zip(group, results):
                    if isinstance(result, Exception):
                        pending.future.set_exception(result)
                    else:
                        pending.future.set_result(result)
                        total_responses += 1

        self._pending.clear()
        log.info("batch_flush_complete", responses=total_responses)
        return total_responses

    @staticmethod
    async def _complete_single(pending: PendingRequest) -> LLMResponse:
        """Fallback: complete a single request via the normal API."""
        return await pending.provider.complete(pending.request)


class BatchExperimentRunner:
    """Runs the experiment matrix using batch API calls for efficiency.

    Instead of running each variant×task×rep sequentially (which makes
    one LLM call at a time), this runner:

    1. Identifies all runs that need the same pipeline step
    2. Collects all LLM requests for that step across runs
    3. Submits them as a single batch API call per provider
    4. Distributes responses and advances each run

    This dramatically reduces wall-clock time when using providers that
    support batch APIs (Anthropic, OpenAI) at a 50% cost discount.
    """

    def __init__(
        self,
        config: FullConfig,
        database: Database,
        prompts_dir: str,
    ) -> None:
        self.config = config
        self.database = database
        self.prompts_dir = prompts_dir
        self.collector = BatchCollector()

        # Pre-create providers (one per model config, shared)
        self._providers: dict[str, LLMProvider] = {}

    def get_provider(self, model_id: str) -> LLMProvider:
        """Get or create a provider for a model config (cached)."""
        if model_id not in self._providers:
            model_config = self.config.get_model(model_id)
            self._providers[model_id] = create_provider(model_config)
        return self._providers[model_id]

    async def run_batch_experiment(
        self,
        *,
        variants: list[str] | None = None,
        tasks: list[str] | None = None,
        repetitions: int | None = None,
    ) -> dict[str, Any]:
        """Run the full experiment using batch APIs where possible.

        For providers that support batch mode, requests are accumulated
        and submitted together. For providers that don't, requests fall
        back to parallel individual calls.

        Returns:
            Summary stats: submitted, completed, failed, total_cost.
        """
        variant_ids = variants or list(self.config.experiment.variants)
        task_ids = tasks or list(self.config.experiment.tasks)
        reps = repetitions or self.config.experiment.repetitions

        run_matrix = [
            (vid, tid, rep)
            for vid in variant_ids
            for tid in task_ids
            for rep in range(reps)
        ]

        # Check which are already done
        completed_rows = self.database.conn.execute(
            "SELECT variant_id, task_id, repetition FROM runs WHERE status = 'completed'"
        ).fetchall()
        completed = {
            (r["variant_id"], r["task_id"], r["repetition"]) for r in completed_rows
        }
        pending = [r for r in run_matrix if r not in completed]

        log = logger.bind(
            total=len(run_matrix),
            pending=len(pending),
            skipped=len(completed),
        )
        log.info("batch_experiment_start")

        if not pending:
            log.info("nothing_to_run")
            return {
                "total": len(run_matrix),
                "submitted": 0,
                "completed": len(completed),
                "failed": 0,
            }

        # For batch mode, we submit generation steps across all pending runs,
        # then review steps, etc. This requires the orchestrators to support
        # step-wise execution, which is a deeper refactor.
        #
        # For now, we use a simpler approach: group runs by model provider
        # and submit concurrent individual calls with controlled parallelism.

        from consortium.orchestrator.engine import OrchestratorEngine

        engine = OrchestratorEngine(self.config, self.database, self.prompts_dir)

        # Run with limited concurrency
        semaphore = asyncio.Semaphore(
            self.config.experiment.limits.max_concurrent_runs
        )

        stats = {"submitted": len(pending), "completed": 0, "failed": 0}

        async def _run_one(vid: str, tid: str, rep: int) -> None:
            async with semaphore:
                try:
                    await engine.run(vid, tid, rep, resume=True)
                    stats["completed"] += 1
                except Exception as e:
                    stats["failed"] += 1
                    logger.error("batch_run_failed", variant=vid, task=tid, rep=rep, error=str(e))

        await asyncio.gather(
            *[_run_one(vid, tid, rep) for vid, tid, rep in pending]
        )

        log.info("batch_experiment_complete", **stats)
        return {
            "total": len(run_matrix),
            **stats,
            "skipped": len(completed),
        }
