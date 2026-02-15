"""FullExperimentRunner — streaming pipeline: generate → eval+coherence → analyze."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import structlog

from consortium.config.models import FullConfig
from consortium.evaluation.pipeline import EvaluationPipeline
from consortium.orchestrator.engine import OrchestratorEngine
from consortium.providers.registry import ProviderRegistry
from consortium.storage.database import Database

logger = structlog.get_logger()


# ── Stats ────────────────────────────────────────────────────────────────────


@dataclass
class _PipelineResult:
    """Per-task result returned by each pipeline coroutine."""

    generation_completed: int = 0
    evaluation_completed: int = 0
    coherence_completed: int = 0
    failed: int = 0


@dataclass
class FullExperimentStats:
    """Aggregate statistics across all stages of the full pipeline."""

    total_runs: int = 0
    generation_completed: int = 0
    evaluation_completed: int = 0
    coherence_completed: int = 0
    failed: int = 0
    analysis_complete: bool = False

    @property
    def all_stages_complete(self) -> bool:
        """True if every run has finished all stages (or failed)."""
        return (
            self.generation_completed + self.failed == self.total_runs
            and self.evaluation_completed + self.failed == self.total_runs
            and self.coherence_completed + self.failed == self.total_runs
        )


# ── Runner ───────────────────────────────────────────────────────────────────


class FullExperimentRunner:
    """Streaming pipeline: generate → eval+coherence → analyze.

    Each (variant, task, rep) flows through stages independently.  As soon
    as one run's generation finishes, its evaluation and coherence checks
    start immediately — without waiting for other runs.

    All stages share the same :class:`ProviderRegistry` for optimal request
    batching across stages.

    Architecture::

        ┌───────────────────────────────────────────────────┐
        │           FullExperimentRunner                     │
        │  ProviderRegistry (shared batchers)                │
        │                                                    │
        │  Per-run pipeline (× N concurrent tasks):          │
        │   1. generate(variant, task, rep)                  │
        │   2. asyncio.gather(eval, coherence)               │
        │                                                    │
        │  Barrier: all runs complete                        │
        │  3. analyze()                                      │
        └───────────────────────────────────────────────────┘
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

    # ── Public API ──────────────────────────────────────────────────────

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
        # Resolve defaults from config
        variant_ids = variants or [
            v.split("_")[0] if "_" in v else v for v in self.config.experiment.variants
        ]
        task_ids = tasks or [
            t.split("_")[0] if "_" in t else t for t in self.config.experiment.tasks
        ]
        n_reps = repetitions or self.config.experiment.repetitions

        run_matrix = [
            (vid, tid, rep) for vid in variant_ids for tid in task_ids for rep in range(n_reps)
        ]

        # Randomize run order using experiment seed for reproducibility
        import random

        rng = random.Random(self.config.experiment.seed)
        rng.shuffle(run_matrix)

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
        semaphore = asyncio.Semaphore(self.config.experiment.limits.max_concurrent_runs)

        # ── Rich Live dashboard ──────────────────────────────────────
        from rich.console import Console, Group
        from rich.live import Live
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
        )
        from rich.table import Table

        console = Console()

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.fields[stage]}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TextColumn("•"),
            TextColumn("[green]\u2713{task.fields[ok]}[/green]"),
            TextColumn("[red]\u2717{task.fields[fail]}[/red]"),
            TextColumn("•"),
            TimeElapsedColumn(),
        )

        n_total = len(run_matrix)
        gen_task = progress.add_task("gen", total=n_total, stage="Generation ", ok=0, fail=0)
        eval_task = progress.add_task("eval", total=n_total, stage="Evaluation ", ok=0, fail=0)
        coh_task = progress.add_task("coh", total=n_total, stage="Coherence  ", ok=0, fail=0)

        # Thread-safe counters for progress (updated from coroutines)
        progress_counters = {
            "gen_ok": 0,
            "gen_fail": 0,
            "eval_ok": 0,
            "eval_fail": 0,
            "coh_ok": 0,
            "coh_fail": 0,
        }

        def _build_batcher_table() -> Table:
            """Build a live batcher status table from registry state."""
            import time as _time

            tbl = Table(title="Batchers (live)", expand=False)
            tbl.add_column("Provider", style="cyan")
            tbl.add_column("Limit", justify="center")
            tbl.add_column("Queue", justify="right")
            tbl.add_column("Next ⏱", justify="right")
            tbl.add_column("💰 Cost", justify="right")
            tbl.add_column("✈ Batches", justify="right")
            tbl.add_column("✈ Req/Batch", justify="right")
            tbl.add_column("✓ Batches", justify="right")
            tbl.add_column("✓ Requests", justify="right")
            tbl.add_column("✓ Req/Batch", justify="right")
            tbl.add_column("✓ Batched%", justify="right")
            tbl.add_column("✓ Min ⏱", justify="right")
            tbl.add_column("✓ Avg ⏱", justify="right")
            tbl.add_column("✓ Max ⏱", justify="right")

            def _fmt_ms(ms: float) -> str:
                """Format milliseconds into human-readable duration."""
                if ms == float("inf") or ms < 0:
                    return "[dim]—[/dim]"
                if ms >= 1000:
                    return f"{ms / 1000:.1f}s"
                return f"{ms:.0f}ms"

            def _fmt_cost(usd: float) -> str:
                """Format cost with appropriate precision."""
                if usd <= 0:
                    return "[dim]$0.00[/dim]"
                if usd < 0.01:
                    return f"[green]${usd:.4f}[/green]"
                if usd < 1.0:
                    return f"[yellow]${usd:.3f}[/yellow]"
                return f"[bold yellow]${usd:.2f}[/bold yellow]"

            live_state = self.registry.batcher_live_state()
            if not live_state:
                tbl.add_row("[dim]no batching-enabled providers[/dim]")
                return tbl

            now = _time.monotonic()
            total_cost = 0.0
            for name, state in sorted(live_state.items()):
                q = state["queue_depth"]
                mx = state["max_batch_size"]
                window = state["window_ms"]
                last_t = state["last_flush_time"]
                in_flight = state["in_flight_flushes"]
                in_flight_reqs = state["in_flight_requests"]
                st = state["stats"]
                cost = state["total_cost"]
                total_cost += cost

                # Time until next flush — always countdown
                if last_t > 0:
                    elapsed_ms = (now - last_t) * 1000.0
                    remaining_ms = window - (elapsed_ms % window)
                    next_str = _fmt_ms(remaining_ms)
                else:
                    next_str = _fmt_ms(window)

                if q > 0:
                    queue_style = "red" if q >= mx else ("yellow" if q > mx // 2 else "green")
                    queue_str = f"[bold {queue_style}]⏳ {q}[/bold {queue_style}]/{mx}"
                else:
                    queue_str = f"[dim]0/{mx}[/dim]"

                # In-flight stats
                if in_flight > 0:
                    flight_batches = f"[bold yellow]{in_flight}[/bold yellow]"
                    flight_rpb = f"[bold yellow]{in_flight_reqs / in_flight:.1f}[/bold yellow]"
                else:
                    flight_batches = "[dim]—[/dim]"
                    flight_rpb = "[dim]—[/dim]"

                tbl.add_row(
                    name,
                    f"{window / 1000:.0f}s / {mx}",
                    queue_str,
                    next_str,
                    _fmt_cost(cost),
                    flight_batches,
                    flight_rpb,
                    str(st.total_flushes),
                    str(st.total_requests),
                    f"{st.avg_batch_size:.1f}",
                    f"{st.batched_pct:.0f}%",
                    _fmt_ms(st.min_handler_ms),
                    _fmt_ms(st.avg_handler_ms),
                    _fmt_ms(st.max_handler_ms),
                )

            # Total row
            tbl.add_row(
                "[bold]TOTAL[/bold]",
                "",
                "",
                "",
                f"[bold]{_fmt_cost(total_cost)}[/bold]",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                end_section=True,
            )
            return tbl

        def _build_dashboard() -> Group:
            """Compose the full dashboard renderable."""
            return Group(progress, _build_batcher_table())

        # Run pipeline with registry
        async with self.registry:
            engine = OrchestratorEngine(
                self.config,
                self.database,
                self.prompts_dir,
                registry=self.registry,
            )
            eval_pipeline = EvaluationPipeline(
                self.config,
                self.database,
                self.prompts_dir,
                registry=self.registry,
            )

            # Launch per-run pipelines concurrently — each returns its own
            # _PipelineResult to avoid shared mutable state.
            run_tasks = [
                asyncio.create_task(
                    self._run_single_pipeline(
                        engine,
                        eval_pipeline,
                        semaphore,
                        vid,
                        tid,
                        rep,
                        pending,
                        force,
                        progress,
                        progress_counters,
                        gen_task,
                        eval_task,
                        coh_task,
                    ),
                    name=f"pipeline-{vid}-{tid}-rep{rep}",
                )
                for vid, tid, rep in run_matrix
            ]

            with Live(
                _build_dashboard(),
                console=console,
                refresh_per_second=2,
                transient=False,
                get_renderable=_build_dashboard,
            ) as live:  # noqa: F841
                # Rich Live auto-refreshes at 2Hz using get_renderable
                # — no manual refresh task needed.

                # Barrier: wait for all runs to complete all stages
                results = await asyncio.gather(*run_tasks, return_exceptions=True)

            # Aggregate per-task results into final stats
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    vid, tid, rep = run_matrix[i]
                    log.error(
                        "pipeline_task_exception",
                        variant=vid,
                        task=tid,
                        rep=rep,
                        error=str(result),
                    )
                    stats.failed += 1
                elif isinstance(result, _PipelineResult):
                    stats.generation_completed += result.generation_completed
                    stats.evaluation_completed += result.evaluation_completed
                    stats.coherence_completed += result.coherence_completed
                    stats.failed += result.failed

            # Print final batcher summary table
            final_table = Table(title="Batcher Summary")
            final_table.add_column("Provider", style="cyan")
            final_table.add_column("Requests", justify="right")
            final_table.add_column("Flushes", justify="right")
            final_table.add_column("Avg Batch", justify="right")
            final_table.add_column("Max Batch", justify="right")
            final_table.add_column("Batched%", justify="right")
            final_table.add_column("Handler ms", justify="right")
            final_table.add_column("Avg Queue ms", justify="right")

            for name, batcher_stats in self.registry.stats.items():
                log.info(
                    "provider_stats",
                    provider=name,
                    total_requests=batcher_stats.total_requests,
                    total_flushes=batcher_stats.total_flushes,
                    avg_batch_size=round(batcher_stats.avg_batch_size, 2),
                    batched_pct=round(batcher_stats.batched_pct, 1),
                    total_handler_ms=round(batcher_stats.total_handler_ms, 1),
                )
                final_table.add_row(
                    name,
                    str(batcher_stats.total_requests),
                    str(batcher_stats.total_flushes),
                    f"{batcher_stats.avg_batch_size:.1f}",
                    str(batcher_stats.max_batch_size_seen),
                    f"{batcher_stats.batched_pct:.0f}%",
                    f"{batcher_stats.total_handler_ms:.0f}",
                    f"{batcher_stats.avg_queue_wait_ms:.1f}",
                )

            if self.registry.stats:
                console.print()
                console.print(final_table)

        # Analysis barrier (only after everything is done)
        if not skip_analysis:
            log.info("analysis_start")
            self._run_analysis(variant_ids, task_ids)
            stats.analysis_complete = True

        log.info(
            "full_experiment_complete",
            generation=stats.generation_completed,
            evaluation=stats.evaluation_completed,
            coherence=stats.coherence_completed,
            failed=stats.failed,
            analysis=stats.analysis_complete,
        )
        return stats

    # ── Per-run pipeline ────────────────────────────────────────────────

    async def _run_single_pipeline(
        self,
        engine: OrchestratorEngine,
        eval_pipeline: EvaluationPipeline,
        semaphore: asyncio.Semaphore,
        variant_id: str,
        task_id: str,
        rep: int,
        pending: dict[str, set],
        force: bool,
        progress: object | None = None,
        progress_counters: dict | None = None,
        gen_task_id: object | None = None,
        eval_task_id: object | None = None,
        coh_task_id: object | None = None,
    ) -> _PipelineResult:
        """Execute all stages for one (variant, task, rep).

        Returns a :class:`_PipelineResult` with per-task counters so the
        caller can aggregate without shared mutable state.
        """
        run_key = (variant_id, task_id, rep)
        log = logger.bind(variant=variant_id, task=task_id, rep=rep)
        result = _PipelineResult()

        def _advance(stage: str, ok: bool = True) -> None:
            """Advance the relevant progress bar."""
            if progress is None or progress_counters is None:
                return
            if stage == "gen":
                key_ok, key_fail, tid = "gen_ok", "gen_fail", gen_task_id
            elif stage == "eval":
                key_ok, key_fail, tid = "eval_ok", "eval_fail", eval_task_id
            else:
                key_ok, key_fail, tid = "coh_ok", "coh_fail", coh_task_id

            if ok:
                progress_counters[key_ok] += 1
            else:
                progress_counters[key_fail] += 1

            progress.update(  # type: ignore[union-attr]
                tid,
                advance=1,
                ok=progress_counters[key_ok],
                fail=progress_counters[key_fail],
            )

        try:
            # Stage 1: Generation (gated by semaphore)
            if run_key in pending["generate"]:
                async with semaphore:
                    log.info("stage_generate_start")
                    design = await engine.run(
                        variant_id,
                        task_id,
                        rep,
                        resume=True,
                        force=force,
                    )
                    result.generation_completed += 1
                    _advance("gen", ok=True)
                    log.info("stage_generate_done", design_id=design.design_id)
            else:
                # Generation already complete — count it
                result.generation_completed += 1
                _advance("gen", ok=True)

            # Stage 2: Get run_id for downstream stages
            run_id = self._get_run_id(variant_id, task_id, rep)
            if run_id is None:
                log.warning("no_run_id_found_skipping_eval")
                return result

            # Stage 3: Eval + Coherence in parallel
            pending_eval = run_key in pending["evaluate"]
            pending_coh = run_key in pending["coherence"]
            eval_coherence_tasks: list[tuple[str, asyncio.Task]] = []

            if pending_eval:
                eval_coherence_tasks.append(
                    ("eval", asyncio.create_task(self._evaluate_run(eval_pipeline, run_id, log)))
                )
            else:
                result.evaluation_completed += 1
                _advance("eval", ok=True)

            if pending_coh:
                eval_coherence_tasks.append(
                    ("coh", asyncio.create_task(self._coherence_run(eval_pipeline, run_id, log)))
                )
            else:
                result.coherence_completed += 1
                _advance("coh", ok=True)

            if eval_coherence_tasks:
                outcomes = await asyncio.gather(
                    *(t for _, t in eval_coherence_tasks),
                    return_exceptions=True,
                )
                for (stage, _task), outcome in zip(eval_coherence_tasks, outcomes, strict=True):
                    if isinstance(outcome, Exception):
                        result.failed += 1
                        _advance(stage, ok=False)
                        log.error(f"stage_{stage}_failed", error=str(outcome))
                    else:
                        if stage == "eval":
                            result.evaluation_completed += 1
                        else:
                            result.coherence_completed += 1
                        _advance(stage, ok=True)

        except Exception as e:
            result.failed += 1
            _advance("gen", ok=False)
            log.error("stage_generate_failed", error=str(e), exc_info=True)

        return result

    async def _evaluate_run(
        self,
        pipeline: EvaluationPipeline,
        run_id: str,
        log: object,
    ) -> None:
        """Evaluate a single run's final design."""
        log.info("stage_eval_start")  # type: ignore[union-attr]
        await pipeline.evaluate(run_id=run_id, quiet=True)
        log.info("stage_eval_done")  # type: ignore[union-attr]

    async def _coherence_run(
        self,
        pipeline: EvaluationPipeline,
        run_id: str,
        log: object,
    ) -> None:
        """Run coherence checks on a single run's final design."""
        log.info("stage_coherence_start")  # type: ignore[union-attr]
        await pipeline.run_all_coherence_checks(run_id=run_id, quiet=True)
        log.info("stage_coherence_done")  # type: ignore[union-attr]

    # ── Pending-runs resolution ─────────────────────────────────────────

    def _get_pending_runs(
        self,
        run_matrix: list[tuple[str, str, int]],
        *,
        force: bool,
    ) -> dict[str, set[tuple[str, str, int]]]:
        """Determine which runs need which stages."""
        pending: dict[str, set[tuple[str, str, int]]] = {
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
                # If generation needed, downstream stages also needed
                pending["evaluate"].add(run_key)
                pending["coherence"].add(run_key)
                continue

            # Generation done — check eval and coherence
            run_id = self._get_run_id(vid, tid, rep)
            if run_id is None:
                # Shouldn't happen for completed runs, but be safe
                pending["generate"].add(run_key)
                pending["evaluate"].add(run_key)
                pending["coherence"].add(run_key)
                continue

            if not self._has_evaluation(run_id) or force:
                pending["evaluate"].add(run_key)
            if not self._has_coherence(run_id) or force:
                pending["coherence"].add(run_key)

        return pending

    # ── Database helpers ────────────────────────────────────────────────

    def _get_run_status(self, variant_id: str, task_id: str, rep: int) -> str | None:
        """Get the status of a run from the database."""
        row = self.database.conn.execute(
            "SELECT status FROM runs WHERE variant_id = ? AND task_id = ? AND repetition = ?",
            (variant_id, task_id, rep),
        ).fetchone()
        return row["status"] if row else None

    def _get_run_id(self, variant_id: str, task_id: str, rep: int) -> str | None:
        """Get the run_id for a (variant, task, rep) triple."""
        row = self.database.conn.execute(
            "SELECT run_id FROM runs WHERE variant_id = ? AND task_id = ? AND repetition = ?",
            (variant_id, task_id, rep),
        ).fetchone()
        return row["run_id"] if row else None

    def _has_evaluation(self, run_id: str) -> bool:
        """Check if a run's final design has been evaluated (has median scores)."""
        row = self.database.conn.execute(
            "SELECT 1 FROM scores_median WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        return row is not None

    def _has_coherence(self, run_id: str) -> bool:
        """Check if a run's final design has coherence checks.

        The coherence_checks table references design_id, not run_id directly,
        so we join through the designs table.
        """
        row = self.database.conn.execute(
            """SELECT 1 FROM coherence_checks cc
               JOIN designs d ON cc.design_id = d.design_id
               WHERE d.run_id = ? AND d.is_final = TRUE
               LIMIT 1""",
            (run_id,),
        ).fetchone()
        return row is not None

    # ── Analysis ────────────────────────────────────────────────────────

    def _run_analysis(self, variant_ids: list[str], task_ids: list[str]) -> None:
        """Run statistical analysis after all runs complete."""
        try:
            from consortium.analysis.framework import generate_decision_framework
            from consortium.analysis.loader import (
                load_coherence_dataframe,
                load_costs_dataframe,
                load_scores_dataframe,
            )

            scores_df = load_scores_dataframe(self.database)
            costs_df = load_costs_dataframe(self.database)

            if scores_df.empty:
                logger.warning("analysis_skipped_no_scores")
                return

            coherence_df = load_coherence_dataframe(self.database)
            if coherence_df.empty:
                coherence_df = None

            output_dir = Path(self.config.experiment.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            generate_decision_framework(
                scores_df,
                costs_df,
                output_dir,
                coherence_df=coherence_df,
            )
            logger.info("analysis_complete", output_dir=str(output_dir))

        except ImportError:
            logger.warning("analysis_skipped_missing_dependencies")
        except Exception as e:
            logger.error("analysis_failed", error=str(e), exc_info=True)
