"""CLI command: run experiments."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()

_DEFAULT_DB = Path("data/consortium.db")
_DEFAULT_CONFIGS = Path("configs")
_DEFAULT_PROMPTS = Path("prompts")


def _load_config_and_db(
    config_dir: Path,
    database: Path,
) -> tuple:
    """Load FullConfig and Database, initialising DB schema if needed."""
    from consortium.config.loader import load_full_config
    from consortium.storage.database import Database

    config = load_full_config(config_dir)
    db = Database(database)
    db.init_schema()
    return config, db


@app.command()
def single(
    variant: str = typer.Argument(..., help="Variant ID (v1-v8)"),
    task: str = typer.Argument(..., help="Task ID (t1-t8)"),
    rep: int = typer.Option(0, "--rep", "-r", help="Repetition index (0-based)"),
    config_dir: Path = typer.Option(_DEFAULT_CONFIGS, "--configs", "-c"),
    database: Path = typer.Option(_DEFAULT_DB, "--database", "-d"),
    prompts_dir: Path = typer.Option(_DEFAULT_PROMPTS, "--prompts", "-p"),
    resume: bool = typer.Option(False, "--resume", help="Resume a failed/incomplete run"),
    force: bool = typer.Option(False, "--force", help="Overwrite a completed run"),
) -> None:
    """Run a single variant × task × repetition."""
    from consortium.orchestrator.engine import OrchestratorEngine
    from consortium.providers.registry import ProviderRegistry

    config, db = _load_config_and_db(config_dir, database)

    async def _run() -> None:
        registry = ProviderRegistry()
        async with registry:
            engine = OrchestratorEngine(config, db, prompts_dir, registry=registry)
            return await engine.run(variant, task, rep, resume=resume, force=force)

    try:
        design = asyncio.run(_run())
        console.print(f"[green]✓ Run completed: {design.design_id}[/green]")
        console.print(f"  Design length: {len(design.full_text):,} chars")
    except RuntimeError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)
    except KeyError as e:
        console.print(f"[red]Config error: {e}[/red]")
        raise typer.Exit(code=1)
    finally:
        db.close()


@app.command()
def experiment(
    config_dir: Path = typer.Option(_DEFAULT_CONFIGS, "--configs", "-c"),
    database: Path = typer.Option(_DEFAULT_DB, "--database", "-d"),
    prompts_dir: Path = typer.Option(_DEFAULT_PROMPTS, "--prompts", "-p"),
    variants: Optional[str] = typer.Option(
        None, "--variants", help="Comma-separated variant IDs (default: all from config)"
    ),
    tasks: Optional[str] = typer.Option(
        None, "--tasks", help="Comma-separated task IDs (default: all from config)"
    ),
    repetitions: Optional[int] = typer.Option(
        None, "--reps", help="Number of repetitions (default: from config)"
    ),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Skip completed runs"),
    force: bool = typer.Option(False, "--force", help="Re-run even completed runs"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report what would run without executing"
    ),
) -> None:
    """Run the full experiment matrix (variants × tasks × repetitions)."""
    from consortium.orchestrator.engine import ExperimentRunner
    from consortium.providers.registry import ProviderRegistry

    config, db = _load_config_and_db(config_dir, database)

    variant_list = variants.split(",") if variants else None
    task_list = tasks.split(",") if tasks else None

    async def _run() -> dict:
        registry = ProviderRegistry()
        async with registry:
            runner = ExperimentRunner(config, db, prompts_dir, registry=registry)
            return await runner.run_experiment(
                variants=variant_list,
                tasks=task_list,
                repetitions=repetitions,
                resume=resume,
                force=force,
                dry_run=dry_run,
            )

    try:
        stats = asyncio.run(_run())

        # Display results
        table = Table(title="Experiment Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right")

        for key, value in stats.items():
            style = ""
            if key == "failed" and value > 0:
                style = "red"
            elif key == "completed":
                style = "green"
            table.add_row(key, str(value), style=style)

        console.print(table)

        if stats.get("failed", 0) > 0:
            console.print(
                f"[yellow]⚠ {stats['failed']} run(s) failed. "
                "Check logs and re-run with --resume.[/yellow]"
            )
    except Exception as e:
        console.print(f"[red]Experiment failed: {e}[/red]")
        raise typer.Exit(code=1)
    finally:
        db.close()


@app.command()
def evaluate(
    config_dir: Path = typer.Option(_DEFAULT_CONFIGS, "--configs", "-c"),
    database: Path = typer.Option(_DEFAULT_DB, "--database", "-d"),
    prompts_dir: Path = typer.Option(_DEFAULT_PROMPTS, "--prompts", "-p"),
    run_id: Optional[str] = typer.Option(
        None, "--run-id", help="Evaluate a specific run (default: all unevaluated)"
    ),
    force: bool = typer.Option(False, "--force", help="Re-evaluate already scored designs"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Report what would be evaluated"),
) -> None:
    """Evaluate completed runs using the evaluator pipeline."""
    from consortium.evaluation.pipeline import EvaluationPipeline
    from consortium.providers.registry import ProviderRegistry

    config, db = _load_config_and_db(config_dir, database)

    async def _run() -> dict:
        registry = ProviderRegistry()
        async with registry:
            pipeline = EvaluationPipeline(config, db, prompts_dir, registry=registry)
            return await pipeline.evaluate(run_id=run_id, force=force, dry_run=dry_run)

    try:
        stats = asyncio.run(_run())

        table = Table(title="Evaluation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right")

        for key, value in stats.items():
            table.add_row(key, str(value))

        console.print(table)
    except Exception as e:
        console.print(f"[red]Evaluation failed: {e}[/red]")
        raise typer.Exit(code=1)
    finally:
        db.close()


@app.command()
def batch(
    config_dir: Path = typer.Option(_DEFAULT_CONFIGS, "--configs", "-c"),
    database: Path = typer.Option(_DEFAULT_DB, "--database", "-d"),
    prompts_dir: Path = typer.Option(_DEFAULT_PROMPTS, "--prompts", "-p"),
    variants: Optional[str] = typer.Option(
        None, "--variants", help="Comma-separated variant IDs (default: all)"
    ),
    tasks: Optional[str] = typer.Option(
        None, "--tasks", help="Comma-separated task IDs (default: all)"
    ),
    repetitions: Optional[int] = typer.Option(
        None, "--reps", help="Number of repetitions (default: from config)"
    ),
) -> None:
    """Run the experiment matrix using batch APIs for efficiency."""
    from consortium.batch.runner import BatchExperimentRunner
    from consortium.providers.registry import ProviderRegistry

    config, db = _load_config_and_db(config_dir, database)

    variant_list = variants.split(",") if variants else None
    task_list = tasks.split(",") if tasks else None

    async def _run() -> dict:
        registry = ProviderRegistry()
        async with registry:
            runner = BatchExperimentRunner(config, db, str(prompts_dir), registry=registry)
            return await runner.run_batch_experiment(
                variants=variant_list,
                tasks=task_list,
                repetitions=repetitions,
            )

    try:
        stats = asyncio.run(_run())

        table = Table(title="Batch Experiment Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right")

        for key, value in stats.items():
            style = ""
            if key == "failed" and value > 0:
                style = "red"
            elif key == "completed":
                style = "green"
            table.add_row(key, str(value), style=style)

        console.print(table)
    except Exception as e:
        console.print(f"[red]Batch experiment failed: {e}[/red]")
        raise typer.Exit(code=1)
    finally:
        db.close()


@app.command()
def status(
    config_dir: Path = typer.Option(_DEFAULT_CONFIGS, "--configs", "-c"),
    database: Path = typer.Option(_DEFAULT_DB, "--database", "-d"),
) -> None:
    """Show experiment progress: how many runs completed, pending, failed."""
    from consortium.config.loader import load_full_config
    from consortium.storage.database import Database

    if not database.exists():
        console.print("[yellow]No database found. Run 'consortium db init' first.[/yellow]")
        raise typer.Exit(code=1)

    config = load_full_config(config_dir)
    db = Database(database)

    try:
        # Count by status
        rows = db.conn.execute(
            "SELECT status, COUNT(*) as cnt FROM runs GROUP BY status"
        ).fetchall()
        status_counts = {row["status"]: row["cnt"] for row in rows}

        # Total expected
        n_variants = len(config.variants)
        n_tasks = len(config.tasks)
        n_reps = config.experiment.repetitions
        total_expected = n_variants * n_tasks * n_reps

        # Cost
        cost_row = db.conn.execute(
            "SELECT COALESCE(SUM(total_cost_usd), 0) as total_cost FROM runs"
        ).fetchone()
        total_cost = cost_row["total_cost"]

        # Evaluations
        eval_row = db.conn.execute(
            "SELECT COUNT(DISTINCT design_id) as cnt FROM scores_median"
        ).fetchone()
        evaluated = eval_row["cnt"]

        # Display
        console.print(f"\n[bold]Experiment: {config.experiment.name}[/bold]")
        console.print(
            f"Matrix: {n_variants} variants × {n_tasks} tasks × "
            f"{n_reps} reps = {total_expected} runs"
        )
        console.print()

        table = Table(title="Run Progress")
        table.add_column("Status", style="cyan")
        table.add_column("Count", justify="right")

        completed = status_counts.get("completed", 0)
        table.add_row("completed", f"[green]{completed}[/green]")
        table.add_row("running", str(status_counts.get("running", 0)))
        table.add_row("failed", f"[red]{status_counts.get('failed', 0)}[/red]")
        table.add_row("pending (not started)", str(total_expected - sum(status_counts.values())))
        table.add_row("───", "───")
        table.add_row("total expected", str(total_expected))

        console.print(table)
        console.print(f"\nEvaluated designs: {evaluated}/{completed}")
        console.print(f"Total cost: ${total_cost:.2f}")
    finally:
        db.close()


@app.command("full-experiment")
def full_experiment(
    config_dir: Path = typer.Option(_DEFAULT_CONFIGS, "--configs", "-c"),
    database: Path = typer.Option(_DEFAULT_DB, "--database", "-d"),
    prompts_dir: Path = typer.Option(_DEFAULT_PROMPTS, "--prompts", "-p"),
    variants: Optional[str] = typer.Option(
        None, "--variants", help="Comma-separated variant IDs (default: all from config)"
    ),
    tasks: Optional[str] = typer.Option(
        None, "--tasks", help="Comma-separated task IDs (default: all from config)"
    ),
    repetitions: Optional[int] = typer.Option(
        None, "--reps", help="Number of repetitions (default: from config)"
    ),
    force: bool = typer.Option(False, "--force", help="Re-run even completed runs"),
    skip_analysis: bool = typer.Option(
        False, "--skip-analysis", help="Skip statistical analysis after runs"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show experiment plan without executing"),
) -> None:
    """Run the full pipeline: generate → evaluate + coherence → analyze.

    Runs stream through stages as they complete — no waiting for all
    generation to finish before starting evaluation. All stages share
    batching providers for maximum throughput.
    """
    from consortium.pipeline.full_experiment import FullExperimentRunner

    config, db = _load_config_and_db(config_dir, database)

    variant_list = variants.split(",") if variants else None
    task_list = tasks.split(",") if tasks else None

    if dry_run:
        _print_full_experiment_dry_run(
            config,
            db,
            variant_list,
            task_list,
            repetitions,
            force,
            skip_analysis,
        )
        db.close()
        return

    runner = FullExperimentRunner(config, db, prompts_dir)

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

        # Rich table output
        table = Table(
            title="Full Experiment Results",
            show_header=True,
            header_style="bold",
        )
        table.add_column("Stage", style="cyan")
        table.add_column("Completed", justify="right")
        table.add_column("Total", justify="right")
        table.add_column("Status", justify="center")

        table.add_row(
            "Generation",
            str(stats.generation_completed),
            str(stats.total_runs),
            "[green]✓[/green]" if stats.generation_completed == stats.total_runs else "…",
        )
        table.add_row(
            "Evaluation",
            str(stats.evaluation_completed),
            str(stats.total_runs),
            "[green]✓[/green]" if stats.evaluation_completed == stats.total_runs else "…",
        )
        table.add_row(
            "Coherence",
            str(stats.coherence_completed),
            str(stats.total_runs),
            "[green]✓[/green]" if stats.coherence_completed == stats.total_runs else "…",
        )
        table.add_row(
            "Analysis",
            "—",
            "—",
            "[green]✓[/green]"
            if stats.analysis_complete
            else ("[dim]skipped[/dim]" if skip_analysis else "…"),
        )

        if stats.failed > 0:
            table.add_row(
                "[red]Failed[/red]",
                f"[red]{stats.failed}[/red]",
                str(stats.total_runs),
                "[red]⚠[/red]",
            )

        console.print(table)

        if stats.failed > 0:
            console.print(
                f"\n[yellow]⚠ {stats.failed} run(s) failed. "
                "Re-run without --force to resume.[/yellow]"
            )
        else:
            console.print(f"\n[green]✓ Pipeline complete: {stats.total_runs} runs[/green]")

    except Exception as e:
        console.print(f"[red]Pipeline failed: {e}[/red]")
        raise typer.Exit(code=1)
    finally:
        db.close()


def _print_full_experiment_dry_run(
    config,
    db,
    variant_list: list[str] | None,
    task_list: list[str] | None,
    repetitions: int | None,
    force: bool,
    skip_analysis: bool,
) -> None:
    """Print a comprehensive experiment plan without executing."""
    from rich.panel import Panel

    from consortium.pipeline.full_experiment import FullExperimentRunner

    exp = config.experiment
    variant_ids = variant_list or [v.split("_")[0] if "_" in v else v for v in exp.variants]
    task_ids = task_list or [t.split("_")[0] if "_" in t else t for t in exp.tasks]
    n_reps = repetitions or exp.repetitions
    total_runs = len(variant_ids) * len(task_ids) * n_reps

    # ── Header ──────────────────────────────────────────────────────────
    console.print()
    console.print(
        Panel(
            f"[bold]{exp.name}[/bold]\n"
            f"[dim]{len(variant_ids)} variants × {len(task_ids)} tasks × {n_reps} reps "
            f"= [bold]{total_runs}[/bold] total runs[/dim]",
            title="[bold cyan]Full Experiment — Dry Run[/bold cyan]",
            border_style="cyan",
        )
    )

    # ── Models ──────────────────────────────────────────────────────────
    model_table = Table(
        title="Models",
        show_header=True,
        header_style="bold",
        title_style="bold magenta",
    )
    model_table.add_column("ID", style="cyan")
    model_table.add_column("Provider")
    model_table.add_column("API Model")
    model_table.add_column("Batching", justify="center")
    model_table.add_column("Window / Max", justify="right")
    model_table.add_column("Context", justify="right")
    model_table.add_column("Input $/M", justify="right")
    model_table.add_column("Output $/M", justify="right")

    for mid, mc in sorted(config.models.items()):
        batching = mc.batching
        if batching and batching.enabled:
            batch_str = "[green]✓[/green]"
            window_str = f"{batching.window_ms:.0f}ms / {batching.max_batch_size}"
        else:
            batch_str = "[dim]—[/dim]"
            window_str = "[dim]—[/dim]"

        model_table.add_row(
            mid,
            mc.provider,
            mc.api_model,
            batch_str,
            window_str,
            f"{mc.context_window:,}",
            f"${mc.pricing.input:.2f}",
            f"${mc.pricing.output:.2f}",
        )

    console.print(model_table)

    # ── Variants ────────────────────────────────────────────────────────
    variant_table = Table(
        title="Variants",
        show_header=True,
        header_style="bold",
        title_style="bold magenta",
    )
    variant_table.add_column("ID", style="cyan")
    variant_table.add_column("Name")
    variant_table.add_column("Agents")
    variant_table.add_column("Models Used")
    variant_table.add_column("Rounds", justify="right")

    for vid in variant_ids:
        vc = config.get_variant(vid)
        # Collect models used by this variant
        agent_roles = []
        models_used = set()
        agents = vc.agents
        for role in [
            "leader",
            "designer",
            "reviewers",
            "parallel_leaders",
            "merger",
            "adversarial_reviewer",
            "judge",
            "evaluator",
        ]:
            agent = getattr(agents, role, None)
            if agent is not None:
                count = getattr(agent, "count", 1)
                label = f"{role}" + (f" ×{count}" if count > 1 else "")
                agent_roles.append(label)
                models_used.add(agent.model)
        # participants is a list[AgentConfig]
        if agents.participants:
            agent_roles.append(f"participants ×{len(agents.participants)}")
            for p in agents.participants:
                models_used.add(p.model)
        # debaters is a list[AgentConfig]
        if agents.debaters:
            agent_roles.append(f"debaters ×{len(agents.debaters)}")
            for d in agents.debaters:
                models_used.add(d.model)
        for spec in agents.specialists:
            agent_roles.append(f"specialist:{spec.id}")
            if spec.model:
                models_used.add(spec.model)

        variant_table.add_row(
            vid,
            vc.name,
            ", ".join(agent_roles) if agent_roles else "[dim]—[/dim]",
            ", ".join(sorted(models_used)) if models_used else "[dim]—[/dim]",
            str(vc.workflow.max_rounds),
        )

    console.print(variant_table)

    # ── Tasks ───────────────────────────────────────────────────────────
    task_table = Table(
        title="Tasks",
        show_header=True,
        header_style="bold",
        title_style="bold magenta",
    )
    task_table.add_column("ID", style="cyan")
    task_table.add_column("Name")
    task_table.add_column("Complexity")
    task_table.add_column("Type")
    task_table.add_column("Rubric")

    for tid in task_ids:
        tc = config.get_task(tid)
        complexity_style = {
            "simple": "green",
            "medium": "yellow",
            "complex": "red",
        }.get(tc.complexity, "")
        task_table.add_row(
            tid,
            tc.name,
            f"[{complexity_style}]{tc.complexity}[/{complexity_style}]",
            tc.design_type,
            tc.rubric or "[dim]—[/dim]",
        )

    console.print(task_table)

    # ── Run Matrix ──────────────────────────────────────────────────────
    runner = FullExperimentRunner(config, db, Path("prompts"))
    run_matrix = [
        (vid, tid, rep) for vid in variant_ids for tid in task_ids for rep in range(n_reps)
    ]
    pending = runner._get_pending_runs(run_matrix, force=force)

    pending_gen = len(pending["generate"])
    pending_eval = len(pending["evaluate"])
    pending_coh = len(pending["coherence"])
    already_done = total_runs - pending_gen

    status_table = Table(
        title="Run Matrix Status",
        show_header=True,
        header_style="bold",
        title_style="bold magenta",
    )
    status_table.add_column("Stage", style="cyan")
    status_table.add_column("Pending", justify="right", style="yellow")
    status_table.add_column("Done", justify="right", style="green")
    status_table.add_column("Total", justify="right")

    status_table.add_row(
        "Generation",
        str(pending_gen),
        str(already_done),
        str(total_runs),
    )
    status_table.add_row(
        "Evaluation",
        str(pending_eval),
        str(total_runs - pending_eval),
        str(total_runs),
    )
    status_table.add_row(
        "Coherence",
        str(pending_coh),
        str(total_runs - pending_coh),
        str(total_runs),
    )
    status_table.add_row(
        "Analysis",
        "[dim]after all runs[/dim]" if not skip_analysis else "[dim]skipped[/dim]",
        "[dim]—[/dim]",
        "[dim]—[/dim]",
    )

    console.print(status_table)

    # ── Pipeline Settings ───────────────────────────────────────────────
    settings_table = Table(
        title="Pipeline Settings",
        show_header=True,
        header_style="bold",
        title_style="bold magenta",
        show_lines=False,
    )
    settings_table.add_column("Setting", style="cyan")
    settings_table.add_column("Value")

    settings_table.add_row("Max concurrent runs", str(exp.limits.max_concurrent_runs))
    settings_table.add_row("Max cost per run", f"${exp.limits.max_cost_per_run_usd:.2f}")
    settings_table.add_row("Max total cost", f"${exp.limits.max_total_cost_usd:.2f}")
    settings_table.add_row("Timeout per run", f"{exp.limits.timeout_seconds}s")
    settings_table.add_row("Max retries", str(exp.limits.max_retries))
    settings_table.add_row("Force re-run", "[green]yes[/green]" if force else "[dim]no[/dim]")
    settings_table.add_row(
        "Skip analysis", "[yellow]yes[/yellow]" if skip_analysis else "[dim]no[/dim]"
    )
    settings_table.add_row("Evaluator model", config.evaluator.model)
    settings_table.add_row("Eval runs per design", str(config.evaluator.runs_per_design))
    settings_table.add_row("Database", str(exp.database))
    settings_table.add_row("Output dir", str(exp.output_dir))

    console.print(settings_table)

    # ── Summary ─────────────────────────────────────────────────────────
    if pending_gen == 0 and pending_eval == 0 and pending_coh == 0:
        console.print("\n[green]✓ All runs already complete. Use --force to re-run.[/green]")
    else:
        console.print(
            f"\n[cyan]→ Would execute {pending_gen} generation(s), "
            f"{pending_eval} evaluation(s), {pending_coh} coherence check(s)[/cyan]"
        )
    console.print()


@app.command()
def coherence(
    config_dir: Path = typer.Option(_DEFAULT_CONFIGS, "--configs", "-c"),
    database: Path = typer.Option(_DEFAULT_DB, "--database", "-d"),
    prompts_dir: Path = typer.Option(_DEFAULT_PROMPTS, "--prompts", "-p"),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Check only this run's designs"),
    all_designs: bool = typer.Option(False, "--all", help="Check all final designs"),
    force: bool = typer.Option(False, "--force", help="Re-check already checked designs"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Report what would be checked"),
) -> None:
    """Run coherence checks on final designs."""
    from consortium.evaluation.pipeline import EvaluationPipeline
    from consortium.providers.registry import ProviderRegistry

    config, db = _load_config_and_db(config_dir, database)

    async def _run() -> dict:
        registry = ProviderRegistry()
        async with registry:
            pipeline = EvaluationPipeline(config, db, prompts_dir, registry=registry)
            return await pipeline.run_all_coherence_checks(
                run_id=run_id,
                force=force,
                dry_run=dry_run,
            )

    try:
        stats = asyncio.run(_run())

        table = Table(title="Coherence Check Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right")

        for key, value in stats.items():
            style = ""
            if key == "failed" and value > 0:
                style = "red"
            elif key == "checked":
                style = "green"
            table.add_row(key, str(value), style=style)

        console.print(table)

        # Show contradictions found
        contradictions = db.conn.execute(
            "SELECT COUNT(*) as cnt FROM coherence_checks WHERE contradicts = TRUE"
        ).fetchone()
        if contradictions and contradictions["cnt"] > 0:
            console.print(
                f"\n[yellow]⚠ {contradictions['cnt']} contradiction(s) found. "
                "Review with 'consortium db export'.[/yellow]"
            )
    except Exception as e:
        console.print(f"[red]Coherence check failed: {e}[/red]")
        raise typer.Exit(code=1)
    finally:
        db.close()
