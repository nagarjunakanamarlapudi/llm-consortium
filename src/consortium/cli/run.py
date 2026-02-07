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
    config_dir: Path, database: Path,
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

    config, db = _load_config_and_db(config_dir, database)
    engine = OrchestratorEngine(config, db, prompts_dir)

    try:
        design = asyncio.run(
            engine.run(variant, task, rep, resume=resume, force=force)
        )
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
    dry_run: bool = typer.Option(False, "--dry-run", help="Report what would run without executing"),
) -> None:
    """Run the full experiment matrix (variants × tasks × repetitions)."""
    from consortium.orchestrator.engine import ExperimentRunner

    config, db = _load_config_and_db(config_dir, database)
    runner = ExperimentRunner(config, db, prompts_dir)

    variant_list = variants.split(",") if variants else None
    task_list = tasks.split(",") if tasks else None

    try:
        stats = asyncio.run(
            runner.run_experiment(
                variants=variant_list,
                tasks=task_list,
                repetitions=repetitions,
                resume=resume,
                force=force,
                dry_run=dry_run,
            )
        )

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

    config, db = _load_config_and_db(config_dir, database)
    pipeline = EvaluationPipeline(config, db, prompts_dir)

    try:
        stats = asyncio.run(
            pipeline.evaluate(run_id=run_id, force=force, dry_run=dry_run)
        )

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

    config, db = _load_config_and_db(config_dir, database)
    runner = BatchExperimentRunner(config, db, str(prompts_dir))

    variant_list = variants.split(",") if variants else None
    task_list = tasks.split(",") if tasks else None

    try:
        stats = asyncio.run(
            runner.run_batch_experiment(
                variants=variant_list,
                tasks=task_list,
                repetitions=repetitions,
            )
        )

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
