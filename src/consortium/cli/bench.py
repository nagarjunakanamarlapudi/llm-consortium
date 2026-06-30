"""CLI command: coding-benchmark scoring (Paper #2, Phase 0 scaffolding).

Mirrors ``cli/run.py`` style. In Phase 0 only two verbs exist:

* ``score`` — execute the benchmark harness in a sandbox over the final code
  artifacts of completed coding runs and persist objective pass/fail results to
  ``code_results``. Pure sandboxed test execution — **no provider/LLM/API calls**.
* ``list``  — list the benchmark runners registered in the harness registry
  (Phase 0 ships only the ``function-smoke`` runner).

Code *generation* and the ``bench run`` verb (topology generation + end-to-end
scoring) arrive in Phase 1.

Phase 0 scores with the host-subprocess ``LocalSandbox`` (dev/test only — it runs
our own fixtures, never untrusted model output). Phase 1 introduces a sandbox
selector and makes the hardened ``DockerSandbox`` the default before any
model-generated code is executed.

All heavy imports are performed lazily inside the command bodies so that
registering this Typer app in :mod:`consortium.cli.main` never depends on the
sibling Phase 0 modules (``evaluation/code_pipeline.py``, ``execution/``) being
present yet.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()

_DEFAULT_DB = Path("data/consortium.db")


@app.command()
def score(
    database: Path = typer.Option(_DEFAULT_DB, "--database", "-d", help="Path to experiment DB"),
    run_id: str | None = typer.Option(
        None, "--run-id", help="Score only this run's code artifacts (default: all unscored)"
    ),
    force: bool = typer.Option(False, "--force", help="Re-score already scored artifacts"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Report what would be scored"),
) -> None:
    """Score completed coding runs via sandboxed benchmark test execution.

    Runs the harness over each final code artifact and writes objective pass/fail
    results to ``code_results``. This is execution-only — it issues **no
    provider/LLM/API calls**. Code generation and the richer ``bench run`` verb
    land in Phase 1.
    """
    from consortium.evaluation.code_pipeline import CodeEvaluationPipeline
    from consortium.storage.database import Database

    db = Database(database)
    db.init_schema()

    try:
        pipeline = CodeEvaluationPipeline(db)
        stats = pipeline.evaluate(run_id=run_id, force=force, dry_run=dry_run)

        title = "Code Scoring Results" + (" (dry run)" if dry_run else "")
        table = Table(title=title)
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right")

        for key in ("total", "scored", "passed", "failed"):
            value = stats.get(key, 0)
            style = ""
            if key == "failed" and value:
                style = "red"
            elif key in ("scored", "passed"):
                style = "green"
            table.add_row(key, str(value), style=style)

        console.print(table)
    except Exception as e:
        console.print(f"[red]Code scoring failed: {e}[/red]")
        raise typer.Exit(code=1) from e
    finally:
        db.close()


@app.command("list")
def list_runners() -> None:
    """List benchmark runners registered in the harness registry.

    Phase 0 ships only the ``function-smoke`` runner; the real benchmark
    harnesses (LiveCodeBench, BigCodeBench, SWE-bench, ...) land in Phase 1+.
    """
    from consortium.execution.harness import available_runners

    names = available_runners()
    if not names:
        console.print("[yellow]No benchmark runners registered.[/yellow]")
        return

    table = Table(title="Registered Benchmark Runners")
    table.add_column("Runner", style="cyan")

    for name in names:
        table.add_row(name)

    console.print(table)
