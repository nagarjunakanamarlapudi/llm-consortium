"""CLI command: verify experiment data integrity."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def prompts(
    database: Path = typer.Option(
        ...,
        "--database",
        "-d",
        help="Path to experiment SQLite database",
        exists=True,
    ),
    checks: str = typer.Option(
        "",
        "--checks",
        "-k",
        help="Comma-separated check names to run (default: all)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show failure details for each check",
    ),
) -> None:
    """Run E2E prompt and data verification checks against an experiment DB."""
    from consortium.verify.checks import CHECK_NAMES, run_all_checks

    names = [n.strip() for n in checks.split(",") if n.strip()] or None

    if names:
        unknown = [n for n in names if n not in CHECK_NAMES]
        if unknown:
            console.print(f"[red]Unknown check(s): {', '.join(unknown)}[/red]")
            console.print(f"Available: {', '.join(sorted(CHECK_NAMES))}")
            raise typer.Exit(code=1)

    console.print(f"[bold]Verifying:[/bold] {database}\n")
    results = run_all_checks(database, names=names)

    # Summary table
    table = Table(title="Prompt Verification Results")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Check", style="cyan")
    table.add_column("Total", justify="right")
    table.add_column("Failures", justify="right")
    table.add_column("Status")

    fail_count = 0
    for i, r in enumerate(results, 1):
        if r.passed:
            status = "[green]PASS[/green]"
        elif r.total == 0:
            status = "[yellow]SKIP[/yellow]"
        else:
            status = "[red]FAIL[/red]"
            fail_count += 1
        table.add_row(
            str(i),
            r.name,
            str(r.total),
            str(r.failures),
            status,
        )

    console.print(table)

    # Details
    if verbose:
        for r in results:
            if r.details:
                console.print(f"\n[bold]{r.name}[/bold] details:")
                for detail in r.details:
                    console.print(f"  {detail}")

    console.print()
    if fail_count:
        console.print(
            f"[red]{fail_count} check(s) failed out of {len(results)}[/red]"
        )
        raise typer.Exit(code=1)

    console.print(
        f"[green]All {len(results)} checks passed[/green]"
    )
