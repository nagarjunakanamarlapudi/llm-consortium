"""CLI commands for timeline and observability views."""

from __future__ import annotations

from pathlib import Path

import typer

from consortium.config.loader import load_full_config
from consortium.observability.exporter import (
    export_chrome_trace,
    export_csv,
    export_json,
    export_rich,
    export_rich_comparison,
    export_rich_summary,
)
from consortium.observability.timeline import TimelineBuilder
from consortium.storage.database import Database

app = typer.Typer(no_args_is_help=True)


def _get_builder(config_dir: str = "configs") -> TimelineBuilder:
    config = load_full_config(config_dir)
    db = Database(config.experiment.database)
    return TimelineBuilder(db)


@app.command()
def show(
    run_id: str | None = typer.Option(None, "--run-id", help="Show timeline for a specific run"),
    variant: str | None = typer.Option(None, "--variant", help="Show summary for a variant"),
    task: str | None = typer.Option(None, "--task", help="Filter by task"),
    compare: str | None = typer.Option(
        None, "--compare", help="Comma-separated variant IDs to compare"
    ),
    fmt: str = typer.Option("rich", "--format", help="Output format: rich|json|csv|chrome"),
    output: str | None = typer.Option(None, "--output", help="Output file path"),
    config_dir: str = typer.Option("configs", "--config-dir", help="Config directory"),
) -> None:
    """Show run timeline, variant summary, or comparison view."""
    builder = _get_builder(config_dir)

    if run_id:
        timeline = builder.build_run_timeline(run_id)
        if fmt == "json" and output:
            export_json(timeline, Path(output))
        elif fmt == "csv" and output:
            export_csv(timeline, Path(output))
        elif fmt == "chrome" and output:
            export_chrome_trace(timeline, Path(output))
        else:
            export_rich(timeline)

    elif compare and task:
        variant_ids = [v.strip() for v in compare.split(",")]
        comparison = builder.build_comparison(variant_ids, task)
        if fmt == "json" and output:
            from dataclasses import asdict
            import json

            Path(output).parent.mkdir(parents=True, exist_ok=True)
            Path(output).write_text(json.dumps(asdict(comparison), indent=2, default=str))
        else:
            export_rich_comparison(comparison)

    elif variant:
        summary = builder.build_variant_summary(variant, task)
        export_rich_summary(summary)

    else:
        typer.echo("Provide --run-id, --variant, or --compare with --task")
        raise typer.Exit(1)


@app.command()
def summary(
    by: str = typer.Option("cost", "--by", help="Sort by: cost|tokens"),
    task: str | None = typer.Option(None, "--task", help="Filter by task"),
    config_dir: str = typer.Option("configs", "--config-dir", help="Config directory"),
) -> None:
    """Show cost/token summary across all variants."""
    config = load_full_config(config_dir)
    db = Database(config.experiment.database)
    builder = TimelineBuilder(db)

    # Get all variant IDs from completed runs
    rows = db.conn.execute(
        "SELECT DISTINCT variant_id FROM runs WHERE status = 'completed'"
    ).fetchall()
    variant_ids = sorted(r["variant_id"] for r in rows)

    if not variant_ids:
        typer.echo("No completed runs found.")
        raise typer.Exit(1)

    from rich.console import Console
    from rich.table import Table

    console = Console()

    title = "Summary by Cost" if by == "cost" else "Summary by Tokens"
    if task:
        title += f" — Task: {task}"
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Variant", style="cyan")
    table.add_column("Runs", justify="right")
    table.add_column("Mean Duration", justify="right")
    table.add_column("Mean Cost ($)", justify="right")
    table.add_column("Mean Tokens", justify="right")

    summaries = [builder.build_variant_summary(vid, task) for vid in variant_ids]
    summaries = [s for s in summaries if s.run_count > 0]

    if by == "cost":
        summaries.sort(key=lambda s: s.mean_cost_usd)
    else:
        summaries.sort(key=lambda s: s.mean_tokens)

    for s in summaries:
        table.add_row(
            s.variant_id,
            str(s.run_count),
            f"{s.mean_duration_ms / 1000:.1f}s",
            f"${s.mean_cost_usd:.4f}",
            f"{s.mean_tokens:,}",
        )

    console.print(table)
