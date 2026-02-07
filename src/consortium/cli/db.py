"""CLI command: database operations."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()


@app.command()
def init(
    database: Path = typer.Option(
        Path("data/consortium.db"),
        "--database",
        "-d",
        help="Database file path",
    ),
) -> None:
    """Initialize the database schema."""
    from consortium.storage.database import Database

    with Database(database) as db:
        db.init_schema()
    console.print(f"[green]Database initialized at {database}[/green]")


@app.command()
def stats(
    database: Path = typer.Option(
        Path("data/consortium.db"),
        "--database",
        "-d",
        help="Database file path",
    ),
) -> None:
    """Show database statistics."""
    from consortium.storage.database import Database

    if not database.exists():
        console.print(f"[red]Database not found: {database}[/red]")
        console.print("Run 'consortium db init' first.")
        raise typer.Exit(code=1)

    with Database(database) as db:
        version = db.get_schema_version()
        counts = db.stats()

    table = Table(title="Database Statistics")
    table.add_column("Table", style="cyan")
    table.add_column("Count", justify="right")

    for table_name, count in counts.items():
        table.add_row(table_name, str(count))

    console.print(f"Schema version: {version}")
    console.print(table)


@app.command()
def export(
    database: Path = typer.Option(
        Path("data/consortium.db"),
        "--database",
        "-d",
        help="Database file path",
    ),
    output: Path = typer.Option(
        Path("data/exports"),
        "--output",
        "-o",
        help="Output directory",
    ),
    fmt: str = typer.Option(
        "csv",
        "--format",
        "-f",
        help="Export format (csv)",
    ),
) -> None:
    """Export database tables to files."""
    import csv
    import sqlite3

    from consortium.storage.database import Database

    if not database.exists():
        console.print(f"[red]Database not found: {database}[/red]")
        raise typer.Exit(code=1)

    output.mkdir(parents=True, exist_ok=True)

    tables = ["runs", "designs", "evaluations", "scores_median", "traces"]
    with Database(database) as db:
        for table_name in tables:
            try:
                rows = db.conn.execute(f"SELECT * FROM {table_name}").fetchall()  # noqa: S608
                if not rows:
                    continue
                out_path = output / f"{table_name}.{fmt}"
                with open(out_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(rows[0].keys())
                    writer.writerows(rows)
                console.print(f"Exported {len(rows)} rows to {out_path}")
            except sqlite3.OperationalError:
                pass

    console.print(f"[green]Export complete to {output}/[/green]")
