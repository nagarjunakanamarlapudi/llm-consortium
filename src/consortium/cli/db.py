"""CLI command: database operations."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()

ALL_TABLES = [
    "runs",
    "designs",
    "reviews",
    "evaluations",
    "scores_median",
    "coherence_checks",
    "traces",
    "batches",
]


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
        "json",
        "--format",
        "-f",
        help="Export format: json or csv",
    ),
) -> None:
    """Export all database tables to JSON (or CSV) files.

    Exports are text-based, diffable, and suitable for committing to git.
    Use 'consortium db load' to import them back into a database.
    """
    import csv
    import json
    import sqlite3

    from consortium.storage.database import Database

    if not database.exists():
        console.print(f"[red]Database not found: {database}[/red]")
        raise typer.Exit(code=1)

    output.mkdir(parents=True, exist_ok=True)
    total_rows = 0

    with Database(database) as db:
        for table_name in ALL_TABLES:
            try:
                rows = db.conn.execute(f"SELECT * FROM {table_name}").fetchall()  # noqa: S608
                if not rows:
                    continue

                out_path = output / f"{table_name}.{fmt}"

                if fmt == "json":
                    data = [dict(row) for row in rows]
                    with open(out_path, "w") as f:
                        json.dump(data, f, indent=2, default=str)
                elif fmt == "csv":
                    with open(out_path, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(rows[0].keys())
                        writer.writerows(rows)
                else:
                    console.print(f"[red]Unsupported format: {fmt}[/red]")
                    raise typer.Exit(code=1)

                total_rows += len(rows)
                console.print(f"  {table_name}: {len(rows)} rows → {out_path}")
            except sqlite3.OperationalError:
                pass

    console.print(f"\n[green]✓ Exported {total_rows} total rows to {output}/[/green]")


@app.command()
def load(
    input_dir: Path = typer.Argument(
        ...,
        help="Directory containing exported JSON files",
    ),
    database: Path = typer.Option(
        Path("data/consortium.db"),
        "--database",
        "-d",
        help="Target database file path",
    ),
    clear: bool = typer.Option(
        False,
        "--clear",
        help="Clear existing data before loading (fresh import)",
    ),
) -> None:
    """Load exported JSON files into the database.

    Reads JSON files from the input directory and inserts them into the
    database. Files are matched to tables by filename (e.g. runs.json → runs).

    Use --clear to wipe the database before loading (for a clean restore).
    Without --clear, existing rows with the same primary key are skipped.
    """
    import json
    import sqlite3

    from consortium.storage.database import Database

    if not input_dir.exists():
        console.print(f"[red]Input directory not found: {input_dir}[/red]")
        raise typer.Exit(code=1)

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        console.print(f"[yellow]No .json files found in {input_dir}[/yellow]")
        raise typer.Exit(code=1)

    with Database(database) as db:
        db.init_schema()

        if clear:
            console.print("[yellow]Clearing existing data...[/yellow]")
            # Delete in reverse dependency order
            for table_name in reversed(ALL_TABLES):
                try:
                    db.conn.execute(f"DELETE FROM {table_name}")  # noqa: S608
                except sqlite3.OperationalError:
                    pass
            db.conn.commit()

        total_rows = 0

        # Load in dependency order (runs first, then tables that reference runs)
        for table_name in ALL_TABLES:
            json_path = input_dir / f"{table_name}.json"
            if not json_path.exists():
                continue

            with open(json_path) as f:
                data = json.load(f)

            if not data:
                continue

            columns = list(data[0].keys())
            placeholders = ", ".join(["?"] * len(columns))
            col_names = ", ".join(columns)

            loaded = 0
            skipped = 0
            for row in data:
                values = [row.get(col) for col in columns]
                try:
                    db.conn.execute(
                        f"INSERT OR IGNORE INTO {table_name} ({col_names}) VALUES ({placeholders})",  # noqa: S608
                        values,
                    )
                    if db.conn.execute("SELECT changes()").fetchone()[0] > 0:
                        loaded += 1
                    else:
                        skipped += 1
                except sqlite3.Error as e:
                    console.print(f"  [red]Error loading row into {table_name}: {e}[/red]")
                    skipped += 1

            db.conn.commit()
            total_rows += loaded
            skip_msg = f" ({skipped} skipped)" if skipped else ""
            console.print(f"  {table_name}: {loaded} rows loaded{skip_msg}")

    console.print(f"\n[green]✓ Loaded {total_rows} total rows into {database}[/green]")
