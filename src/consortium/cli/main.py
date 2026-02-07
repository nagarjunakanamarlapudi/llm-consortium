"""Top-level CLI for the LLM Consortium experiment framework."""

from __future__ import annotations

import typer

from consortium.cli.db import app as db_app
from consortium.cli.validate import app as validate_app

app = typer.Typer(
    name="consortium",
    help="LLM Consortium Experiment Framework",
    no_args_is_help=True,
)

app.add_typer(validate_app, name="validate", help="Validate configuration files")
app.add_typer(db_app, name="db", help="Database operations (init, stats, export)")


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
    debug: bool = typer.Option(False, "--debug", help="Debug mode"),
) -> None:
    """LLM Consortium — multi-agent experiment framework."""
    import logging

    import structlog

    log_level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
    )


if __name__ == "__main__":
    app()
