"""Top-level CLI for the LLM Consortium experiment framework."""

from __future__ import annotations

import typer

from consortium.cli.analyze import app as analyze_app
from consortium.cli.db import app as db_app
from consortium.cli.run import app as run_app
from consortium.cli.timeline import app as timeline_app
from consortium.cli.validate import app as validate_app
from consortium.cli.verify import app as verify_app

app = typer.Typer(
    name="consortium",
    help="LLM Consortium Experiment Framework",
    no_args_is_help=True,
)

app.add_typer(run_app, name="run", help="Run experiments (single, experiment, evaluate)")
app.add_typer(validate_app, name="validate", help="Validate configuration files")
app.add_typer(db_app, name="db", help="Database operations (init, stats, export)")
app.add_typer(timeline_app, name="timeline", help="LLM call timeline views")
app.add_typer(analyze_app, name="analyze", help="Statistical analysis and visualization")
app.add_typer(verify_app, name="verify", help="E2E prompt and data verification checks")


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
    debug: bool = typer.Option(False, "--debug", help="Debug mode"),
    env_file: str = typer.Option(".env", "--env", help="Path to .env file"),
) -> None:
    """LLM Consortium — multi-agent experiment framework."""
    import logging
    from pathlib import Path

    import structlog

    # Load .env file into os.environ
    env_path = Path(env_file)
    if env_path.exists():
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)

    log_level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
    )


if __name__ == "__main__":
    app()
