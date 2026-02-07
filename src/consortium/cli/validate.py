"""CLI command: validate configuration files."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()


@app.callback(invoke_without_command=True)
def validate(
    config_dir: Path = typer.Option(
        Path("configs"),
        "--configs",
        "-c",
        help="Configuration directory to validate",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Strict mode: fail on warnings too",
    ),
) -> None:
    """Validate all configuration files in the given directory."""
    from consortium.config.loader import (
        load_evaluator_config,
        load_experiment_config,
        load_full_config,
        load_model_config,
        load_rubric_config,
        load_task_config,
        load_variant_config,
    )

    errors: list[tuple[str, str]] = []
    warnings: list[tuple[str, str]] = []
    successes: list[str] = []

    # Validate experiment config
    exp_path = config_dir / "experiment.yaml"
    if exp_path.exists():
        try:
            load_experiment_config(exp_path)
            successes.append("experiment.yaml")
        except Exception as e:
            errors.append(("experiment.yaml", str(e)))
    else:
        errors.append(("experiment.yaml", "File not found"))

    # Validate model configs
    models_dir = config_dir / "models"
    if models_dir.exists():
        for f in sorted(models_dir.glob("*.yaml")):
            try:
                mc = load_model_config(f)
                if mc.provider not in ("openai", "anthropic", "google", "ollama"):
                    warnings.append((f.name, f"Unknown provider: {mc.provider}"))
                successes.append(f"models/{f.name}")
            except Exception as e:
                errors.append((f"models/{f.name}", str(e)))
    else:
        warnings.append(("models/", "Directory not found"))

    # Validate variant configs
    variants_dir = config_dir / "variants"
    if variants_dir.exists():
        for f in sorted(variants_dir.glob("*.yaml")):
            try:
                load_variant_config(f)
                successes.append(f"variants/{f.name}")
            except Exception as e:
                errors.append((f"variants/{f.name}", str(e)))
    else:
        warnings.append(("variants/", "Directory not found"))

    # Validate task configs
    tasks_dir = config_dir / "tasks"
    if tasks_dir.exists():
        for f in sorted(tasks_dir.glob("*.yaml")):
            try:
                load_task_config(f)
                successes.append(f"tasks/{f.name}")
            except Exception as e:
                errors.append((f"tasks/{f.name}", str(e)))
    else:
        warnings.append(("tasks/", "Directory not found"))

    # Validate rubric configs
    rubrics_dir = config_dir / "rubrics"
    if rubrics_dir.exists():
        for f in sorted(rubrics_dir.glob("*.yaml")):
            try:
                load_rubric_config(f)
                successes.append(f"rubrics/{f.name}")
            except Exception as e:
                errors.append((f"rubrics/{f.name}", str(e)))
    else:
        warnings.append(("rubrics/", "Directory not found"))

    # Validate evaluator config
    eval_path = config_dir / "evaluator.yaml"
    if eval_path.exists():
        try:
            load_evaluator_config(eval_path)
            successes.append("evaluator.yaml")
        except Exception as e:
            errors.append(("evaluator.yaml", str(e)))

    # Cross-reference validation: try loading the full config
    try:
        full = load_full_config(config_dir)

        # Check that experiment references exist
        for variant_ref in full.experiment.variants:
            found = any(v_id.startswith(variant_ref.split("_")[0]) for v_id in full.variants)
            if not found:
                warnings.append(
                    ("experiment.yaml", f"Variant '{variant_ref}' referenced but not found")
                )

        for task_ref in full.experiment.tasks:
            found = any(t_id.startswith(task_ref.split("_")[0]) for t_id in full.tasks)
            if not found:
                warnings.append(
                    ("experiment.yaml", f"Task '{task_ref}' referenced but not found")
                )
    except Exception as e:
        errors.append(("full_config", str(e)))

    # Display results
    table = Table(title="Configuration Validation Results")
    table.add_column("File", style="cyan")
    table.add_column("Status")
    table.add_column("Details")

    for name in successes:
        table.add_row(name, "[green]PASS[/green]", "")

    for name, msg in warnings:
        table.add_row(name, "[yellow]WARN[/yellow]", msg)

    for name, msg in errors:
        table.add_row(name, "[red]FAIL[/red]", msg)

    console.print(table)
    console.print()

    if errors:
        console.print(
            f"[red]Validation failed: {len(errors)} error(s), "
            f"{len(warnings)} warning(s), {len(successes)} passed[/red]"
        )
        raise typer.Exit(code=1)

    if warnings and strict:
        console.print(
            f"[yellow]Strict mode: {len(warnings)} warning(s) treated as errors[/yellow]"
        )
        raise typer.Exit(code=1)

    console.print(
        f"[green]All {len(successes)} configs valid"
        f" ({len(warnings)} warning(s))[/green]"
    )
