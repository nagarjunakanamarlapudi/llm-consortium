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


@app.command("run")
def run(
    benchmark: str = typer.Argument(..., help="Benchmark: humanevalplus | mbppplus"),
    conditions: str = typer.Option(
        "base-sonnet,base-gpt52,base-gptoss",
        "--conditions",
        help="Comma-separated condition names (see benchmarks.conditions.ALL_CONDITIONS)",
    ),
    limit: int = typer.Option(20, "--limit", "-n", help="Number of problems (pilot subset; 0=all)"),
    reps: int = typer.Option(2, "--reps", "-r", help="Repetitions per (condition, problem)"),
    config_dir: Path = typer.Option(Path("configs"), "--configs", "-c"),
    database: Path = typer.Option(_DEFAULT_DB, "--database", "-d"),
    prompts_dir: Path = typer.Option(Path("prompts"), "--prompts", "-p"),
    max_cost: float = typer.Option(50.0, "--max-cost", help="Abort if total cost exceeds USD"),
) -> None:
    """Generate code via consortium topologies for a benchmark subset (Phase A).

    Builds an in-memory experiment (DO-inference model roster x the requested
    conditions x the benchmark subset) and runs it through the existing
    orchestrator, persisting final code artifacts to ``designs``. Score them
    afterwards with ``bench grade``. **Issues real provider/LLM calls.**
    """
    import asyncio

    from dotenv import load_dotenv

    from consortium.benchmarks import build_conditions, load_benchmark
    from consortium.config.loader import load_model_config
    from consortium.config.models import ExperimentConfig, FullConfig, LimitsConfig
    from consortium.orchestrator.engine import ExperimentRunner
    from consortium.providers.registry import ProviderRegistry
    from consortium.storage.database import Database

    load_dotenv()  # ensure DO_INFERENCE_API_KEY etc. are available

    cond_names = [c.strip() for c in conditions.split(",") if c.strip()]
    try:
        variants = build_conditions(cond_names)
        tasks_list = load_benchmark(benchmark, limit or None)
    except KeyError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from e

    tasks = {t.id: t for t in tasks_list}

    # Load every model config so any condition's model ref resolves.
    models = {}
    models_dir = config_dir / "models"
    for f in sorted(models_dir.glob("*.yaml")):
        mc = load_model_config(f)
        models[mc.id] = mc

    exp = ExperimentConfig(
        name=f"code-{benchmark}",
        variants=list(variants.keys()),
        tasks=list(tasks.keys()),
        repetitions=reps,
        database=str(database),
        limits=LimitsConfig(
            max_total_cost_usd=max_cost,
            max_cost_per_run_usd=5.0,
            max_tokens_per_run=500_000,
        ),
    )
    cfg = FullConfig(experiment=exp, models=models, variants=variants, tasks=tasks)

    db = Database(database)
    db.init_schema()

    console.print(
        f"[cyan]bench run[/cyan] benchmark={benchmark} "
        f"conditions={cond_names} problems={len(tasks)} reps={reps} "
        f"→ {len(variants) * len(tasks) * reps} runs"
    )

    async def _run() -> dict:
        registry = ProviderRegistry()
        async with registry:
            runner = ExperimentRunner(cfg, db, prompts_dir, registry=registry)
            return await runner.run_experiment(
                variants=list(variants.keys()),
                tasks=list(tasks.keys()),
                repetitions=reps,
                resume=True,
            )

    try:
        stats = asyncio.run(_run())
        table = Table(title="Generation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right")
        for key in ("total", "pending", "completed", "failed", "skipped"):
            table.add_row(key, str(stats.get(key, 0)))
        console.print(table)
    except Exception as e:
        console.print(f"[red]bench run failed: {e}[/red]")
        raise typer.Exit(code=1) from e
    finally:
        db.close()


@app.command("grade")
def grade(
    benchmark: str = typer.Argument(..., help="Official benchmark: humanevalplus | mbppplus"),
    run_id: str | None = typer.Option(None, "--run-id", help="Grade only this run"),
    force: bool = typer.Option(False, "--force", help="Re-grade already-graded designs"),
    database: Path = typer.Option(_DEFAULT_DB, "--database", "-d"),
    image: str = typer.Option("consortium-evalplus", "--image", help="EvalPlus harness image"),
) -> None:
    """Score final code artifacts with the official EvalPlus Docker harness.

    Writes objective pass@1 (HumanEval+/MBPP+ *plus* status) to ``code_results``.
    Requires the ``consortium-evalplus`` Docker image (see docker/evalplus.Dockerfile).
    """
    from consortium.evaluation.evalplus_scorer import EvalPlusScorer
    from consortium.storage.database import Database

    db = Database(database)
    db.init_schema()
    try:
        scorer = EvalPlusScorer(db, image=image)
        stats = scorer.score(benchmark, run_id=run_id, force=force)
        table = Table(title=f"EvalPlus Scoring — {benchmark}")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right")
        for key in ("total", "scored", "passed", "failed"):
            table.add_row(key, str(stats.get(key, 0)))
        if stats.get("scored"):
            rate = stats["passed"] / stats["scored"] * 100
            table.add_row("pass@1", f"{rate:.1f}%", style="green")
        console.print(table)
    except Exception as e:
        console.print(f"[red]Grading failed: {e}[/red]")
        raise typer.Exit(code=1) from e
    finally:
        db.close()


@app.command("report")
def report(
    benchmark: str = typer.Argument(..., help="Benchmark to summarize (e.g. humanevalplus)"),
    baseline: str = typer.Option(
        "v1a_sonnet", "--baseline", help="Variant id to compare consortium arms against (McNemar)"
    ),
    database: Path = typer.Option(_DEFAULT_DB, "--database", "-d"),
    markdown: Path | None = typer.Option(
        None, "--markdown", help="Also write a markdown results file to this path"
    ),
) -> None:
    """Print pass@1 / pass@k per condition and McNemar vs a baseline."""
    from consortium.analysis.code_stats import condition_results, mcnemar
    from consortium.storage.database import Database

    db = Database(database)
    db.init_schema()
    try:
        results = condition_results(db, benchmark)
        if not results:
            console.print(f"[yellow]No scored results for {benchmark}.[/yellow]")
            return

        if markdown is not None:
            lines = [f"## {benchmark} — pass@1\n",
                     "| condition | problems | pass@1 | 95% CI | pass@k |",
                     "|---|---:|---:|---|---:|"]
            for r in results:
                lines.append(
                    f"| {r.variant_id} | {r.n_problems} | {r.pass_at_1 * 100:.1f}% | "
                    f"[{r.ci_low * 100:.1f}, {r.ci_high * 100:.1f}] | {r.pass_at_k * 100:.1f}% |"
                )
            md_others = [r.variant_id for r in results if r.variant_id != baseline]
            if any(r.variant_id == baseline for r in results) and md_others:
                lines += [f"\n### McNemar vs {baseline}\n",
                          "| condition | both | neither | cond-only | base-only | p-value |",
                          "|---|---:|---:|---:|---:|---:|"]
                for vid in md_others:
                    m = mcnemar(db, benchmark, vid, baseline)
                    lines.append(
                        f"| {vid} | {m['both_pass']} | {m['neither_pass']} | "
                        f"{m['a_only']} | {m['b_only']} | {m['p_value']:.4f} |"
                    )
            markdown.parent.mkdir(parents=True, exist_ok=True)
            markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
            console.print(f"[green]wrote {markdown}[/green]")

        t = Table(title=f"pass@1 — {benchmark}")
        for col in ("condition", "problems", "samples", "pass@1", "95% CI", "pass@k"):
            t.add_column(col, justify="right" if col != "condition" else "left")
        for r in results:
            t.add_row(
                r.variant_id, str(r.n_problems), str(r.n_samples),
                f"{r.pass_at_1 * 100:.1f}%", f"[{r.ci_low * 100:.1f}, {r.ci_high * 100:.1f}]",
                f"{r.pass_at_k * 100:.1f}%",
            )
        console.print(t)

        others = [r.variant_id for r in results if r.variant_id != baseline]
        if any(r.variant_id == baseline for r in results) and others:
            mt = Table(title=f"McNemar vs {baseline}")
            for col in ("condition", "both", "neither", "cond-only", "base-only", "p-value"):
                mt.add_column(col, justify="right" if col != "condition" else "left")
            for vid in others:
                m = mcnemar(db, benchmark, vid, baseline)
                sig = "[green]" if m["p_value"] < 0.05 else ""
                mt.add_row(
                    vid, str(m["both_pass"]), str(m["neither_pass"]),
                    str(m["a_only"]), str(m["b_only"]),
                    f"{sig}{m['p_value']:.4f}",
                )
            console.print(mt)
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
