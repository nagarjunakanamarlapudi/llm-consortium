"""CLI commands for analysis and visualization."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True)
console = Console()

_DEFAULT_DB = Path("data/consortium.db")
_DEFAULT_CONFIGS = Path("configs")
_DEFAULT_OUTPUT = Path("data/exports")


def _load(config_dir: Path, database: Path):
    from consortium.config.loader import load_full_config
    from consortium.storage.database import Database

    config = load_full_config(config_dir)
    db = Database(database)
    return config, db


@app.command()
def doctor(
    config_dir: Path = typer.Option(_DEFAULT_CONFIGS, "--configs", "-c"),
    database: Path = typer.Option(_DEFAULT_DB, "--database", "-d"),
) -> None:
    """Check data completeness for the experiment matrix."""
    from consortium.analysis.loader import check_completeness

    config, db = _load(config_dir, database)

    try:
        variant_ids = sorted(config.variants.keys())
        task_ids = sorted(config.tasks.keys())
        reps = config.experiment.repetitions

        report = check_completeness(db, variant_ids, task_ids, reps)

        console.print("\n[bold]Experiment Completeness Report[/bold]\n")
        console.print(
            f"Expected:  {report.expected_runs} runs ({report.n_variants} variants x {report.n_tasks} tasks x {report.n_reps} reps)"
        )
        console.print(f"Completed: {report.actual_runs} runs ({report.run_coverage:.1%})")
        console.print(f"Evaluated: {report.evaluated_designs} designs ({report.eval_coverage:.1%})")
        console.print(f"Coherence: {report.coherence_checked} designs ({report.coherence_coverage:.1%})")

        if report.missing_runs:
            console.print(f"\n[yellow]Missing runs ({len(report.missing_runs)}):[/yellow]")
            for v, t, r in report.missing_runs[:20]:
                console.print(f"  {v} x {t} x rep{r}")
            if len(report.missing_runs) > 20:
                console.print(f"  ... and {len(report.missing_runs) - 20} more")

        if report.missing_evaluations:
            console.print(
                f"\n[yellow]Missing evaluations ({len(report.missing_evaluations)}):[/yellow]"
            )
            for did in report.missing_evaluations[:10]:
                console.print(f"  {did}")

        if report.missing_coherence:
            console.print(
                f"\n[yellow]Missing coherence checks ({len(report.missing_coherence)}):[/yellow]"
            )
            for did in report.missing_coherence[:10]:
                console.print(f"  {did}")
            if len(report.missing_coherence) > 10:
                console.print(f"  ... and {len(report.missing_coherence) - 10} more")

        if report.is_complete:
            console.print("\n[green]✓ All data complete![/green]")
        else:
            console.print("\n[yellow]⚠ Incomplete data — see recommendations above[/yellow]")
    finally:
        db.close()


@app.command()
def ranking(
    config_dir: Path = typer.Option(_DEFAULT_CONFIGS, "--configs", "-c"),
    database: Path = typer.Option(_DEFAULT_DB, "--database", "-d"),
    output_dir: Path = typer.Option(_DEFAULT_OUTPUT, "--output-dir", "-o"),
) -> None:
    """Rank variants using Friedman test + Nemenyi post-hoc."""
    from consortium.analysis.loader import load_scores_dataframe
    from consortium.analysis.statistics import rank_variants_overall

    _, db = _load(config_dir, database)

    try:
        scores_df = load_scores_dataframe(db)
        result = rank_variants_overall(scores_df)

        if result is None:
            console.print("[yellow]Insufficient data for ranking[/yellow]")
            return

        table = Table(title="Variant Ranking (Friedman Test)")
        table.add_column("Rank", justify="right")
        table.add_column("Variant", style="cyan")
        table.add_column("Mean Rank", justify="right")
        table.add_column("Mean Score", justify="right")

        for i, (vid, mr, ms) in enumerate(
            zip(result.variant_ids, result.mean_ranks, result.mean_scores), 1
        ):
            table.add_row(str(i), vid, f"{mr:.2f}", f"{ms:.3f}")

        console.print(table)
        console.print(
            f"\nFriedman χ² = {result.statistic:.2f}, p = {result.p_value:.4f}"
            f" {'[green](significant)[/green]' if result.significant else '[yellow](not significant)[/yellow]'}"
        )

        if result.posthoc:
            console.print("\n[bold]Post-hoc pairwise p-values (Bonferroni adjusted):[/bold]")
            for pair, p in sorted(result.posthoc.items()):
                sig = "✓" if p < 0.05 else " "
                console.print(f"  {sig} {pair}: p={p:.4f}")

        # Export CSV
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "tables" / "variant_ranking_overall.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        import pandas as pd

        pd.DataFrame(
            {
                "variant_id": result.variant_ids,
                "mean_rank": result.mean_ranks,
                "mean_score": result.mean_scores,
            }
        ).to_csv(csv_path, index=False)
        console.print(f"\n[dim]Saved to {csv_path}[/dim]")
    finally:
        db.close()


@app.command()
def baseline(
    config_dir: Path = typer.Option(_DEFAULT_CONFIGS, "--configs", "-c"),
    database: Path = typer.Option(_DEFAULT_DB, "--database", "-d"),
    baseline_variant: str = typer.Option("v1", "--baseline", help="Baseline variant ID"),
    output_dir: Path = typer.Option(_DEFAULT_OUTPUT, "--output-dir", "-o"),
) -> None:
    """Compare all variants to a baseline using Wilcoxon tests."""
    from consortium.analysis.loader import load_scores_dataframe
    from consortium.analysis.statistics import compare_to_baseline

    _, db = _load(config_dir, database)

    try:
        scores_df = load_scores_dataframe(db)
        results = compare_to_baseline(scores_df, baseline_variant)

        if not results:
            console.print("[yellow]Insufficient data for baseline comparison[/yellow]")
            return

        table = Table(title=f"Baseline Comparison (vs {baseline_variant})")
        table.add_column("Variant", style="cyan")
        table.add_column("p-value", justify="right")
        table.add_column("p-adjusted", justify="right")
        table.add_column("Effect Size", justify="right")
        table.add_column("Significant?")

        for r in results:
            sig = "[green]✓[/green]" if r.significant else "[dim]✗[/dim]"
            table.add_row(
                r.variant_b,
                f"{r.p_value:.4f}",
                f"{r.p_adjusted:.4f}",
                f"{r.effect_size:.3f}",
                sig,
            )

        console.print(table)
    finally:
        db.close()


@app.command()
def pairwise(
    config_dir: Path = typer.Option(_DEFAULT_CONFIGS, "--configs", "-c"),
    database: Path = typer.Option(_DEFAULT_DB, "--database", "-d"),
    output_dir: Path = typer.Option(_DEFAULT_OUTPUT, "--output-dir", "-o"),
) -> None:
    """All-pairs Wilcoxon signed-rank comparisons."""
    from consortium.analysis.loader import load_scores_dataframe
    from consortium.analysis.statistics import pairwise_comparisons

    _, db = _load(config_dir, database)

    try:
        scores_df = load_scores_dataframe(db)
        results = pairwise_comparisons(scores_df)

        if not results:
            console.print("[yellow]Insufficient data for pairwise comparisons[/yellow]")
            return

        table = Table(title="Pairwise Comparisons (Wilcoxon + Bonferroni)")
        table.add_column("Pair", style="cyan")
        table.add_column("p-adjusted", justify="right")
        table.add_column("Effect", justify="right")
        table.add_column("Sig?")

        for r in results:
            sig = "✓" if r.significant else ""
            table.add_row(
                f"{r.variant_a} vs {r.variant_b}",
                f"{r.p_adjusted:.4f}",
                f"{r.effect_size:.3f}",
                sig,
            )

        console.print(table)

        # Export
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "tables" / "pairwise_significance.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        import pandas as pd

        pd.DataFrame(
            [
                {
                    "variant_a": r.variant_a,
                    "variant_b": r.variant_b,
                    "p_value": r.p_value,
                    "p_adjusted": r.p_adjusted,
                    "effect_size": r.effect_size,
                    "significant": r.significant,
                }
                for r in results
            ]
        ).to_csv(csv_path, index=False)
        console.print(f"\n[dim]Saved to {csv_path}[/dim]")
    finally:
        db.close()


@app.command()
def axes(
    config_dir: Path = typer.Option(_DEFAULT_CONFIGS, "--configs", "-c"),
    database: Path = typer.Option(_DEFAULT_DB, "--database", "-d"),
) -> None:
    """2x2x2 axis effect analysis (authority, roles, dynamics)."""
    from consortium.analysis.loader import load_scores_dataframe
    from consortium.analysis.statistics import analyze_axis_effect

    _, db = _load(config_dir, database)

    try:
        scores_df = load_scores_dataframe(db)

        table = Table(title="Axis Effect Analysis (Mann-Whitney U)")
        table.add_column("Axis", style="cyan")
        table.add_column("Level A")
        table.add_column("Level B")
        table.add_column("Mean A", justify="right")
        table.add_column("Mean B", justify="right")
        table.add_column("p-value", justify="right")
        table.add_column("Effect", justify="right")
        table.add_column("Sig?")

        for axis in ("authority", "roles", "dynamics"):
            result = analyze_axis_effect(scores_df, axis)
            if result:
                sig = "✓" if result.significant else ""
                table.add_row(
                    result.axis_name,
                    result.level_a,
                    result.level_b,
                    f"{result.mean_a:.3f}",
                    f"{result.mean_b:.3f}",
                    f"{result.p_value:.4f}",
                    f"{result.effect_size:.3f}",
                    sig,
                )

        console.print(table)
    finally:
        db.close()


@app.command()
def pareto(
    config_dir: Path = typer.Option(_DEFAULT_CONFIGS, "--configs", "-c"),
    database: Path = typer.Option(_DEFAULT_DB, "--database", "-d"),
    output_dir: Path = typer.Option(_DEFAULT_OUTPUT, "--output-dir", "-o"),
) -> None:
    """Pareto frontier: quality vs. cost."""
    from consortium.analysis.loader import load_costs_dataframe, load_scores_dataframe
    from consortium.analysis.pareto import plot_pareto
    from consortium.analysis.statistics import token_efficiency

    _, db = _load(config_dir, database)

    try:
        scores_df = load_scores_dataframe(db)
        costs_df = load_costs_dataframe(db)
        efficiency = token_efficiency(scores_df, costs_df)

        if not efficiency:
            console.print("[yellow]Insufficient data for Pareto analysis[/yellow]")
            return

        import pandas as pd

        eff_df = pd.DataFrame(
            [
                {
                    "variant_id": e.variant_id,
                    "mean_score": e.mean_score,
                    "mean_cost": e.mean_cost,
                }
                for e in efficiency
            ]
        )

        fig_dir = output_dir / "figures"
        plot_pareto(eff_df, fig_dir)
        console.print(f"[green]✓ Pareto frontier saved to {fig_dir}[/green]")
    finally:
        db.close()


@app.command()
def heatmap(
    config_dir: Path = typer.Option(_DEFAULT_CONFIGS, "--configs", "-c"),
    database: Path = typer.Option(_DEFAULT_DB, "--database", "-d"),
    output_dir: Path = typer.Option(_DEFAULT_OUTPUT, "--output-dir", "-o"),
) -> None:
    """Generate all heatmap visualizations."""
    from consortium.analysis.heatmaps import (
        coherence_variant_heatmap,
        complexity_variant_heatmap,
        task_variant_heatmap,
        variant_dimension_heatmap,
    )
    from consortium.analysis.loader import load_coherence_dataframe, load_scores_dataframe

    _, db = _load(config_dir, database)

    try:
        scores_df = load_scores_dataframe(db)
        if scores_df.empty:
            console.print("[yellow]No scores data available[/yellow]")
            return

        fig_dir = output_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        variant_dimension_heatmap(scores_df, fig_dir / "heatmap_variant_dimension.png")
        task_variant_heatmap(scores_df, fig_dir / "heatmap_task_variant.png")

        if "complexity" in scores_df.columns:
            complexity_variant_heatmap(scores_df, fig_dir / "heatmap_complexity_variant.png")

        coherence_df = load_coherence_dataframe(db)
        if not coherence_df.empty:
            coherence_variant_heatmap(coherence_df, fig_dir / "heatmap_coherence.png")

        console.print(f"[green]✓ Heatmaps saved to {fig_dir}[/green]")
    finally:
        db.close()


@app.command()
def efficiency(
    config_dir: Path = typer.Option(_DEFAULT_CONFIGS, "--configs", "-c"),
    database: Path = typer.Option(_DEFAULT_DB, "--database", "-d"),
    output_dir: Path = typer.Option(_DEFAULT_OUTPUT, "--output-dir", "-o"),
) -> None:
    """Token/cost efficiency analysis."""
    from consortium.analysis.loader import load_costs_dataframe, load_scores_dataframe
    from consortium.analysis.statistics import token_efficiency

    _, db = _load(config_dir, database)

    try:
        scores_df = load_scores_dataframe(db)
        costs_df = load_costs_dataframe(db)
        results = token_efficiency(scores_df, costs_df)

        if not results:
            console.print("[yellow]Insufficient data[/yellow]")
            return

        table = Table(title="Token Efficiency")
        table.add_column("Variant", style="cyan")
        table.add_column("Mean Score", justify="right")
        table.add_column("Mean Cost", justify="right")
        table.add_column("Mean Tokens", justify="right")
        table.add_column("Quality/$", justify="right")
        table.add_column("Quality/1K tok", justify="right")

        for r in results:
            table.add_row(
                r.variant_id,
                f"{r.mean_score:.3f}",
                f"${r.mean_cost:.4f}",
                f"{r.mean_tokens:,}",
                f"{r.quality_per_dollar:.1f}",
                f"{r.quality_per_1k_tokens:.4f}",
            )

        console.print(table)
    finally:
        db.close()


@app.command()
def framework(
    config_dir: Path = typer.Option(_DEFAULT_CONFIGS, "--configs", "-c"),
    database: Path = typer.Option(_DEFAULT_DB, "--database", "-d"),
    output_dir: Path = typer.Option(_DEFAULT_OUTPUT, "--output-dir", "-o"),
) -> None:
    """Generate the decision framework (§10.2)."""
    from consortium.analysis.framework import generate_decision_framework
    from consortium.analysis.loader import (
        load_coherence_dataframe,
        load_costs_dataframe,
        load_scores_dataframe,
    )

    _, db = _load(config_dir, database)

    try:
        scores_df = load_scores_dataframe(db)
        costs_df = load_costs_dataframe(db)
        coherence_df = load_coherence_dataframe(db)
        fw = generate_decision_framework(
            scores_df, costs_df, output_dir,
            coherence_df=coherence_df if not coherence_df.empty else None,
        )

        console.print(f"\n[green]✓ Decision framework generated[/green]")
        console.print(f"  Rows: {len(fw.rows)}")
        console.print(f"  Pareto variants: {', '.join(fw.pareto_variants)}")
        console.print(f"  Runs analyzed: {fw.n_runs_analyzed}")

        if fw.warnings:
            for w in fw.warnings:
                console.print(f"  [yellow]⚠ {w}[/yellow]")
    finally:
        db.close()


@app.command()
def predictions(
    config_dir: Path = typer.Option(_DEFAULT_CONFIGS, "--configs", "-c"),
    database: Path = typer.Option(_DEFAULT_DB, "--database", "-d"),
    output_dir: Path = typer.Option(_DEFAULT_OUTPUT, "--output-dir", "-o"),
) -> None:
    """Validate thesis predictions P1-P8 (§10.1)."""
    from consortium.analysis.framework import (
        export_prediction_report,
        validate_predictions,
    )
    from consortium.analysis.loader import (
        load_coherence_dataframe,
        load_costs_dataframe,
        load_scores_dataframe,
    )

    _, db = _load(config_dir, database)

    try:
        scores_df = load_scores_dataframe(db)
        costs_df = load_costs_dataframe(db)
        coherence_df = load_coherence_dataframe(db)
        results = validate_predictions(
            scores_df, costs_df,
            coherence_df=coherence_df if not coherence_df.empty else None,
        )

        table = Table(title="Prediction Validation (§10.1)")
        table.add_column("ID", style="cyan")
        table.add_column("Prediction")
        table.add_column("Supported?")
        table.add_column("p-value", justify="right")
        table.add_column("Evidence")

        for r in results:
            check = "[green]✓[/green]" if r.supported else "[red]✗[/red]"
            p_str = f"{r.p_value:.4f}" if r.p_value is not None else "N/A"
            table.add_row(r.prediction_id, r.description, check, p_str, r.evidence[:60])

        console.print(table)

        export_prediction_report(results, output_dir)
        console.print(f"\n[dim]Report saved to {output_dir}/reports/prediction_validation.md[/dim]")
    finally:
        db.close()


@app.command()
def coherence(
    config_dir: Path = typer.Option(_DEFAULT_CONFIGS, "--configs", "-c"),
    database: Path = typer.Option(_DEFAULT_DB, "--database", "-d"),
    output_dir: Path = typer.Option(_DEFAULT_OUTPUT, "--output-dir", "-o"),
) -> None:
    """Analyze coherence check results across variants."""
    from consortium.analysis.heatmaps import coherence_variant_heatmap
    from consortium.analysis.loader import (
        load_coherence_dataframe,
        load_scores_dataframe,
    )
    from consortium.analysis.statistics import (
        coherence_quality_correlation,
        compare_coherence_to_baseline,
        rank_variants_by_coherence,
    )

    _, db = _load(config_dir, database)

    try:
        coherence_df = load_coherence_dataframe(db)
        if coherence_df.empty:
            console.print("[yellow]No coherence data available. Run 'consortium run coherence' first.[/yellow]")
            return

        # Per-variant summary
        summary_table = Table(title="Coherence Summary by Variant")
        summary_table.add_column("Variant", style="cyan")
        summary_table.add_column("Designs", justify="right")
        summary_table.add_column("Pairs Checked", justify="right")
        summary_table.add_column("Contradictions", justify="right")
        summary_table.add_column("Coherence Rate", justify="right")

        for vid, group in coherence_df.groupby("variant_id"):
            n_designs = len(group)
            total_pairs = int(group["pairs_checked"].sum())
            total_contradictions = int(group["contradictions"].sum())
            mean_rate = float(group["coherence_rate"].mean())
            style = "red" if mean_rate < 0.8 else ""
            summary_table.add_row(
                str(vid),
                str(n_designs),
                str(total_pairs),
                str(total_contradictions),
                f"{mean_rate:.1%}",
                style=style,
            )

        console.print(summary_table)

        # Friedman ranking
        ranking_result = rank_variants_by_coherence(coherence_df)
        if ranking_result:
            console.print(
                f"\nFriedman χ² = {ranking_result.statistic:.2f}, p = {ranking_result.p_value:.4f}"
                f" {'[green](significant)[/green]' if ranking_result.significant else '[yellow](not significant)[/yellow]'}"
            )

        # Baseline comparison
        bl_results = compare_coherence_to_baseline(coherence_df)
        if bl_results:
            bl_table = Table(title="Coherence vs Baseline (v1)")
            bl_table.add_column("Variant", style="cyan")
            bl_table.add_column("p-adjusted", justify="right")
            bl_table.add_column("Effect Size", justify="right")
            bl_table.add_column("Sig?")

            for r in bl_results:
                sig = "[green]✓[/green]" if r.significant else "[dim]✗[/dim]"
                bl_table.add_row(
                    r.variant_b,
                    f"{r.p_adjusted:.4f}",
                    f"{r.effect_size:.3f}",
                    sig,
                )
            console.print(bl_table)

        # Quality-coherence correlation
        scores_df = load_scores_dataframe(db)
        if not scores_df.empty:
            corr = coherence_quality_correlation(scores_df, coherence_df)
            if corr:
                sig_str = "[green](significant)[/green]" if corr.significant else "[yellow](not significant)[/yellow]"
                console.print(
                    f"\nQuality–Coherence correlation: Spearman r={corr.spearman_r:.3f}, "
                    f"p={corr.p_value:.4f} {sig_str} (n={corr.n_designs})"
                )

        # Heatmap
        fig_dir = output_dir / "figures"
        coherence_variant_heatmap(coherence_df, fig_dir / "heatmap_coherence.png")

        # Export CSV
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "tables" / "coherence_summary.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        coherence_df.to_csv(csv_path, index=False)
        console.print(f"\n[dim]Saved to {csv_path}[/dim]")
    finally:
        db.close()


@app.command(name="all")
def analyze_all(
    config_dir: Path = typer.Option(_DEFAULT_CONFIGS, "--configs", "-c"),
    database: Path = typer.Option(_DEFAULT_DB, "--database", "-d"),
    output_dir: Path = typer.Option(_DEFAULT_OUTPUT, "--output-dir", "-o"),
) -> None:
    """Run all analysis commands and generate full export."""
    from consortium.analysis.framework import (
        export_prediction_report,
        generate_decision_framework,
        validate_predictions,
    )
    from consortium.analysis.heatmaps import (
        coherence_variant_heatmap,
        complexity_variant_heatmap,
        task_variant_heatmap,
        variant_dimension_heatmap,
    )
    from consortium.analysis.loader import (
        check_completeness,
        load_coherence_dataframe,
        load_costs_dataframe,
        load_scores_dataframe,
    )
    from consortium.analysis.pareto import plot_pareto
    from consortium.analysis.statistics import (
        compare_to_baseline,
        pairwise_comparisons,
        rank_variants_by_coherence,
        rank_variants_overall,
        token_efficiency,
        variance_analysis,
    )

    config, db = _load(config_dir, database)

    try:
        console.print("[bold]Running full analysis suite...[/bold]\n")

        # 1. Doctor
        variant_ids = sorted(config.variants.keys())
        task_ids = sorted(config.tasks.keys())
        report = check_completeness(db, variant_ids, task_ids, config.experiment.repetitions)
        console.print(
            f"Completeness: {report.run_coverage:.1%} runs, {report.eval_coverage:.1%} evals"
        )

        # 2. Load data
        scores_df = load_scores_dataframe(db)
        costs_df = load_costs_dataframe(db)

        if scores_df.empty:
            console.print("[yellow]No scores available — skipping analysis[/yellow]")
            return

        # 3. Rankings
        result = rank_variants_overall(scores_df)
        if result:
            console.print(f"Friedman: χ²={result.statistic:.2f}, p={result.p_value:.4f}")

        # 4. Baseline
        bl_results = compare_to_baseline(scores_df)
        sig_count = sum(1 for r in bl_results if r.significant)
        console.print(f"Baseline comparisons: {sig_count}/{len(bl_results)} significant")

        # 5. Pairwise
        pw_results = pairwise_comparisons(scores_df)
        sig_pw = sum(1 for r in pw_results if r.significant)
        console.print(f"Pairwise: {sig_pw}/{len(pw_results)} significant pairs")

        # 6. Variance
        va_results = variance_analysis(scores_df)
        if va_results:
            console.print(f"Lowest CV: {va_results[0].variant_id} ({va_results[0].cv:.4f})")

        # 7. Efficiency
        eff_results = token_efficiency(scores_df, costs_df)

        # 8. Pareto
        if eff_results:
            import pandas as pd

            eff_df = pd.DataFrame(
                [
                    {
                        "variant_id": e.variant_id,
                        "mean_score": e.mean_score,
                        "mean_cost": e.mean_cost,
                    }
                    for e in eff_results
                ]
            )
            plot_pareto(eff_df, output_dir / "figures")

        # 9. Heatmaps
        fig_dir = output_dir / "figures"
        variant_dimension_heatmap(scores_df, fig_dir / "heatmap_variant_dimension.png")
        task_variant_heatmap(scores_df, fig_dir / "heatmap_task_variant.png")
        if "complexity" in scores_df.columns:
            complexity_variant_heatmap(scores_df, fig_dir / "heatmap_complexity_variant.png")

        # 10. Coherence
        coherence_df = load_coherence_dataframe(db)
        if not coherence_df.empty:
            coh_ranking = rank_variants_by_coherence(coherence_df)
            if coh_ranking:
                console.print(
                    f"Coherence Friedman: χ²={coh_ranking.statistic:.2f}, "
                    f"p={coh_ranking.p_value:.4f}"
                )
            coherence_variant_heatmap(coherence_df, fig_dir / "heatmap_coherence.png")
        else:
            console.print("Coherence: [dim]no data (run 'consortium run coherence')[/dim]")

        # 11. Framework
        fw = generate_decision_framework(
            scores_df, costs_df, output_dir,
            coherence_df=coherence_df if not coherence_df.empty else None,
        )
        console.print(f"Framework: {len(fw.rows)} rows, {len(fw.pareto_variants)} Pareto variants")

        # 12. Predictions
        pred_results = validate_predictions(
            scores_df, costs_df,
            coherence_df=coherence_df if not coherence_df.empty else None,
        )
        export_prediction_report(pred_results, output_dir)
        supported = sum(1 for r in pred_results if r.supported)
        console.print(f"Predictions: {supported}/{len(pred_results)} supported")

        console.print(f"\n[green]✓ Full analysis complete. Exports at {output_dir}[/green]")
    finally:
        db.close()
