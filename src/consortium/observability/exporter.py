"""Export timeline data to various formats."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from consortium.observability.timeline import (
        ComparisonView,
        RunTimeline,
        VariantSummary,
    )

logger = structlog.get_logger()


# ── Rich Console Export ──────────────────────────────────────────────────────


def export_rich(timeline: RunTimeline) -> None:
    """Print a Rich console table showing the run timeline."""
    console = Console()

    console.print(f"\n[bold]Run Timeline: {timeline.run_id}[/bold]")
    console.print(
        f"Variant: [cyan]{timeline.variant_id}[/cyan]  "
        f"Task: [cyan]{timeline.task_id}[/cyan]  "
        f"Duration: [yellow]{timeline.total_duration_ms:.0f}ms[/yellow]  "
        f"Cost: [green]${timeline.total_cost_usd:.4f}[/green]  "
        f"Tokens: {timeline.total_tokens:,}"
    )

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Agent", style="cyan")
    table.add_column("Role")
    table.add_column("Step")
    table.add_column("Round", justify="right")
    table.add_column("Started", style="dim")
    table.add_column("Duration (ms)", justify="right")
    table.add_column("In", justify="right")
    table.add_column("Cached", justify="right")
    table.add_column("Out", justify="right")
    table.add_column("Cost ($)", justify="right")
    table.add_column("Model")

    # Collect all events across swim lanes and sort chronologically
    all_events = []
    for lane in timeline.swim_lanes:
        all_events.extend(lane.events)
    all_events.sort(key=lambda ev: ev.started_at)

    for ev in all_events:
        started_str = ev.started_at.strftime("%H:%M:%S")
        cached_str = f"{ev.cached_input_tokens:,}" if ev.cached_input_tokens else "-"
        table.add_row(
            ev.agent_id,
            ev.agent_role,
            ev.step,
            str(ev.round_num),
            started_str,
            f"{ev.duration_ms:.0f}",
            f"{ev.input_tokens:,}",
            cached_str,
            f"{ev.output_tokens:,}",
            f"${ev.cost_usd:.4f}",
            ev.model,
        )

    console.print(table)

    if timeline.parallel_groups:
        console.print(
            f"\n[dim]Detected {len(timeline.parallel_groups)} parallel execution group(s)[/dim]"
        )


def export_rich_summary(summary: VariantSummary) -> None:
    """Print a Rich summary table for a variant."""
    console = Console()

    console.print(f"\n[bold]Variant Summary: {summary.variant_id}[/bold]")
    if summary.task_id:
        console.print(f"Task: [cyan]{summary.task_id}[/cyan]")
    console.print(
        f"Runs: {summary.run_count}  "
        f"Mean Duration: [yellow]{summary.mean_duration_ms:.0f}ms[/yellow]  "
        f"Mean Cost: [green]${summary.mean_cost_usd:.4f}[/green]  "
        f"Mean Tokens: {summary.mean_tokens:,}"
    )

    if summary.cost_breakdown_by_step:
        table = Table(title="Cost Breakdown by Step", show_header=True)
        table.add_column("Step", style="cyan")
        table.add_column("Mean Cost ($)", justify="right")
        table.add_column("Mean Tokens", justify="right")

        for step in sorted(summary.cost_breakdown_by_step):
            cost = summary.cost_breakdown_by_step[step]
            tokens = summary.token_breakdown_by_step.get(step, 0)
            table.add_row(step, f"${cost:.4f}", f"{tokens:,.0f}")

        console.print(table)


def export_rich_comparison(comparison: ComparisonView) -> None:
    """Print a Rich comparison table across variants."""
    console = Console()

    console.print(f"\n[bold]Variant Comparison — Task: {comparison.task_id}[/bold]")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Variant", style="cyan")
    table.add_column("Runs", justify="right")
    table.add_column("Mean Duration (ms)", justify="right")
    table.add_column("Mean Cost ($)", justify="right")
    table.add_column("Mean Tokens", justify="right")

    for s in comparison.variants:
        table.add_row(
            s.variant_id,
            str(s.run_count),
            f"{s.mean_duration_ms:.0f}",
            f"${s.mean_cost_usd:.4f}",
            f"{s.mean_tokens:,}",
        )

    console.print(table)


# ── JSON Export ──────────────────────────────────────────────────────────────


def _serialize(obj: object) -> object:
    """Custom serializer for dataclasses with datetime fields."""
    from datetime import datetime as dt

    if isinstance(obj, dt):
        return obj.isoformat()
    return str(obj)


def export_json(timeline: RunTimeline, path: Path) -> None:
    """Export timeline to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(timeline)
    path.write_text(json.dumps(data, indent=2, default=_serialize))
    logger.info("timeline.export.json", path=str(path))


# ── Chrome Trace Export ──────────────────────────────────────────────────────


def export_chrome_trace(timeline: RunTimeline, path: Path) -> None:
    """Export timeline in Chrome Trace Event format for chrome://tracing.

    Each agent gets its own thread (tid). Each LLM call is an 'X' (complete) event.
    """
    trace_events: list[dict] = []

    # Assign stable tid per agent
    agent_ids = sorted({lane.agent_id for lane in timeline.swim_lanes})
    agent_tid = {aid: idx + 1 for idx, aid in enumerate(agent_ids)}

    for lane in timeline.swim_lanes:
        tid = agent_tid.get(lane.agent_id, 0)
        for ev in lane.events:
            ts_us = ev.started_at.timestamp() * 1_000_000  # epoch microseconds
            dur_us = ev.duration_ms * 1000
            trace_events.append(
                {
                    "name": ev.step,
                    "cat": "llm",
                    "ph": "X",
                    "ts": ts_us,
                    "dur": dur_us,
                    "pid": 1,
                    "tid": tid,
                    "args": {
                        "agent_id": ev.agent_id,
                        "role": ev.agent_role,
                        "model": ev.model,
                        "tokens": ev.input_tokens + ev.output_tokens,
                        "cost": ev.cost_usd,
                        "round": ev.round_num,
                    },
                }
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace_events, indent=2))
    logger.info("timeline.export.chrome_trace", path=str(path), events=len(trace_events))


# ── CSV Export ───────────────────────────────────────────────────────────────


def export_csv(timeline: RunTimeline, path: Path) -> None:
    """Export timeline events to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "trace_id",
        "run_id",
        "agent_id",
        "agent_role",
        "step",
        "round_num",
        "started_at",
        "ended_at",
        "duration_ms",
        "input_tokens",
        "output_tokens",
        "cost_usd",
        "model",
    ]

    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()

    for lane in timeline.swim_lanes:
        for ev in lane.events:
            writer.writerow(
                {
                    "trace_id": ev.trace_id,
                    "run_id": ev.run_id,
                    "agent_id": ev.agent_id,
                    "agent_role": ev.agent_role,
                    "step": ev.step,
                    "round_num": ev.round_num,
                    "started_at": ev.started_at.isoformat(),
                    "ended_at": ev.ended_at.isoformat(),
                    "duration_ms": f"{ev.duration_ms:.1f}",
                    "input_tokens": ev.input_tokens,
                    "output_tokens": ev.output_tokens,
                    "cost_usd": f"{ev.cost_usd:.6f}",
                    "model": ev.model,
                }
            )

    path.write_text(buf.getvalue())
    logger.info("timeline.export.csv", path=str(path))
