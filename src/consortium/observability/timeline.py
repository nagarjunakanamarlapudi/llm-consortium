"""Observability timeline — data models and builder for run trace analysis."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog

from consortium.storage.database import Database

logger = structlog.get_logger()


# ── Data Models ──────────────────────────────────────────────────────────────


@dataclass
class TraceEvent:
    trace_id: str
    run_id: str
    agent_id: str
    step: str  # generation, review, revision, merge, etc.
    round_num: int
    started_at: datetime
    ended_at: datetime
    duration_ms: float
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    cost_usd: float
    model: str
    prompt_hash: str
    agent_role: str = ""
    # Optional large fields — load on demand
    prompt_text: str | None = None
    response_text: str | None = None


@dataclass
class SwimLane:
    agent_id: str
    role: str
    events: list[TraceEvent] = field(default_factory=list)


@dataclass
class RunTimeline:
    run_id: str
    variant_id: str
    task_id: str
    total_duration_ms: float
    total_cost_usd: float
    total_tokens: int
    swim_lanes: list[SwimLane] = field(default_factory=list)
    parallel_groups: list[list[str]] = field(default_factory=list)


@dataclass
class VariantSummary:
    variant_id: str
    task_id: str | None
    run_count: int
    mean_duration_ms: float
    mean_cost_usd: float
    mean_tokens: int
    cost_breakdown_by_step: dict[str, float] = field(default_factory=dict)
    token_breakdown_by_step: dict[str, float] = field(default_factory=dict)


@dataclass
class ComparisonView:
    task_id: str
    variants: list[VariantSummary] = field(default_factory=list)


# ── Timeline Builder ─────────────────────────────────────────────────────────


def _parse_dt(s: str | None) -> datetime | None:
    """Parse an ISO‑format timestamp string from the DB."""
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _duration_ms(start: datetime | None, end: datetime | None) -> float:
    if start and end:
        return (end - start).total_seconds() * 1000
    return 0.0


class TimelineBuilder:
    """Builds timeline views from the traces table."""

    def __init__(self, database: Database) -> None:
        self.db = database

    # ── Public API ────────────────────────────────────────────────────────

    def build_run_timeline(self, run_id: str) -> RunTimeline:
        """Query traces for a single run, group by agent, detect parallelism."""
        conn = self.db.conn
        rows = conn.execute(
            """SELECT trace_id, run_id, agent_id, agent_role, step, round,
                      started_at, ended_at, input_tokens, output_tokens,
                      COALESCE(cached_input_tokens, 0) AS cached_input_tokens,
                      cost_usd, api_model, system_prompt_hash,
                      prompt_text, response_text
               FROM traces
               WHERE run_id = ?
               ORDER BY started_at""",
            (run_id,),
        ).fetchall()

        # Get run metadata
        run_row = conn.execute(
            "SELECT variant_id, task_id FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()

        variant_id = run_row["variant_id"] if run_row else ""
        task_id = run_row["task_id"] if run_row else ""

        events: list[TraceEvent] = []
        for row in rows:
            started = _parse_dt(row["started_at"])
            ended = _parse_dt(row["ended_at"])
            ev = TraceEvent(
                trace_id=row["trace_id"],
                run_id=row["run_id"],
                agent_id=row["agent_id"],
                agent_role=row["agent_role"],
                step=row["step"],
                round_num=row["round"],
                started_at=started or datetime.min,
                ended_at=ended or datetime.min,
                duration_ms=_duration_ms(started, ended),
                input_tokens=row["input_tokens"],
                output_tokens=row["output_tokens"],
                cached_input_tokens=row["cached_input_tokens"],
                cost_usd=row["cost_usd"],
                model=row["api_model"],
                prompt_hash=row["system_prompt_hash"] or "",
                prompt_text=row["prompt_text"],
                response_text=row["response_text"],
            )
            events.append(ev)

        # Group into swim lanes by agent_id
        lanes_map: dict[str, SwimLane] = {}
        for ev in events:
            if ev.agent_id not in lanes_map:
                lanes_map[ev.agent_id] = SwimLane(
                    agent_id=ev.agent_id,
                    role=ev.agent_role,
                )
            lanes_map[ev.agent_id].events.append(ev)

        # Detect parallelism — overlapping [started_at, ended_at] intervals
        parallel_groups = self._detect_parallelism(events)

        total_tokens = sum(e.input_tokens + e.output_tokens for e in events)
        total_cost = sum(e.cost_usd for e in events)
        if events:
            first_start = min(e.started_at for e in events)
            last_end = max(e.ended_at for e in events)
            total_duration = _duration_ms(first_start, last_end)
        else:
            total_duration = 0.0

        return RunTimeline(
            run_id=run_id,
            variant_id=variant_id,
            task_id=task_id,
            total_duration_ms=total_duration,
            total_cost_usd=total_cost,
            total_tokens=total_tokens,
            swim_lanes=list(lanes_map.values()),
            parallel_groups=parallel_groups,
        )

    def build_variant_summary(self, variant_id: str, task_id: str | None = None) -> VariantSummary:
        """Aggregate timing/cost stats across repetitions for a variant."""
        conn = self.db.conn

        if task_id:
            run_rows = conn.execute(
                """SELECT run_id, duration_seconds, total_cost_usd,
                          total_input_tokens, total_output_tokens
                   FROM runs
                   WHERE variant_id = ? AND task_id = ? AND status = 'completed'""",
                (variant_id, task_id),
            ).fetchall()
        else:
            run_rows = conn.execute(
                """SELECT run_id, duration_seconds, total_cost_usd,
                          total_input_tokens, total_output_tokens
                   FROM runs
                   WHERE variant_id = ? AND status = 'completed'""",
                (variant_id,),
            ).fetchall()

        if not run_rows:
            return VariantSummary(
                variant_id=variant_id,
                task_id=task_id,
                run_count=0,
                mean_duration_ms=0.0,
                mean_cost_usd=0.0,
                mean_tokens=0,
            )

        durations = [(r["duration_seconds"] or 0) * 1000 for r in run_rows]
        costs = [r["total_cost_usd"] or 0 for r in run_rows]
        tokens = [
            (r["total_input_tokens"] or 0) + (r["total_output_tokens"] or 0) for r in run_rows
        ]

        # Cost and token breakdown by step
        run_ids = [r["run_id"] for r in run_rows]
        placeholders = ",".join("?" * len(run_ids))
        step_rows = conn.execute(
            f"""SELECT step, SUM(cost_usd) as total_cost,
                       SUM(input_tokens + output_tokens) as total_tokens
                FROM traces
                WHERE run_id IN ({placeholders})
                GROUP BY step""",
            run_ids,
        ).fetchall()

        n = len(run_rows)
        cost_by_step = {r["step"]: r["total_cost"] / n for r in step_rows}
        token_by_step = {r["step"]: r["total_tokens"] / n for r in step_rows}

        return VariantSummary(
            variant_id=variant_id,
            task_id=task_id,
            run_count=len(run_rows),
            mean_duration_ms=statistics.mean(durations) if durations else 0.0,
            mean_cost_usd=statistics.mean(costs) if costs else 0.0,
            mean_tokens=int(statistics.mean(tokens)) if tokens else 0,
            cost_breakdown_by_step=cost_by_step,
            token_breakdown_by_step=token_by_step,
        )

    def build_comparison(self, variant_ids: list[str], task_id: str) -> ComparisonView:
        """Build side-by-side comparison of variants on a task."""
        summaries = [self.build_variant_summary(vid, task_id) for vid in variant_ids]
        return ComparisonView(task_id=task_id, variants=summaries)

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _detect_parallelism(events: list[TraceEvent]) -> list[list[str]]:
        """Detect groups of trace events with overlapping time intervals."""
        if len(events) < 2:
            return []

        groups: list[list[str]] = []
        sorted_events = sorted(events, key=lambda e: e.started_at)

        for i, ev_a in enumerate(sorted_events):
            group = [ev_a.trace_id]
            for ev_b in sorted_events[i + 1 :]:
                # Overlap: a starts before b ends AND b starts before a ends
                if ev_a.started_at < ev_b.ended_at and ev_b.started_at < ev_a.ended_at:
                    group.append(ev_b.trace_id)
            if len(group) > 1:
                # Deduplicate — only add if this exact group doesn't exist
                group_set = frozenset(group)
                if not any(frozenset(g) == group_set for g in groups):
                    groups.append(group)

        return groups
