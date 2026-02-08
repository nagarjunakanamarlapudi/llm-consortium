"""Load experiment data from the database into DataFrames for analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd
import structlog

if TYPE_CHECKING:
    from consortium.storage.database import Database

logger = structlog.get_logger()


@dataclass
class CompletenessReport:
    """Report on data completeness for the experiment matrix."""

    n_variants: int = 0
    n_tasks: int = 0
    n_reps: int = 0
    expected_runs: int = 0
    actual_runs: int = 0
    evaluated_designs: int = 0
    missing_runs: list[tuple[str, str, int]] = field(default_factory=list)
    missing_evaluations: list[str] = field(default_factory=list)
    coherence_checked: int = 0
    missing_coherence: list[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return (
            len(self.missing_runs) == 0
            and len(self.missing_evaluations) == 0
            and len(self.missing_coherence) == 0
        )

    @property
    def run_coverage(self) -> float:
        return self.actual_runs / self.expected_runs if self.expected_runs else 0.0

    @property
    def eval_coverage(self) -> float:
        return self.evaluated_designs / self.actual_runs if self.actual_runs else 0.0

    @property
    def coherence_coverage(self) -> float:
        return self.coherence_checked / self.evaluated_designs if self.evaluated_designs else 0.0


def load_scores_dataframe(db: Database) -> pd.DataFrame:
    """Load scores into a DataFrame: join scores_median <- designs <- runs.

    Returns a DataFrame with columns:
        variant_id, task_id, repetition, design_id, run_id,
        overall_median, dimension_medians (JSON string),
        krippendorff_alpha, disagreement_flags
    """
    query = """
        SELECT
            r.variant_id,
            r.task_id,
            r.repetition,
            d.design_id,
            d.run_id,
            sm.overall_median,
            sm.dimension_medians,
            sm.krippendorff_alpha,
            sm.disagreement_flags
        FROM scores_median sm
        JOIN designs d ON sm.design_id = d.design_id
        JOIN runs r ON d.run_id = r.run_id
        WHERE d.is_final = TRUE
          AND r.status = 'completed'
        ORDER BY r.variant_id, r.task_id, r.repetition
    """
    rows = db.conn.execute(query).fetchall()
    if not rows:
        logger.warning("loader.no_scores")
        return pd.DataFrame()

    df = pd.DataFrame([dict(row) for row in rows])
    logger.info("loader.scores_loaded", rows=len(df))
    return df


def load_coherence_dataframe(db: Database) -> pd.DataFrame:
    """Load coherence data: one row per design with coherence rate.

    Joins coherence_checks <- designs <- runs and aggregates per design.

    Returns a DataFrame with columns:
        variant_id, task_id, repetition, design_id, run_id,
        pairs_checked, contradictions, coherence_rate
    """
    query = """
        SELECT
            r.variant_id,
            r.task_id,
            r.repetition,
            d.design_id,
            d.run_id,
            COUNT(*) as pairs_checked,
            SUM(CASE WHEN cc.contradicts = 1 THEN 1 ELSE 0 END) as contradictions
        FROM coherence_checks cc
        JOIN designs d ON cc.design_id = d.design_id
        JOIN runs r ON d.run_id = r.run_id
        WHERE d.is_final = TRUE
          AND r.status = 'completed'
        GROUP BY d.design_id
        ORDER BY r.variant_id, r.task_id, r.repetition
    """
    rows = db.conn.execute(query).fetchall()
    if not rows:
        logger.warning("loader.no_coherence")
        return pd.DataFrame()

    df = pd.DataFrame([dict(row) for row in rows])
    df["coherence_rate"] = 1.0 - (df["contradictions"] / df["pairs_checked"])
    logger.info("loader.coherence_loaded", rows=len(df))
    return df


def load_costs_dataframe(db: Database) -> pd.DataFrame:
    """Load cost/token data from runs.

    Returns a DataFrame with columns:
        variant_id, task_id, repetition, run_id,
        total_cost_usd, total_input_tokens, total_output_tokens,
        duration_seconds
    """
    query = """
        SELECT
            variant_id,
            task_id,
            repetition,
            run_id,
            total_cost_usd,
            total_input_tokens,
            total_output_tokens,
            duration_seconds
        FROM runs
        WHERE status = 'completed'
        ORDER BY variant_id, task_id, repetition
    """
    rows = db.conn.execute(query).fetchall()
    if not rows:
        logger.warning("loader.no_costs")
        return pd.DataFrame()

    df = pd.DataFrame([dict(row) for row in rows])
    logger.info("loader.costs_loaded", rows=len(df))
    return df


def check_completeness(
    db: Database,
    expected_variants: list[str],
    expected_tasks: list[str],
    expected_reps: int,
) -> CompletenessReport:
    """Check the experiment matrix for missing runs and evaluations."""
    report = CompletenessReport(
        n_variants=len(expected_variants),
        n_tasks=len(expected_tasks),
        n_reps=expected_reps,
        expected_runs=len(expected_variants) * len(expected_tasks) * expected_reps,
    )

    # Check completed runs
    completed = db.conn.execute(
        "SELECT variant_id, task_id, repetition FROM runs WHERE status = 'completed'"
    ).fetchall()
    completed_set = {(r["variant_id"], r["task_id"], r["repetition"]) for r in completed}
    report.actual_runs = len(completed_set)

    for v in expected_variants:
        for t in expected_tasks:
            for rep in range(expected_reps):
                if (v, t, rep) not in completed_set:
                    report.missing_runs.append((v, t, rep))

    # Check evaluations for completed designs
    evaluated = db.conn.execute(
        """SELECT d.design_id
           FROM designs d
           JOIN scores_median sm ON d.design_id = sm.design_id
           WHERE d.is_final = TRUE"""
    ).fetchall()
    evaluated_set = {r["design_id"] for r in evaluated}
    report.evaluated_designs = len(evaluated_set)

    # Find unevaluated final designs
    all_final = db.conn.execute("SELECT design_id FROM designs WHERE is_final = TRUE").fetchall()
    for row in all_final:
        if row["design_id"] not in evaluated_set:
            report.missing_evaluations.append(row["design_id"])

    # Check coherence coverage
    coherence_checked = db.conn.execute(
        """SELECT DISTINCT design_id
           FROM coherence_checks"""
    ).fetchall()
    coherence_set = {r["design_id"] for r in coherence_checked}
    report.coherence_checked = len(coherence_set)

    for design_id in evaluated_set:
        if design_id not in coherence_set:
            report.missing_coherence.append(design_id)

    logger.info(
        "loader.completeness",
        run_coverage=f"{report.run_coverage:.1%}",
        eval_coverage=f"{report.eval_coverage:.1%}",
        missing_runs=len(report.missing_runs),
        missing_evals=len(report.missing_evaluations),
        coherence_coverage=f"{report.coherence_coverage:.1%}",
        missing_coherence=len(report.missing_coherence),
    )
    return report
