"""Objective-result statistics for the coding consortium (Paper #2).

Reads the ``code_results`` table and computes the metrics the paper reports:

* **pass@1** per condition/benchmark (mean over problems x reps).
* **pass@k** ("solved if any rep passes") - the oracle/any-pass upper bound.
* **McNemar's exact test** on paired per-problem outcomes (the correct test for
  two conditions graded on the *same* problems).
* **Bootstrap CIs** on a condition's pass@1 and on a consortium-baseline delta.
* **Per-problem win/loss** between two conditions (the "where does the
  consortium fix vs. regress" matrix).

These are paired-binary analogues of Paper #1's continuous-quality statistics.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from consortium.storage.database import Database

_BOOTSTRAP_ITERS = 10_000
_BOOTSTRAP_SEED = 42


@dataclass(frozen=True)
class ConditionResult:
    """pass@1 / pass@k summary for one condition on one benchmark."""

    variant_id: str
    n_problems: int
    n_samples: int
    pass_at_1: float
    pass_at_k: float
    ci_low: float
    ci_high: float


def fetch_results(db: Database, benchmark: str) -> list[dict[str, Any]]:
    """All scored rows for *benchmark* with their condition + problem + rep."""
    rows = db.conn.execute(
        "SELECT cr.problem_id, cr.passed, r.variant_id, r.repetition "
        "FROM code_results cr JOIN runs r ON cr.run_id = r.run_id "
        "WHERE cr.benchmark = ?",
        (benchmark,),
    ).fetchall()
    return [dict(r) for r in rows]


def _by_condition(rows: list[dict[str, Any]]) -> dict[str, dict[str, list[bool]]]:
    """{variant_id: {problem_id: [passed per rep]}}."""
    out: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        out[r["variant_id"]][r["problem_id"]].append(bool(r["passed"]))
    return out


def _bootstrap_ci(values: list[float]) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for the mean of *values*."""
    if not values:
        return (0.0, 0.0)
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    arr = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(arr), size=(_BOOTSTRAP_ITERS, len(arr)))
    means = arr[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def condition_results(db: Database, benchmark: str) -> list[ConditionResult]:
    """pass@1, pass@k and a bootstrap CI for every condition on *benchmark*."""
    grouped = _by_condition(fetch_results(db, benchmark))
    results: list[ConditionResult] = []
    for vid, problems in grouped.items():
        # pass@1 per problem (mean over reps); pass@k per problem (any rep passes)
        per_problem_rate = [sum(reps) / len(reps) for reps in problems.values()]
        any_pass = [1.0 if any(reps) else 0.0 for reps in problems.values()]
        all_samples = [1.0 if p else 0.0 for reps in problems.values() for p in reps]
        lo, hi = _bootstrap_ci(per_problem_rate)
        results.append(
            ConditionResult(
                variant_id=vid,
                n_problems=len(problems),
                n_samples=len(all_samples),
                pass_at_1=float(np.mean(all_samples)) if all_samples else 0.0,
                pass_at_k=float(np.mean(any_pass)) if any_pass else 0.0,
                ci_low=lo,
                ci_high=hi,
            )
        )
    results.sort(key=lambda r: r.variant_id)
    return results


def _problem_passed(reps: list[bool]) -> bool:
    """Collapse a problem's reps to a single pass/fail (majority, ties→pass)."""
    return sum(reps) * 2 >= len(reps)


def mcnemar(db: Database, benchmark: str, variant_a: str, variant_b: str) -> dict[str, Any]:
    """Exact McNemar test on paired per-problem outcomes for two conditions.

    Returns counts plus the exact two-sided p-value (binomial on the
    discordant pairs). ``b`` = A passes / B fails, ``c`` = A fails / B passes.
    """
    grouped = _by_condition(fetch_results(db, benchmark))
    a, b = grouped.get(variant_a, {}), grouped.get(variant_b, {})
    shared = sorted(set(a) & set(b))
    n_b = n_c = both = neither = 0
    for pid in shared:
        ap, bp = _problem_passed(a[pid]), _problem_passed(b[pid])
        if ap and not bp:
            n_b += 1
        elif bp and not ap:
            n_c += 1
        elif ap and bp:
            both += 1
        else:
            neither += 1
    discordant = n_b + n_c
    p_value = (
        float(stats.binomtest(min(n_b, n_c), discordant, 0.5).pvalue) if discordant else 1.0
    )
    return {
        "variant_a": variant_a,
        "variant_b": variant_b,
        "n_paired": len(shared),
        "both_pass": both,
        "neither_pass": neither,
        "a_only": n_b,  # A fixes what B misses
        "b_only": n_c,  # B fixes what A misses
        "p_value": p_value,
    }
