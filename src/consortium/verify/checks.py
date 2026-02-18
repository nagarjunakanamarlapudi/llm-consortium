"""E2E verification checks for experiment databases.

Each check function accepts a sqlite3.Connection and returns a CheckResult.
Use ``run_all_checks`` to execute the full suite.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CheckResult:
    """Outcome of a single verification check."""

    name: str
    passed: bool
    total: int
    failures: int
    details: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Framework-level Jinja2 variables that should NEVER appear literally in
# rendered prompt text.  Content-level curly braces (e.g. ``{{order_id}}``)
# are ignored because they come from the LLM-generated design text.
# ---------------------------------------------------------------------------
_FRAMEWORK_VARS = [
    "system_name",
    "problem_statement",
    "perspective",
    "previous_design",
    "review",
    "designs",
    "round",
    "position",
    "own_design",
    "other_designs",
    "design_text",
    "debate_history",
    "task_context",
]

_FRAMEWORK_RE = re.compile(
    r"\{\{\s*(" + "|".join(_FRAMEWORK_VARS) + r")\s*\}\}"
)

_DESIGN_KEYWORDS = [
    "api",
    "database",
    "component",
    "service",
    "architecture",
    "scalability",
    "interface",
    "endpoint",
    "schema",
    "module",
    "microservice",
]


# ── Prompt Integrity (1-3) ─────────────────────────────────────────────────


def check_template_variables(conn: sqlite3.Connection) -> CheckResult:
    """1. No unreplaced framework Jinja2 variables in rendered prompts."""
    rows = conn.execute(
        "SELECT run_id, round, agent_role, prompt_text FROM traces "
        "WHERE prompt_text IS NOT NULL"
    ).fetchall()
    failures: list[str] = []
    for row in rows:
        matches = _FRAMEWORK_RE.findall(row["prompt_text"])
        if matches:
            unique = sorted(set(matches))
            failures.append(
                f"{row['run_id']} round={row['round']} "
                f"agent={row['agent_role']}: {{{{{', '.join(unique)}}}}}"
            )
    return CheckResult(
        name="template_variables",
        passed=len(failures) == 0,
        total=len(rows),
        failures=len(failures),
        details=failures[:10],
    )


def check_empty_prompts(conn: sqlite3.Connection) -> CheckResult:
    """2. No null or empty prompt_text in traces."""
    total = conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
    rows = conn.execute(
        "SELECT run_id, round, agent_role FROM traces "
        "WHERE prompt_text IS NULL OR TRIM(prompt_text) = ''"
    ).fetchall()
    failures = [
        f"{r['run_id']} round={r['round']} agent={r['agent_role']}"
        for r in rows
    ]
    return CheckResult(
        name="empty_prompts",
        passed=len(failures) == 0,
        total=total,
        failures=len(failures),
        details=failures[:10],
    )


def check_empty_responses(conn: sqlite3.Connection) -> CheckResult:
    """3. No null or empty response_text in traces."""
    total = conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
    rows = conn.execute(
        "SELECT run_id, round, agent_role FROM traces "
        "WHERE response_text IS NULL OR TRIM(response_text) = ''"
    ).fetchall()
    failures = [
        f"{r['run_id']} round={r['round']} agent={r['agent_role']}"
        for r in rows
    ]
    return CheckResult(
        name="empty_responses",
        passed=len(failures) == 0,
        total=total,
        failures=len(failures),
        details=failures[:10],
    )


# ── Content Injection (4-7) ────────────────────────────────────────────────


def check_task_name_injection(conn: sqlite3.Connection) -> CheckResult:
    """4. Generation prompts contain the task system_name."""
    rows = conn.execute(
        "SELECT t.run_id, t.round, t.prompt_text, r.task_config "
        "FROM traces t JOIN runs r ON t.run_id = r.run_id "
        "WHERE t.step = 'generation' AND t.round = 0 "
        "AND t.prompt_text IS NOT NULL"
    ).fetchall()
    failures: list[str] = []
    for row in rows:
        try:
            task_cfg = json.loads(row["task_config"])
            sys_name = task_cfg.get("system_name", "")
        except (json.JSONDecodeError, TypeError):
            continue
        if sys_name and sys_name.lower() not in row["prompt_text"].lower():
            failures.append(
                f"{row['run_id']}: missing system_name '{sys_name}'"
            )
    return CheckResult(
        name="task_name_injection",
        passed=len(failures) == 0,
        total=len(rows),
        failures=len(failures),
        details=failures[:10],
    )


def check_perspective_injection(conn: sqlite3.Connection) -> CheckResult:
    """5. Non-leader reviewer/specialist agents have perspective in prompt."""
    rows = conn.execute(
        "SELECT t.run_id, t.round, t.agent_role, t.agent_id, "
        "       t.prompt_text, t.variant_id "
        "FROM traces t "
        "WHERE t.prompt_text IS NOT NULL "
        "AND (t.agent_role LIKE '%reviewer%' "
        "     OR t.agent_role LIKE '%specialist%' "
        "     OR t.agent_role LIKE '%adversarial%')"
    ).fetchall()
    failures: list[str] = []
    perspective_keywords = [
        "perspective",
        "role",
        "focus",
        "expertise",
        "viewpoint",
        "lens",
        "angle",
        "standpoint",
        "advocate",
        "critic",
    ]
    for row in rows:
        prompt_lower = row["prompt_text"].lower()
        has_perspective = any(kw in prompt_lower for kw in perspective_keywords)
        if not has_perspective:
            failures.append(
                f"{row['run_id']} {row['agent_role']}: "
                f"no perspective keyword found"
            )
    return CheckResult(
        name="perspective_injection",
        passed=len(failures) == 0,
        total=len(rows),
        failures=len(failures),
        details=failures[:10],
    )


def check_review_contains_design(conn: sqlite3.Connection) -> CheckResult:
    """6. Review-step prompts reference the prior design (chars 500-700)."""
    reviews = conn.execute(
        "SELECT t.run_id, t.round, t.agent_role, t.prompt_text "
        "FROM traces t WHERE t.step = 'review' AND t.prompt_text IS NOT NULL"
    ).fetchall()
    designs = conn.execute(
        "SELECT run_id, round, full_text FROM designs "
        "WHERE full_text IS NOT NULL AND LENGTH(full_text) > 700"
    ).fetchall()

    # Build lookup: (run_id, round) -> design snippet
    design_map: dict[tuple[str, int], str] = {}
    for d in designs:
        snippet = d["full_text"][500:700]
        design_map[(d["run_id"], d["round"])] = snippet

    failures: list[str] = []
    checked = 0
    for row in reviews:
        # Review at round N references design from round N-1
        key = (row["run_id"], row["round"] - 1)
        snippet = design_map.get(key)
        if snippet is None:
            continue  # design too short or missing — skip
        checked += 1
        if snippet not in row["prompt_text"]:
            failures.append(
                f"{row['run_id']} round={row['round']} "
                f"{row['agent_role']}: design snippet not found"
            )
    return CheckResult(
        name="review_contains_design",
        passed=len(failures) == 0,
        total=checked,
        failures=len(failures),
        details=failures[:10],
    )


def check_revision_contains_review(conn: sqlite3.Connection) -> CheckResult:
    """7. Revision-step prompts reference prior review feedback (chars 100-300)."""
    revisions = conn.execute(
        "SELECT t.run_id, t.round, t.agent_role, t.prompt_text "
        "FROM traces t WHERE t.step = 'revision' AND t.prompt_text IS NOT NULL"
    ).fetchall()
    # Review responses from prior round
    review_responses = conn.execute(
        "SELECT run_id, round, response_text FROM traces "
        "WHERE step = 'review' AND response_text IS NOT NULL "
        "AND LENGTH(response_text) > 300"
    ).fetchall()

    # Build lookup: (run_id, round) -> review snippet
    review_map: dict[tuple[str, int], str] = {}
    for r in review_responses:
        snippet = r["response_text"][100:300]
        review_map[(r["run_id"], r["round"])] = snippet

    failures: list[str] = []
    checked = 0
    for row in revisions:
        # Revision at round N uses review from round N-1 or same round
        for review_round in (row["round"], row["round"] - 1):
            key = (row["run_id"], review_round)
            snippet = review_map.get(key)
            if snippet is not None:
                checked += 1
                if snippet not in row["prompt_text"]:
                    failures.append(
                        f"{row['run_id']} round={row['round']} "
                        f"{row['agent_role']}: review snippet not found"
                    )
                break
    return CheckResult(
        name="revision_contains_review",
        passed=len(failures) == 0,
        total=checked,
        failures=len(failures),
        details=failures[:10],
    )


# ── Prompt Size Progression (8) ───────────────────────────────────────────


def check_prompt_size_growth(conn: sqlite3.Connection) -> CheckResult:
    """8. Prompt sizes generally increase across rounds within a run."""
    rows = conn.execute(
        "SELECT run_id, round, step, LENGTH(prompt_text) as plen "
        "FROM traces WHERE prompt_text IS NOT NULL "
        "ORDER BY run_id, round, step"
    ).fetchall()

    # Group by run_id: collect (round, step, plen)
    runs: dict[str, list[tuple[int, str, int]]] = {}
    for row in rows:
        runs.setdefault(row["run_id"], []).append(
            (row["round"], row["step"], row["plen"])
        )

    failures: list[str] = []
    for run_id, entries in runs.items():
        if len(entries) < 2:
            continue
        # Compare max prompt size per round
        round_max: dict[int, int] = {}
        for rnd, _step, plen in entries:
            round_max[rnd] = max(round_max.get(rnd, 0), plen)
        sorted_rounds = sorted(round_max.items())
        for i in range(1, len(sorted_rounds)):
            prev_rnd, prev_size = sorted_rounds[i - 1]
            curr_rnd, curr_size = sorted_rounds[i]
            # Allow some shrinkage (e.g. final summary rounds), flag big drops
            if curr_size < prev_size * 0.5:
                failures.append(
                    f"{run_id}: round {curr_rnd} ({curr_size:,}) < "
                    f"50% of round {prev_rnd} ({prev_size:,})"
                )

    return CheckResult(
        name="prompt_size_growth",
        passed=len(failures) == 0,
        total=len(runs),
        failures=len(failures),
        details=failures[:10],
    )


# ── Deep Content Checks (9-12) ────────────────────────────────────────────


def check_revision_design_markers(conn: sqlite3.Connection) -> CheckResult:
    """9. Revision prompts mention design-relevant keywords."""
    rows = conn.execute(
        "SELECT run_id, round, agent_role, prompt_text FROM traces "
        "WHERE step = 'revision' AND prompt_text IS NOT NULL"
    ).fetchall()
    failures: list[str] = []
    for row in rows:
        prompt_lower = row["prompt_text"].lower()
        hits = sum(1 for kw in _DESIGN_KEYWORDS if kw in prompt_lower)
        if hits < 2:
            failures.append(
                f"{row['run_id']} round={row['round']}: "
                f"only {hits} design keywords found"
            )
    return CheckResult(
        name="revision_design_markers",
        passed=len(failures) == 0,
        total=len(rows),
        failures=len(failures),
        details=failures[:10],
    )


def check_review_has_design_output(conn: sqlite3.Connection) -> CheckResult:
    """10. Review prompts contain a snippet from the design they review."""
    # This is similar to check 6 but checks design full_text output directly
    reviews = conn.execute(
        "SELECT t.run_id, t.round, t.agent_role, t.prompt_text "
        "FROM traces t WHERE t.step = 'review' AND t.prompt_text IS NOT NULL"
    ).fetchall()
    # Get design response_text from generation traces
    gen_responses = conn.execute(
        "SELECT run_id, round, response_text FROM traces "
        "WHERE step = 'generation' AND response_text IS NOT NULL "
        "AND LENGTH(response_text) > 700"
    ).fetchall()

    resp_map: dict[tuple[str, int], str] = {}
    for g in gen_responses:
        snippet = g["response_text"][500:700]
        resp_map[(g["run_id"], g["round"])] = snippet

    failures: list[str] = []
    checked = 0
    for row in reviews:
        key = (row["run_id"], row["round"])
        snippet = resp_map.get(key)
        if snippet is None:
            continue
        checked += 1
        if snippet not in row["prompt_text"]:
            failures.append(
                f"{row['run_id']} round={row['round']} "
                f"{row['agent_role']}: gen response snippet not in review prompt"
            )
    return CheckResult(
        name="review_has_design_output",
        passed=len(failures) == 0,
        total=checked,
        failures=len(failures),
        details=failures[:10],
    )


def check_revision_has_review_text(conn: sqlite3.Connection) -> CheckResult:
    """11. Revision prompts contain a snippet from review feedback."""
    revisions = conn.execute(
        "SELECT run_id, round, agent_role, prompt_text FROM traces "
        "WHERE step = 'revision' AND prompt_text IS NOT NULL"
    ).fetchall()
    review_responses = conn.execute(
        "SELECT run_id, round, response_text FROM traces "
        "WHERE step = 'review' AND response_text IS NOT NULL "
        "AND LENGTH(response_text) > 300"
    ).fetchall()

    review_map: dict[tuple[str, int], str] = {}
    for r in review_responses:
        snippet = r["response_text"][100:300]
        review_map[(r["run_id"], r["round"])] = snippet

    failures: list[str] = []
    checked = 0
    for row in revisions:
        for review_round in (row["round"], row["round"] - 1):
            key = (row["run_id"], review_round)
            snippet = review_map.get(key)
            if snippet is not None:
                checked += 1
                if snippet not in row["prompt_text"]:
                    failures.append(
                        f"{row['run_id']} round={row['round']} "
                        f"{row['agent_role']}: review text snippet not found"
                    )
                break
    return CheckResult(
        name="revision_has_review_text",
        passed=len(failures) == 0,
        total=checked,
        failures=len(failures),
        details=failures[:10],
    )


def check_parallel_agent_symmetry(conn: sqlite3.Connection) -> CheckResult:
    """12. Parallel agents in generation steps receive distinct prompts.

    Review-step agents may legitimately receive identical prompts (e.g. V2
    reviewers all reviewing the same design).  This check only flags
    *generation* steps where multiple agents produce identical prompts,
    which would indicate missing perspective differentiation.
    """
    rows = conn.execute(
        "SELECT run_id, round, step, agent_id, prompt_text, variant_id "
        "FROM traces WHERE prompt_text IS NOT NULL "
        "AND step = 'generation' "
        "ORDER BY run_id, round, step"
    ).fetchall()

    # Group by (run_id, round, step) -> list of (agent_id, prompt_text)
    groups: dict[tuple[str, int, str], list[tuple[str, str]]] = {}
    for row in rows:
        key = (row["run_id"], row["round"], row["step"])
        groups.setdefault(key, []).append(
            (row["agent_id"], row["prompt_text"])
        )

    failures: list[str] = []
    checked = 0
    for (run_id, rnd, step), agents in groups.items():
        if len(agents) < 2:
            continue
        checked += 1
        # Check that at least some agents have different prompts
        unique_prompts = set(p for _, p in agents)
        if len(unique_prompts) == 1 and len(agents) > 1:
            failures.append(
                f"{run_id} round={rnd} step={step}: "
                f"{len(agents)} agents with identical prompts"
            )
    return CheckResult(
        name="parallel_agent_symmetry",
        passed=len(failures) == 0,
        total=checked,
        failures=len(failures),
        details=failures[:10],
    )


# ── Evaluation Integrity (13-15) ──────────────────────────────────────────


def check_evaluation_scores(conn: sqlite3.Connection) -> CheckResult:
    """13. All evaluations have valid scores (0-10) and parseable dimensions."""
    rows = conn.execute(
        "SELECT evaluation_id, design_id, overall_score, dimension_scores "
        "FROM evaluations"
    ).fetchall()
    failures: list[str] = []
    for row in rows:
        # Check overall_score range
        score = row["overall_score"]
        if score is None or score < 0 or score > 10:
            failures.append(
                f"eval {row['evaluation_id']}: "
                f"overall_score={score} out of range"
            )
            continue
        # Check dimension_scores is valid JSON (dict or list-of-dicts)
        dims = row["dimension_scores"]
        if dims is None:
            failures.append(
                f"eval {row['evaluation_id']}: null dimension_scores"
            )
            continue
        try:
            parsed = json.loads(dims)
            if isinstance(parsed, dict):
                pass  # OK
            elif isinstance(parsed, list) and all(
                isinstance(d, dict) for d in parsed
            ):
                pass  # list of dimension dicts — also OK
            else:
                failures.append(
                    f"eval {row['evaluation_id']}: "
                    f"dimension_scores unexpected type: {type(parsed).__name__}"
                )
        except json.JSONDecodeError:
            failures.append(
                f"eval {row['evaluation_id']}: "
                f"dimension_scores not valid JSON"
            )
    return CheckResult(
        name="evaluation_scores",
        passed=len(failures) == 0,
        total=len(rows),
        failures=len(failures),
        details=failures[:10],
    )


def check_eval_coverage(conn: sqlite3.Connection) -> CheckResult:
    """14. Every final design has >= 3 evaluator runs."""
    rows = conn.execute(
        "SELECT d.design_id, d.run_id, "
        "       COUNT(e.evaluation_id) as eval_count "
        "FROM designs d "
        "LEFT JOIN evaluations e ON d.design_id = e.design_id "
        "WHERE d.is_final = 1 "
        "GROUP BY d.design_id"
    ).fetchall()
    failures: list[str] = []
    for row in rows:
        if row["eval_count"] < 3:
            failures.append(
                f"{row['run_id']} design={row['design_id'][:12]}...: "
                f"only {row['eval_count']} evals (expected >= 3)"
            )
    return CheckResult(
        name="eval_coverage",
        passed=len(failures) == 0,
        total=len(rows),
        failures=len(failures),
        details=failures[:10],
    )


def check_coherence_integrity(conn: sqlite3.Connection) -> CheckResult:
    """15. All coherence checks have non-null section_pair and explanation."""
    rows = conn.execute(
        "SELECT check_id, design_id, section_pair, explanation "
        "FROM coherence_checks"
    ).fetchall()
    failures: list[str] = []
    for row in rows:
        if not row["section_pair"]:
            failures.append(
                f"check {row['check_id']}: null section_pair"
            )
        if not row["explanation"]:
            failures.append(
                f"check {row['check_id']}: null explanation"
            )
    return CheckResult(
        name="coherence_integrity",
        passed=len(failures) == 0,
        total=len(rows),
        failures=len(failures),
        details=failures[:10],
    )


# ── Cross-Run Integrity (16-17) ───────────────────────────────────────────


def check_context_chaining(conn: sqlite3.Connection) -> CheckResult:
    """16. Multi-round runs chain context (round N response in round N+1 prompt)."""
    rows = conn.execute(
        "SELECT run_id, round, step, prompt_text, response_text "
        "FROM traces WHERE prompt_text IS NOT NULL "
        "AND response_text IS NOT NULL "
        "ORDER BY run_id, round, step"
    ).fetchall()

    # Group by run_id -> sorted list of (round, step, prompt, response)
    runs: dict[str, list[tuple[int, str, str, str]]] = {}
    for row in rows:
        runs.setdefault(row["run_id"], []).append(
            (row["round"], row["step"], row["prompt_text"], row["response_text"])
        )

    failures: list[str] = []
    checked = 0
    for run_id, entries in runs.items():
        rounds_present = sorted(set(rnd for rnd, *_ in entries))
        if len(rounds_present) < 2:
            continue

        for i in range(len(rounds_present) - 1):
            prev_round = rounds_present[i]
            next_round = rounds_present[i + 1]

            # Get last response from prev_round
            prev_responses = [
                resp for rnd, _s, _p, resp in entries if rnd == prev_round
            ]
            if not prev_responses:
                continue
            last_resp = prev_responses[-1]
            if len(last_resp) < 200:
                continue

            # Check if snippet appears in any next_round prompt
            snippet = last_resp[:200]
            next_prompts = [
                prompt for rnd, _s, prompt, _r in entries if rnd == next_round
            ]
            if not next_prompts:
                continue

            checked += 1
            found = any(snippet in p for p in next_prompts)
            if not found:
                failures.append(
                    f"{run_id}: round {prev_round} response "
                    f"not found in round {next_round} prompts"
                )

    return CheckResult(
        name="context_chaining",
        passed=len(failures) == 0,
        total=checked,
        failures=len(failures),
        details=failures[:10],
    )


def check_cross_task_contamination(conn: sqlite3.Connection) -> CheckResult:
    """17. No prompt for task X contains design text from a different task."""
    # Get design snippets per task
    designs = conn.execute(
        "SELECT d.design_id, r.task_id, d.full_text "
        "FROM designs d JOIN runs r ON d.run_id = r.run_id "
        "WHERE d.is_final = 1 AND LENGTH(d.full_text) > 700"
    ).fetchall()
    # Get all prompts with their task_id
    prompts = conn.execute(
        "SELECT t.trace_id, t.run_id, r.task_id, t.prompt_text "
        "FROM traces t JOIN runs r ON t.run_id = r.run_id "
        "WHERE t.prompt_text IS NOT NULL"
    ).fetchall()

    # Build per-task design snippets (chars 500-700 from middle of text)
    task_snippets: dict[str, list[str]] = {}
    for d in designs:
        snippet = d["full_text"][500:700]
        task_snippets.setdefault(d["task_id"], []).append(snippet)

    failures: list[str] = []
    checked = 0
    for p in prompts:
        prompt_task = p["task_id"]
        for other_task, snippets in task_snippets.items():
            if other_task == prompt_task:
                continue
            checked += 1
            for snippet in snippets:
                if snippet in p["prompt_text"]:
                    failures.append(
                        f"{p['run_id']}: prompt contains "
                        f"design text from task {other_task}"
                    )
                    break

    return CheckResult(
        name="cross_task_contamination",
        passed=len(failures) == 0,
        total=checked,
        failures=len(failures),
        details=failures[:10],
    )


# ── Runner ─────────────────────────────────────────────────────────────────

ALL_CHECKS = [
    check_template_variables,
    check_empty_prompts,
    check_empty_responses,
    check_task_name_injection,
    check_perspective_injection,
    check_review_contains_design,
    check_revision_contains_review,
    check_prompt_size_growth,
    check_revision_design_markers,
    check_review_has_design_output,
    check_revision_has_review_text,
    check_parallel_agent_symmetry,
    check_evaluation_scores,
    check_eval_coverage,
    check_coherence_integrity,
    check_context_chaining,
    check_cross_task_contamination,
]

CHECK_NAMES = {fn.__name__.removeprefix("check_"): fn for fn in ALL_CHECKS}


def run_all_checks(
    db_path: Path,
    *,
    names: list[str] | None = None,
) -> list[CheckResult]:
    """Run verification checks against an experiment database.

    Parameters
    ----------
    db_path:
        Path to the SQLite experiment database.
    names:
        Optional list of check names to run.  If ``None``, all checks run.

    Returns
    -------
    list[CheckResult]
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    checks = ALL_CHECKS
    if names:
        checks = [CHECK_NAMES[n] for n in names if n in CHECK_NAMES]

    results: list[CheckResult] = []
    try:
        for check_fn in checks:
            results.append(check_fn(conn))
    finally:
        conn.close()

    return results
