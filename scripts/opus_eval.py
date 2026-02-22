#!/usr/bin/env python3
"""
Opus 4.6 Second Evaluation of LLM Consortium Designs
=====================================================
Evaluates all 480 final designs with Claude Opus 4.6,
runs coherence checks, computes median scores.
Uses 10 parallel async workers with DB checkpointing.
"""

import asyncio
import json
import os
import re
import sqlite3
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import anthropic

# ── Configuration ──────────────────────────────────────────────────────────
NUM_WORKERS = 10
EVAL_RUNS_PER_DESIGN = 3
MODEL = "claude-opus-4-6-20250219"
MAX_TOKENS_EVAL = 4096
MAX_TOKENS_COHERENCE = 1024
TEMPERATURE_EVAL = 0.3
TEMPERATURE_COHERENCE = 0.1
DB_PATH = "data/opus_eval.db"

# Cost per token (Opus 4.6)
INPUT_COST_PER_M = 15.0
OUTPUT_COST_PER_M = 75.0

# ── Task metadata ──────────────────────────────────────────────────────────
TASK_META = {
    "t1": {"name": "URL Shortener Service", "complexity": "simple", "design_type": "system", "rubric": "system_design"},
    "t2": {"name": "Rate Limiter Service", "complexity": "simple", "design_type": "application", "rubric": "application_design"},
    "t3": {"name": "Multi-Channel Notification System", "complexity": "medium", "design_type": "system", "rubric": "system_design"},
    "t4": {"name": "Distributed Task Queue and Job Scheduler", "complexity": "medium", "design_type": "application", "rubric": "application_design"},
    "t5": {"name": "Multi-Tenant SaaS Billing Platform", "complexity": "complex", "design_type": "system", "rubric": "system_design"},
    "t6": {"name": "Video Streaming Platform", "complexity": "complex", "design_type": "system", "rubric": "system_design"},
    "t7": {"name": "Rule-Based Access Control Module", "complexity": "complex", "design_type": "application", "rubric": "application_design"},
    "t8": {"name": "ML Training and Inference Platform", "complexity": "complex", "design_type": "application", "rubric": "application_design"},
}

# ── Rubric definitions ─────────────────────────────────────────────────────
SYSTEM_DESIGN_DIMS = [
    {"id": "requirement_coverage", "name": "Requirements Understanding", "weight": 0.8,
     "description": "How thoroughly the design addresses all stated requirements and constraints.",
     "anchors": {"1": "Ignores most requirements; major gaps.", "2": "Addresses some but misses critical ones.",
                 "3": "Covers core requirements; minor gaps.", "4": "Addresses all stated requirements with reasonable completeness.",
                 "5": "Comprehensive coverage with proactive identification of implicit needs."}},
    {"id": "capacity_estimation", "name": "Capacity Estimation", "weight": 0.8,
     "description": "Quality of back-of-envelope calculations for storage, bandwidth, and compute.",
     "anchors": {"1": "No estimation or wildly incorrect.", "2": "Partial estimation; off by orders of magnitude.",
                 "3": "Reasonable estimates for primary dimensions.", "4": "Thorough estimation with justified assumptions.",
                 "5": "Precise multi-dimensional estimation with growth projections."}},
    {"id": "architecture_clarity", "name": "Architectural Clarity", "weight": 1.2,
     "description": "How clearly the high-level architecture is communicated and how well components are defined.",
     "anchors": {"1": "No clear architecture; incoherent.", "2": "Vague components; unclear interactions.",
                 "3": "Identifiable components with basic interactions.", "4": "Well-defined components, clear data flow, explicit contracts.",
                 "5": "Crystal-clear architecture with precise boundaries, protocols, failure modes."}},
    {"id": "data_architecture", "name": "Data Architecture", "weight": 1.0,
     "description": "Quality of data model design, storage technology choices, and consistency model.",
     "anchors": {"1": "No data model; inappropriate storage.", "2": "Basic schema; no justification.",
                 "3": "Reasonable model; some rationale.", "4": "Well-designed schema, justified storage, clear consistency model.",
                 "5": "Sophisticated with partitioning, caching, migration plan."}},
    {"id": "api_design", "name": "API Design", "weight": 0.8,
     "description": "Quality of interface definitions, endpoint design, and contract specification.",
     "anchors": {"1": "No API defined.", "2": "Basic endpoints without contracts.",
                 "3": "Core endpoints with schemas; some gaps.", "4": "Comprehensive API with versioning, pagination, error codes.",
                 "5": "Production-grade with rate limiting, idempotency, backward compatibility."}},
    {"id": "scalability_strategy", "name": "Scalability Strategy", "weight": 1.2,
     "description": "How well the design handles growth in users, data, and traffic.",
     "anchors": {"1": "No scalability consideration.", "2": "Mentions scaling but no strategy.",
                 "3": "Basic horizontal scaling.", "4": "Multi-dimensional scaling with partitioning, caching, async.",
                 "5": "Sophisticated with auto-scaling, capacity planning, graceful degradation."}},
    {"id": "reliability_design", "name": "Reliability & Fault Tolerance", "weight": 1.2,
     "description": "Fault tolerance, redundancy, disaster recovery, and availability guarantees.",
     "anchors": {"1": "No fault tolerance; SPOFs.", "2": "Acknowledges need but no mechanisms.",
                 "3": "Basic redundancy and failover.", "4": "Multi-region with circuit breakers, health checks, recovery.",
                 "5": "Comprehensive with chaos engineering readiness, RTO/RPO, blast radius containment."}},
    {"id": "security_design", "name": "Security Posture", "weight": 0.8,
     "description": "Authentication, authorization, encryption, and security threat mitigation.",
     "anchors": {"1": "No security measures.", "2": "Basic auth but no defense in depth.",
                 "3": "Standard auth/authz with encryption.", "4": "Comprehensive with RBAC, audit logging, threat modeling.",
                 "5": "Production-grade with zero-trust, secrets management, compliance."}},
    {"id": "right_sizing", "name": "Right-Sizing / Simplicity", "weight": 0.5,
     "description": "Whether design complexity is proportional to the problem.",
     "anchors": {"1": "Massively over/under-engineered.", "2": "Significant disproportionate complexity.",
                 "3": "Mostly appropriate; some unnecessary layers.", "4": "Complexity proportional; minimal unnecessary abstractions.",
                 "5": "Perfectly right-sized with elegant simplicity."}},
    {"id": "trade_off_awareness", "name": "Trade-off Honesty", "weight": 0.7,
     "description": "Explicit acknowledgment and justification of design trade-offs.",
     "anchors": {"1": "No trade-offs discussed.", "2": "Mentions in passing.",
                 "3": "Identifies key trade-offs with basic rationale.", "4": "Explicit analysis with alternatives and justified choices.",
                 "5": "Deep reasoning with quantified impact and sensitivity analysis."}},
    {"id": "operability", "name": "Operability", "weight": 1.0,
     "description": "Monitoring, observability, deployment strategy, and operational readiness.",
     "anchors": {"1": "No operational considerations.", "2": "Basic logging; no monitoring.",
                 "3": "Standard monitoring with metrics, logs, alerting.", "4": "Comprehensive observability with tracing, SLOs, runbooks, CI/CD.",
                 "5": "Production-grade with canary deploys, feature flags, playbooks, cost monitoring."}},
]

APP_DESIGN_DIMS = [
    {"id": "requirement_coverage", "name": "Requirements & Scope Clarity", "weight": 0.6,
     "description": "How thoroughly the design addresses all stated requirements, identifies ambiguities, and defines clear scope boundaries.",
     "anchors": {"1": "Requirements parroted or ignored. No assumptions. Scope vague.",
                 "2": "Addresses some but misses critical ones. Scope unclear.",
                 "3": "Covers core requirements; minor gaps.",
                 "4": "Addresses all stated requirements. Clear in/out of scope.",
                 "5": "Comprehensive with proactive identification of implicit needs."}},
    {"id": "domain_model", "name": "Domain Model Quality", "weight": 1.4,
     "description": "Quality of entity modeling, relationships, invariants, and domain logic encapsulation.",
     "anchors": {"1": "Anemic data bags. No invariants. Domain logic scattered.",
                 "2": "Basic entities but poor boundaries.",
                 "3": "Reasonable model with relationships; some boundary issues.",
                 "4": "Well-structured with aggregates, value objects, invariants enforced.",
                 "5": "Rich model with bounded contexts, event-driven integration, ubiquitous language."}},
    {"id": "module_structure", "name": "Module Structure & Dependency Direction", "weight": 1.2,
     "description": "Organization of code into cohesive, loosely-coupled modules with clear dependency direction.",
     "anchors": {"1": "Flat or tangled. Domain imports infrastructure. Circular deps.",
                 "2": "Some separation but high coupling.",
                 "3": "Identifiable modules with reasonable separation.",
                 "4": "Well-defined boundaries with dependencies flowing inward.",
                 "5": "Explicit public APIs. Dependencies inward. Navigable from folder structure alone."}},
    {"id": "interface_design", "name": "Interface & Contract Design", "weight": 1.0,
     "description": "Quality of public APIs, contracts, type expressiveness, and abstraction boundaries.",
     "anchors": {"1": "No interfaces. Raw dicts. No contracts.",
                 "2": "Basic interfaces but leaky.",
                 "3": "Defined interfaces; some inconsistency.",
                 "4": "Clean interfaces with consistent patterns. Types are expressive.",
                 "5": "Minimal, well-named with clear contracts (pre/postconditions, error semantics)."}},
    {"id": "data_flow_clarity", "name": "Data Flow Clarity", "weight": 0.8,
     "description": "Clarity of data flow through the application, including sequence diagrams and I/O boundaries.",
     "anchors": {"1": "No data flow docs. Hidden side effects.",
                 "2": "Partial description; missing key paths.",
                 "3": "Primary paths described. I/O boundaries mostly visible.",
                 "4": "Sequence diagrams for critical paths. Clear who calls whom.",
                 "5": "Comprehensive with all critical paths. Transformations documented."}},
    {"id": "error_handling", "name": "Error Handling Rigor", "weight": 1.0,
     "description": "Error taxonomy, recovery strategies, partial failure handling, and context propagation.",
     "anchors": {"1": "No strategy. Bare try/except. Errors swallowed.",
                 "2": "Generic catch-all; poor feedback.",
                 "3": "Typed errors for common cases. Errors carry some context.",
                 "4": "Comprehensive hierarchy with recovery. Partial failure addressed. Correlation IDs.",
                 "5": "Error taxonomy with clear handling rules per category. No broad catch."}},
    {"id": "testability", "name": "Testability", "weight": 1.2,
     "description": "How well the architecture enables the testing pyramid, with injectable dependencies.",
     "anchors": {"1": "Untestable: logic intertwined with I/O. Global state.",
                 "2": "Difficult to test; extensive mocking needed.",
                 "3": "Reasonably testable with standard DI.",
                 "4": "Enables testing pyramid. 70%+ logic testable without I/O. Injectable interfaces.",
                 "5": "Test-first design with examples. Time/randomness injected. Contract/property-based testing."}},
    {"id": "state_management", "name": "State Management & Lifecycle", "weight": 0.6,
     "description": "How application state is organized, mutated, and synchronized.",
     "anchors": {"1": "Ad hoc. Raw status strings. Hidden global mutable state.",
                 "2": "Basic handling but unclear ownership.",
                 "3": "Defined boundaries with reasonable encapsulation.",
                 "4": "Stateful components identified. Transitions explicit. Persistence mapping defined.",
                 "5": "No hidden mutable state. State machines or event sourcing. Full auditability."}},
    {"id": "concurrency_safety", "name": "Concurrency & Safety", "weight": 0.6,
     "description": "How the design handles concurrent access, race conditions, and thread safety.",
     "anchors": {"1": "Not mentioned in concurrent system. No protection.",
                 "2": "Acknowledges concurrency but coarse locks.",
                 "3": "Basic thread safety; some deadlock potential.",
                 "4": "Shared mutable state identified and protected. Model explicit.",
                 "5": "Provably safe. Back-pressure exists. Or single-threaded and justified."}},
    {"id": "cross_cutting_concerns", "name": "Cross-Cutting Concerns", "weight": 0.6,
     "description": "Handling of logging, metrics, configuration, validation.",
     "anchors": {"1": "No mention of logging, observability, config.",
                 "2": "Some utilities but inconsistent.",
                 "3": "Identified with basic middleware/decorator patterns.",
                 "4": "Logging, metrics, validation, config with concrete strategies.",
                 "5": "Elegant architecture with composable middleware. Sensitive data not logged."}},
    {"id": "design_decision_justification", "name": "Design Decision Justification", "weight": 0.5,
     "description": "Explicit discussion of trade-offs, alternatives, and rationale.",
     "anchors": {"1": "Decisions presented as obvious. No alternatives.",
                 "2": "Mentions trade-offs without analysis.",
                 "3": "Identifies key trade-offs with basic rationale.",
                 "4": "Every major choice includes alternatives and acknowledged trade-offs.",
                 "5": "Deep reasoning grounded in specific constraints. Reversibility considered."}},
    {"id": "right_sizing", "name": "Right-Sizing & Simplicity", "weight": 0.5,
     "description": "Whether design complexity is proportional to the problem.",
     "anchors": {"1": "Over-engineered (8-layer for CRUD) or under-designed. Premature abstraction.",
                 "2": "Significant disproportionate complexity.",
                 "3": "Mostly appropriate; some unnecessary layers.",
                 "4": "Complexity proportional. Abstractions for demonstrated needs. YAGNI applied.",
                 "5": "Perfectly right-sized. Simpler approach considered and ruled out with justification."}},
]

SYSTEM_COHERENCE_PAIRS = [
    ("Data Model & Storage Design", "Scalability & Performance Strategy"),
    ("API Design", "Security Design"),
    ("Reliability & Failure Handling", "Trade-off Summary Table"),
]

APP_COHERENCE_PAIRS = [
    ("Error Handling Strategy", "State Management & Lifecycle"),
    ("State Management & Lifecycle", "Concurrency & Thread Safety"),
    ("Interface & Contract Design", "Domain Model"),
    ("Testing Architecture", "Module / Package Structure"),
]


def get_rubric(task_id: str):
    meta = TASK_META[task_id]
    if meta["rubric"] == "system_design":
        return SYSTEM_DESIGN_DIMS, SYSTEM_COHERENCE_PAIRS
    return APP_DESIGN_DIMS, APP_COHERENCE_PAIRS


def build_eval_prompt(design_text: str, task_id: str) -> str:
    meta = TASK_META[task_id]
    dims, _ = get_rubric(task_id)

    rubric_section = ""
    for dim in dims:
        rubric_section += f"\n### {dim['name']} (weight: {dim['weight']})\n{dim['description']}\n\nScore anchors:\n"
        for score, anchor in dim["anchors"].items():
            rubric_section += f"- **{score}**: {anchor}\n"

    return f"""You are an expert evaluator of software system designs. Your task is to score the following design against a detailed rubric. You must be objective, consistent, and thorough.

## Design to Evaluate

{design_text}

## Task Context

**System:** {meta['name']}
**Complexity:** {meta['complexity']}
**Design Type:** {meta['design_type']}

## Evaluation Rubric

Score EACH dimension on a scale of 1.0 to 5.0 (use increments of 0.5). For each dimension:
- Provide the score
- Provide a 2-3 sentence justification citing specific evidence from the design
- List any specific blockers (issues preventing a higher score)

{rubric_section}

## Output Format

You MUST respond with valid JSON in exactly this format:

```json
{{
  "dimension_scores": [
    {{
      "dimension": "dimension_id",
      "score": 4.0,
      "justification": "Brief justification with evidence.",
      "blockers": ["Specific issue 1", "Specific issue 2"]
    }}
  ],
  "overall_score": 3.8,
  "qualitative_summary": "2-3 sentence overall assessment."
}}
```

The `overall_score` should be the weighted average of dimension scores (using the weights above).

Score the design now. Be rigorous and consistent."""


def build_coherence_prompt(design_text: str, section_a: str, section_b: str, design_type: str) -> str:
    return f"""You are checking a {design_type} design document for internal contradictions between two specific sections.

A contradiction exists when one section makes a claim, decision, or assumption that is incompatible with what another section states or implies.

## Full Design Document

{design_text}

## Sections to Compare

**Section A:** {section_a}
**Section B:** {section_b}

Locate both sections in the design above and analyze them for contradictions.

## What Counts as a Contradiction

- One section chooses eventual consistency but the other assumes strong consistency for queries
- One section exposes unauthenticated endpoints but the other claims all endpoints require auth
- One section says "retry all failed writes" but the other warns about write amplification from retries
- One section specifies PostgreSQL but the other references DynamoDB-specific patterns
- One section claims RPO=0 but the other uses async replication

Minor differences in phrasing or emphasis are NOT contradictions. Focus on decisions and claims that are logically incompatible.

## Output Format

Respond with ONLY valid JSON (no markdown fences, no preamble):

If you find a contradiction:
{{"contradicts": true, "severity": "high", "explanation": "Section A states X while Section B assumes Y. These are incompatible because Z.", "section_a_claim": "specific claim from section A", "section_b_claim": "conflicting claim from section B"}}

If the sections are consistent:
{{"contradicts": false, "severity": "none", "explanation": "Both sections consistently assume/use X.", "section_a_claim": "", "section_b_claim": ""}}

Analyze now."""


def parse_json_response(text: str) -> dict:
    """Extract JSON from response, handling markdown fences."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting from markdown fence
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try finding first { to last }
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON from response: {text[:200]}...")


# ── Progress tracking ──────────────────────────────────────────────────────
@dataclass
class Stats:
    total_designs: int = 480
    evals_done: int = 0
    evals_total: int = 0
    coherence_done: int = 0
    coherence_total: int = 0
    medians_done: int = 0
    errors: int = 0
    start_time: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    lock: asyncio.Lock = None

    def __post_init__(self):
        self.lock = asyncio.Lock()
        self.evals_total = self.total_designs * EVAL_RUNS_PER_DESIGN
        # coherence_total is set dynamically

    async def incr_eval(self, inp_tok=0, out_tok=0):
        async with self.lock:
            self.evals_done += 1
            self.input_tokens += inp_tok
            self.output_tokens += out_tok

    async def incr_coherence(self, inp_tok=0, out_tok=0):
        async with self.lock:
            self.coherence_done += 1
            self.input_tokens += inp_tok
            self.output_tokens += out_tok

    async def incr_median(self):
        async with self.lock:
            self.medians_done += 1

    async def incr_error(self):
        async with self.lock:
            self.errors += 1

    def cost(self):
        return (self.input_tokens * INPUT_COST_PER_M + self.output_tokens * OUTPUT_COST_PER_M) / 1_000_000

    def elapsed(self):
        return time.time() - self.start_time

    def rate(self):
        e = self.elapsed()
        if e == 0:
            return 0
        return (self.evals_done + self.coherence_done) / e * 60  # calls per minute


async def progress_printer(stats: Stats):
    """Print progress every 15 seconds."""
    while True:
        await asyncio.sleep(15)
        elapsed = stats.elapsed()
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        pct_eval = (stats.evals_done / stats.evals_total * 100) if stats.evals_total > 0 else 0
        pct_coh = (stats.coherence_done / stats.coherence_total * 100) if stats.coherence_total > 0 else 0

        print(
            f"\r[{mins:02d}:{secs:02d}] "
            f"Evals: {stats.evals_done}/{stats.evals_total} ({pct_eval:.1f}%) | "
            f"Coherence: {stats.coherence_done}/{stats.coherence_total} ({pct_coh:.1f}%) | "
            f"Medians: {stats.medians_done}/{stats.total_designs} | "
            f"Errors: {stats.errors} | "
            f"Rate: {stats.rate():.1f}/min | "
            f"Cost: ${stats.cost():.2f} | "
            f"Tokens: {(stats.input_tokens + stats.output_tokens):,}",
            flush=True,
        )


# ── DB helpers ─────────────────────────────────────────────────────────────
class DB:
    def __init__(self, path: str):
        self.path = path
        self.lock = asyncio.Lock()

    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    async def get_pending_designs(self) -> list:
        """Get designs that haven't been fully evaluated yet."""
        async with self.lock:
            conn = self._conn()
            c = conn.cursor()
            c.execute("""
                SELECT d.design_id, d.full_text, d.run_id, r.task_id, r.variant_id, r.repetition
                FROM designs d
                JOIN runs r ON d.run_id = r.run_id
                WHERE d.is_final = 1
                ORDER BY r.variant_id, r.task_id, r.repetition
            """)
            rows = c.fetchall()
            conn.close()
            return rows

    async def get_eval_count(self, design_id: str) -> int:
        async with self.lock:
            conn = self._conn()
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM evaluations WHERE design_id = ?", (design_id,))
            count = c.fetchone()[0]
            conn.close()
            return count

    async def save_evaluation(self, eval_id, design_id, evaluator_run, dimension_scores,
                              overall_score, qualitative_summary, input_tokens, output_tokens):
        cost = (input_tokens * INPUT_COST_PER_M + output_tokens * OUTPUT_COST_PER_M) / 1_000_000
        async with self.lock:
            conn = self._conn()
            conn.execute("""
                INSERT OR IGNORE INTO evaluations
                (evaluation_id, design_id, evaluator_run, evaluator_model, dimension_scores,
                 overall_score, qualitative_summary, input_tokens, output_tokens, cost_usd)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (eval_id, design_id, evaluator_run, MODEL, json.dumps(dimension_scores),
                  overall_score, qualitative_summary, input_tokens, output_tokens, cost))
            conn.commit()
            conn.close()

    async def save_coherence(self, check_id, design_id, section_pair, contradicts, explanation):
        async with self.lock:
            conn = self._conn()
            conn.execute("""
                INSERT OR IGNORE INTO coherence_checks
                (check_id, design_id, section_pair, contradicts, explanation)
                VALUES (?, ?, ?, ?, ?)
            """, (check_id, design_id, section_pair, contradicts, explanation))
            conn.commit()
            conn.close()

    async def get_coherence_count(self, design_id: str) -> int:
        async with self.lock:
            conn = self._conn()
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM coherence_checks WHERE design_id = ?", (design_id,))
            count = c.fetchone()[0]
            conn.close()
            return count

    async def save_median(self, design_id, run_id, dimension_medians, overall_median,
                          disagreement_flags, blocker_count, blockers_json):
        async with self.lock:
            conn = self._conn()
            conn.execute("""
                INSERT OR REPLACE INTO scores_median
                (design_id, run_id, dimension_medians, overall_median,
                 disagreement_flags, blocker_count, blockers_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (design_id, run_id, json.dumps(dimension_medians), overall_median,
                  json.dumps(disagreement_flags), blocker_count, blockers_json))
            conn.commit()
            conn.close()

    async def has_median(self, design_id: str) -> bool:
        async with self.lock:
            conn = self._conn()
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM scores_median WHERE design_id = ?", (design_id,))
            count = c.fetchone()[0]
            conn.close()
            return count > 0

    async def update_progress(self, design_id: str, **kwargs):
        async with self.lock:
            conn = self._conn()
            sets = ", ".join(f"{k} = ?" for k in kwargs)
            vals = list(kwargs.values()) + [design_id]
            conn.execute(f"UPDATE progress SET {sets} WHERE design_id = ?", vals)
            conn.commit()
            conn.close()

    async def get_evaluations(self, design_id: str) -> list:
        async with self.lock:
            conn = self._conn()
            c = conn.cursor()
            c.execute("SELECT dimension_scores, overall_score FROM evaluations WHERE design_id = ?", (design_id,))
            rows = c.fetchall()
            conn.close()
            return [(json.loads(r[0]), r[1]) for r in rows]


# ── API caller with retries ───────────────────────────────────────────────
async def call_opus(client: anthropic.AsyncAnthropic, prompt: str, max_tokens: int,
                    temperature: float, semaphore: asyncio.Semaphore, stats: Stats,
                    max_retries: int = 3) -> tuple:
    """Call Claude Opus 4.6 with retry logic. Returns (text, input_tokens, output_tokens)."""
    for attempt in range(max_retries):
        try:
            async with semaphore:
                response = await client.messages.create(
                    model=MODEL,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
            text = response.content[0].text
            inp = response.usage.input_tokens
            out = response.usage.output_tokens
            return text, inp, out
        except anthropic.RateLimitError:
            wait = 2 ** (attempt + 1) * 5
            print(f"\n  [Rate limit] Waiting {wait}s before retry {attempt+1}/{max_retries}...", flush=True)
            await asyncio.sleep(wait)
        except anthropic.APIError as e:
            wait = 2 ** attempt * 3
            print(f"\n  [API Error: {e}] Retry {attempt+1}/{max_retries} in {wait}s...", flush=True)
            await asyncio.sleep(wait)
        except Exception as e:
            await stats.incr_error()
            print(f"\n  [Unexpected error: {e}]", flush=True)
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(5)
    raise RuntimeError("Max retries exceeded")


# ── Worker logic ───────────────────────────────────────────────────────────
async def evaluate_design(worker_id: int, design_id: str, design_text: str, task_id: str,
                          run_id: str, variant_id: str, rep: int,
                          client: anthropic.AsyncAnthropic, db: DB,
                          semaphore: asyncio.Semaphore, stats: Stats):
    """Evaluate a single design: 3 eval runs + coherence checks + median computation."""

    meta = TASK_META[task_id]
    dims, coherence_pairs = get_rubric(task_id)

    # ── Phase 1: Evaluation runs ──
    existing_evals = await db.get_eval_count(design_id)
    evals_needed = EVAL_RUNS_PER_DESIGN - existing_evals

    for run_idx in range(existing_evals, EVAL_RUNS_PER_DESIGN):
        prompt = build_eval_prompt(design_text, task_id)
        try:
            text, inp, out = await call_opus(client, prompt, MAX_TOKENS_EVAL, TEMPERATURE_EVAL, semaphore, stats)
            parsed = parse_json_response(text)

            eval_id = uuid.uuid4().hex
            await db.save_evaluation(
                eval_id=eval_id,
                design_id=design_id,
                evaluator_run=run_idx,
                dimension_scores=parsed.get("dimension_scores", []),
                overall_score=parsed.get("overall_score", 0.0),
                qualitative_summary=parsed.get("qualitative_summary", ""),
                input_tokens=inp,
                output_tokens=out,
            )
            await stats.incr_eval(inp, out)
        except Exception as e:
            await stats.incr_error()
            print(f"\n  [W{worker_id}] Eval error {variant_id}/{task_id}/r{rep} run{run_idx}: {e}", flush=True)

    # ── Phase 2: Coherence checks ──
    existing_coherence = await db.get_coherence_count(design_id)
    if existing_coherence < len(coherence_pairs):
        for i, (sec_a, sec_b) in enumerate(coherence_pairs):
            if i < existing_coherence:
                continue
            pair_str = f"{sec_a} <-> {sec_b}"
            prompt = build_coherence_prompt(design_text, sec_a, sec_b, meta["design_type"])
            try:
                text, inp, out = await call_opus(client, prompt, MAX_TOKENS_COHERENCE, TEMPERATURE_COHERENCE, semaphore, stats)
                parsed = parse_json_response(text)
                check_id = uuid.uuid4().hex
                await db.save_coherence(
                    check_id=check_id,
                    design_id=design_id,
                    section_pair=pair_str,
                    contradicts=parsed.get("contradicts", False),
                    explanation=parsed.get("explanation", ""),
                )
                await stats.incr_coherence(inp, out)
            except Exception as e:
                await stats.incr_error()
                print(f"\n  [W{worker_id}] Coherence error {variant_id}/{task_id}/r{rep} pair {i}: {e}", flush=True)

    # ── Phase 3: Compute median scores ──
    if not await db.has_median(design_id):
        evals = await db.get_evaluations(design_id)
        if len(evals) >= EVAL_RUNS_PER_DESIGN:
            try:
                # Collect scores per dimension
                dim_scores = {}
                all_blockers = []
                for dim_list, _ in evals:
                    for d in dim_list:
                        dim_id = d["dimension"]
                        if dim_id not in dim_scores:
                            dim_scores[dim_id] = []
                        dim_scores[dim_id].append(d["score"])
                        all_blockers.extend(d.get("blockers", []))

                # Compute medians
                dim_medians = {}
                disagreement_flags = {}
                for dim_id, scores in dim_scores.items():
                    dim_medians[dim_id] = median(scores)
                    score_range = max(scores) - min(scores)
                    if score_range > 1.5:
                        disagreement_flags[dim_id] = {
                            "range": score_range,
                            "scores": scores,
                        }

                # Overall median
                overall_scores = [os for _, os in evals]
                overall_med = median(overall_scores)

                # Deduplicate blockers
                unique_blockers = list(set(all_blockers))

                await db.save_median(
                    design_id=design_id,
                    run_id=run_id,
                    dimension_medians=dim_medians,
                    overall_median=overall_med,
                    disagreement_flags=disagreement_flags if disagreement_flags else None,
                    blocker_count=len(unique_blockers),
                    blockers_json=json.dumps(unique_blockers),
                )
                await stats.incr_median()
            except Exception as e:
                await stats.incr_error()
                print(f"\n  [W{worker_id}] Median error {variant_id}/{task_id}/r{rep}: {e}", flush=True)


async def worker(worker_id: int, queue: asyncio.Queue,
                 client: anthropic.AsyncAnthropic, db: DB,
                 semaphore: asyncio.Semaphore, stats: Stats):
    """Worker that pulls designs from queue and evaluates them."""
    while True:
        try:
            item = queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        design_id, design_text, run_id, task_id, variant_id, rep = item
        try:
            await evaluate_design(
                worker_id, design_id, design_text, task_id, run_id,
                variant_id, rep, client, db, semaphore, stats,
            )
        except Exception as e:
            await stats.incr_error()
            print(f"\n  [W{worker_id}] Fatal error on {variant_id}/{task_id}/r{rep}: {e}", flush=True)

        queue.task_done()


async def main():
    print("=" * 80)
    print("  OPUS 4.6 SECOND EVALUATION — LLM Consortium Designs")
    print("=" * 80)
    print(f"  Model:     {MODEL}")
    print(f"  Workers:   {NUM_WORKERS}")
    print(f"  Eval runs: {EVAL_RUNS_PER_DESIGN} per design")
    print(f"  Database:  {DB_PATH}")
    print("=" * 80)

    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = anthropic.AsyncAnthropic(api_key=api_key)
    db = DB(DB_PATH)

    # Load all designs
    designs = await db.get_pending_designs()
    print(f"\n  Found {len(designs)} final designs to evaluate")

    # Compute coherence totals
    total_coherence = 0
    for d in designs:
        task_id = d[3]
        _, pairs = get_rubric(task_id)
        total_coherence += len(pairs)

    stats = Stats(total_designs=len(designs))
    stats.coherence_total = total_coherence
    stats.start_time = time.time()

    # Check for already-completed work
    already_evals = 0
    already_coherence = 0
    already_medians = 0
    for d in designs:
        ec = await db.get_eval_count(d[0])
        already_evals += ec
        cc = await db.get_coherence_count(d[0])
        already_coherence += cc
        if await db.has_median(d[0]):
            already_medians += 1

    stats.evals_done = already_evals
    stats.coherence_done = already_coherence
    stats.medians_done = already_medians

    if already_evals > 0 or already_coherence > 0:
        print(f"  Resuming: {already_evals} evals, {already_coherence} coherence checks, {already_medians} medians already done")

    # Filter to designs that still need work
    pending = []
    for d in designs:
        design_id = d[0]
        ec = await db.get_eval_count(design_id)
        _, pairs = get_rubric(d[3])
        cc = await db.get_coherence_count(design_id)
        hm = await db.has_median(design_id)
        if ec < EVAL_RUNS_PER_DESIGN or cc < len(pairs) or not hm:
            pending.append(d)

    print(f"  Pending:  {len(pending)} designs still need work")
    if not pending:
        print("\n  All designs already evaluated! Nothing to do.")
        return

    # Build work queue
    queue = asyncio.Queue()
    for d in pending:
        queue.put_nowait(d)

    # Semaphore to control API concurrency
    semaphore = asyncio.Semaphore(NUM_WORKERS)

    print(f"\n  Starting {NUM_WORKERS} workers...")
    print(f"  Total API calls needed: ~{len(pending) * EVAL_RUNS_PER_DESIGN + total_coherence - already_coherence - already_evals}")
    print("-" * 80)

    # Start progress printer
    progress_task = asyncio.create_task(progress_printer(stats))

    # Start workers
    workers = [
        asyncio.create_task(worker(i, queue, client, db, semaphore, stats))
        for i in range(NUM_WORKERS)
    ]

    await asyncio.gather(*workers)
    progress_task.cancel()

    # Final stats
    elapsed = stats.elapsed()
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    print(f"\n\n{'=' * 80}")
    print(f"  EVALUATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"  Time:       {mins}m {secs}s")
    print(f"  Evals:      {stats.evals_done}/{stats.evals_total}")
    print(f"  Coherence:  {stats.coherence_done}/{stats.coherence_total}")
    print(f"  Medians:    {stats.medians_done}/{stats.total_designs}")
    print(f"  Errors:     {stats.errors}")
    print(f"  Cost:       ${stats.cost():.2f}")
    print(f"  Tokens:     {stats.input_tokens + stats.output_tokens:,}")
    print(f"  Rate:       {stats.rate():.1f} calls/min")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    asyncio.run(main())
