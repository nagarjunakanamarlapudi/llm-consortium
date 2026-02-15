# LLM Consortium — Implementation Plan (Updated)

> **Goal:** Build a fully configurable, observable, reproducible experiment pipeline that
> executes 8 consortium variants × 8 design tasks × 5 repetitions = 320 runs, evaluates
> every output via LLM-based rubric scoring, and culminates in generating
> **Section 10.2 — The Practitioner Decision Framework**.

---

## Implementation Phases (Actual)

| Phase | Scope | Status |
|-------|-------|--------|
| **Phase 1** | Prompts, rubrics, templates, thesis proposal | ✅ COMPLETE |
| **Phase 2** | Config system, providers, agents, orchestrators, engine, CLI, evaluation pipeline, batch layer | ✅ COMPLETE |
| **Phase 3** | Observability, analysis, pilot runs, full experiment, decision framework | 🔴 NOT STARTED |

---

## Table of Contents

1. [Current Codebase (What's Built)](#1-current-codebase)
2. [Phase 3 — Remaining Work](#2-phase-3--remaining-work)
3. [Phase 3A — Observability & Timeline](#3a-observability--timeline)
4. [Phase 3B — Analysis & Visualization](#3b-analysis--visualization)
5. [Phase 3C — Coherence Checker Integration](#3c-coherence-checker)
6. [Phase 3D — Pilot Runs & Evaluator Validation](#3d-pilot-runs)
7. [Phase 3E — Full Experiment Execution](#3e-full-experiment)
8. [Phase 3F — Decision Framework Generation](#3f-decision-framework)
9. [Phase 3G — Tests](#3g-tests)
10. [Database Schema](#4-database-schema)
11. [CLI Reference](#5-cli-reference)
12. [Config File Reference](#6-config-reference)

---

## 1. Current Codebase (What's Built)

### Actual Project Structure

```
llm-consortium/
├── pyproject.toml
├── uv.lock
├── .python-version                 # Python 3.12
├── .env.example
├── .env                            # API keys (gitignored)
├── .gitignore
├── README.md
│
├── configs/                        # ALL configuration
│   ├── experiment.yaml             # top-level experiment config
│   ├── evaluator.yaml              # evaluator pipeline config
│   ├── coverage_matrix.yaml        # rubric dimension → design section mapping
│   ├── models/                     # 6 model provider definitions
│   │   ├── anthropic_sonnet.yaml
│   │   ├── anthropic_opus.yaml
│   │   ├── openai_gpt4_1.yaml
│   │   ├── openai_gpt4o.yaml
│   │   ├── google_gemini3.yaml
│   │   └── ollama_local.yaml
│   ├── variants/                   # 8 variant configs
│   │   ├── v1_baseline.yaml
│   │   ├── v2_leader_reviewers.yaml
│   │   ├── v3_parallel_merge.yaml
│   │   ├── v4_adversarial.yaml
│   │   ├── v5_specialist_panel.yaml
│   │   ├── v6_rotating_leader.yaml
│   │   ├── v7_consensus.yaml
│   │   └── v8_structured_debate.yaml
│   ├── tasks/                      # 8 design tasks (4 system + 4 application)
│   │   ├── t1_url_shortener.yaml       # system
│   │   ├── t2_chat_app.yaml            # application
│   │   ├── t3_ecommerce.yaml           # application
│   │   ├── t4_notification.yaml        # system
│   │   ├── t5_trading_platform.yaml    # application
│   │   ├── t6_video_streaming.yaml     # system
│   │   ├── t7_autonomous_fleet.yaml    # system
│   │   └── t8_ml_platform.yaml         # application
│   └── rubrics/                    # evaluation rubrics
│       ├── system_design.yaml      # 11 dimensions for system tasks
│       └── application_design.yaml # 12 dimensions for application tasks
│
├── prompts/                        # Jinja2 templates
│   ├── generation/
│   │   ├── design_system.j2        # system design prompt
│   │   └── design_application.j2   # application design prompt
│   ├── review/
│   │   ├── general_review.j2       # standard review (v2, v6)
│   │   ├── specialist_review.j2    # domain-specific review (v5)
│   │   └── adversarial_review.j2   # reject/accept gate (v4)
│   ├── synthesis/
│   │   ├── merge_naive.j2          # simple merge (v3a)
│   │   ├── merge_rubric_guided.j2  # rubric-weighted merge (v3b)
│   │   ├── merge_dialectical.j2    # thesis-antithesis merge (v3c)
│   │   └── consensus_synthesis.j2  # convergence synthesis (v7)
│   ├── debate/
│   │   ├── position_assignment.j2  # debate stance (v8)
│   │   ├── rebuttal.j2             # counter-argument (v8)
│   │   └── judge_ruling.j2         # final ruling (v8)
│   └── evaluation/
│       ├── evaluate_design.j2      # rubric scoring prompt
│       └── coherence_check.j2      # internal consistency check
│
├── src/consortium/
│   ├── __init__.py
│   ├── py.typed
│   │
│   ├── config/                     # ✅ COMPLETE
│   │   ├── loader.py               # YAML loading, env interpolation, merging
│   │   ├── models.py               # Pydantic frozen models (all config types)
│   │   └── registry.py             # global config registry
│   │
│   ├── providers/                  # ✅ COMPLETE
│   │   ├── base.py                 # LLMProvider protocol, LLMRequest/Response
│   │   ├── anthropic.py            # Messages API + Batch API
│   │   ├── openai.py               # Chat Completions + Batch API
│   │   ├── google.py               # Gemini GenerateContent
│   │   ├── ollama.py               # OpenAI-compatible local API
│   │   └── factory.py              # create_provider() from config
│   │
│   ├── agents/                     # ✅ COMPLETE
│   │   ├── base.py                 # BaseAgent with _call_llm() (includes tracing)
│   │   ├── designer.py             # DesignerAgent (generate + revise)
│   │   ├── reviewer.py             # ReviewerAgent (critique)
│   │   ├── specialist.py           # SpecialistReviewer (domain-specific)
│   │   ├── adversary.py            # AdversarialReviewer (reject/accept)
│   │   ├── merger.py               # MergerAgent (synthesize)
│   │   └── judge.py                # JudgeAgent (debate adjudicator)
│   │
│   ├── orchestrator/               # ✅ COMPLETE
│   │   ├── engine.py               # OrchestratorEngine + ExperimentRunner
│   │   ├── context.py              # RunContext, DesignArtifact, ReviewArtifact, LLMCallTrace
│   │   ├── factory.py              # create_variant_orchestrator() + instantiate_agents()
│   │   └── variants/
│   │       ├── base.py             # VariantOrchestrator ABC
│   │       ├── v1_baseline.py      # Single agent self-review
│   │       ├── v2_leader_reviewers.py  # Leader + K parallel reviewers
│   │       ├── v3_parallel_merge.py    # K parallel → merge (3 strategies)
│   │       ├── v4_adversarial.py       # Leader + adversarial gate
│   │       ├── v5_specialist_panel.py  # Leader + N specialist reviewers
│   │       ├── v6_rotating_leader.py   # Round-robin leadership
│   │       ├── v7_consensus.py         # Parallel → share → converge
│   │       └── v8_structured_debate.py # Debate → judge → synthesize
│   │
│   ├── evaluation/                 # ✅ COMPLETE
│   │   └── pipeline.py             # EvaluationPipeline: 3× eval, median, alpha, disagreement
│   │
│   ├── batch/                      # ✅ COMPLETE
│   │   └── runner.py               # BatchCollector + BatchExperimentRunner
│   │
│   ├── prompts/                    # ✅ COMPLETE
│   │   └── renderer.py             # Jinja2 with StrictUndefined, custom filters
│   │
│   ├── storage/                    # ✅ COMPLETE
│   │   ├── database.py             # SQLite wrapper, schema DDL, init, stats
│   │   └── migrations.py           # schema versioning (basic)
│   │
│   ├── cli/                        # ✅ COMPLETE
│   │   ├── main.py                 # Typer app: run, validate, db subcommands
│   │   ├── run.py                  # single, experiment, evaluate, batch, status
│   │   ├── validate.py             # config validation
│   │   └── db.py                   # db init, stats, export
│   │
│   ├── observability/              # 🔴 EMPTY (only __init__.py)
│   │   └── __init__.py
│   │
│   └── analysis/                   # 🔴 EMPTY (only __init__.py)
│       └── __init__.py
│
├── docs/
│   ├── IMPLEMENTATION_PLAN.md      # this file
│   ├── thesis_proposal_v3.docx
│   ├── thesis_proposal_v3.docx.md
│   └── llm_consortium_taxonomy.jsx
│
├── data/                           # runtime data (gitignored)
│   └── consortium.db
│
├── tests/                          # 🔴 NOT IMPLEMENTED
└── scripts/                        # 🔴 NOT IMPLEMENTED
```

### Key Architecture Decisions (Implemented)

1. **Config system:** Pydantic frozen models → YAML with `${ENV_VAR:-default}` interpolation → global registry. Zero hardcoded values.

2. **Provider abstraction:** `LLMProvider` protocol with `complete()`, `complete_batch()`, `supports_batch()`, `estimate_cost()`. Factory creates providers from `ModelConfig`. Each provider handles retry, cost calculation, and batch submission internally.

3. **Agent hierarchy:** `BaseAgent._call_llm()` handles: request construction → provider invocation → trace recording → limit checking. All 7 agent types inherit from this. Tracing is automatic — no separate tracing middleware needed.

4. **Variant orchestrators:** Factory registry pattern. Each variant reads its YAML config and wires agents accordingly. Sub-variants are config-driven branches (e.g., v3 reads `sub_variant` to select merge template).

5. **Database:** SQLite with WAL mode. Schema directly in `database.py` (no ORM). Tables: runs, designs, reviews, evaluations, scores_median, coherence_checks, traces, batches. All with proper indexes.

6. **Engine:** `OrchestratorEngine.run()` handles: config resolution → agent instantiation → variant execution → DB persistence → checkpoint tracking. `ExperimentRunner` handles the full variant×task×rep matrix with skip-if-done, cost limits, and resume.

7. **Evaluation:** `EvaluationPipeline` calls evaluator LLM N times per design, parses JSON scores, computes medians, detects disagreement (range > 1.5), and computes Krippendorff's alpha proxy.

8. **Batch layer:** `BatchCollector` (deferred execution with `asyncio.Future`) + `BatchExperimentRunner` (concurrent runs with semaphore). Providers handle actual batch API submission.

9. **Templates:** All templates use `design_type` to branch between system/application language. Templates receive `rubric_dimensions` for rubric-aware prompts. Coverage matrix maps dimensions → design sections.

10. **Task balance:** 4 system design tasks (t1, t4, t6, t7) + 4 application design tasks (t2, t3, t5, t8), each with corresponding rubric.

---

## 2. Phase 3 — Remaining Work

### Priority-Ordered Task List

| Priority | Task | Module | Estimated Effort | Dependencies |
|----------|------|--------|------------------|--------------|
| **P0** | Pilot run (v1+v2, t1+t5, 2 reps) | CLI + engine | 1 day | API keys configured |
| **P1** | Observability: timeline builder | `observability/timeline.py` | 2 days | traces in DB |
| **P1** | Observability: exporters | `observability/exporter.py` | 1 day | timeline.py |
| **P1** | CLI: `consortium timeline` command | `cli/timeline.py` | 0.5 day | exporter.py |
| **P2** | Coherence checker integration | `evaluation/pipeline.py` | 1 day | evaluation pipeline |
| **P2** | Analysis: statistical tests | `analysis/statistics.py` | 2 days | scipy, pandas |
| **P2** | Analysis: Pareto frontier | `analysis/pareto.py` | 1 day | plotly |
| **P2** | Analysis: heatmaps | `analysis/heatmaps.py` | 1 day | seaborn |
| **P2** | Analysis: decision framework gen | `analysis/framework.py` | 2 days | statistics.py |
| **P2** | CLI: `consortium analyze` command | `cli/analyze.py` | 1 day | analysis modules |
| **P3** | Evaluator validation (alpha, human) | manual + scripts | 2 days | pilot data |
| **P3** | Full experiment execution (320 runs) | scripts | 3-5 days | pilot validated |
| **P3** | Unit tests | `tests/unit/` | 3 days | — |
| **P3** | Integration tests | `tests/integration/` | 2 days | mock providers |
| **P3** | Convenience scripts | `scripts/` | 0.5 day | — |

---

## 3A. Observability & Timeline

### Files to Create

```
src/consortium/observability/
├── __init__.py          # exists
├── timeline.py          # NEW: TimelineBuilder
├── exporter.py          # NEW: JSON, HTML, Chrome trace, CSV export
└── dashboard.py         # NEW (optional): Rich console or Streamlit dashboard
```

### timeline.py — TimelineBuilder

```python
class TimelineBuilder:
    """Builds timeline views from LLMCallTrace records in the database.
    
    Queries the traces table and produces structured timeline data
    that can be rendered in multiple formats.
    """
    
    def __init__(self, database: Database):
        self.database = database
    
    def build_run_timeline(self, run_id: str) -> RunTimeline:
        """Build timeline for a single run."""
        # Query traces for this run, ordered by started_at
        # Group by agent_id to create swim lanes
        # Identify parallel calls (overlapping time ranges)
        ...
    
    def build_variant_summary(self, variant_id: str, task_id: str | None = None) -> VariantSummary:
        """Aggregate timeline stats across runs for a variant."""
        # Mean/median duration per step
        # Cost breakdown by agent role
        # Token usage by step
        ...
    
    def build_comparison(self, variant_ids: list[str], task_id: str) -> ComparisonView:
        """Compare timelines across variants for the same task."""
        ...
```

### exporter.py — Export Formats

| Format | Method | Use Case |
|--------|--------|----------|
| Rich console | `export_rich(timeline)` | Terminal debugging |
| JSON | `export_json(timeline)` | Programmatic analysis |
| Chrome Trace | `export_chrome_trace(timeline)` | `chrome://tracing` interactive |
| HTML | `export_html(timeline)` | Standalone shareable report |
| CSV | `export_csv(timeline)` | Spreadsheet import |

### CLI Command

```bash
consortium timeline --run-id <id>                    # single run
consortium timeline --variant v3 --task t5           # aggregate
consortium timeline --task t5 --compare v1 v2 v3     # comparison
consortium timeline --summary cost-by-variant        # cost summary
consortium timeline --format json --output out.json  # export
```

### cli/timeline.py

```python
app = typer.Typer()

@app.command()
def show(
    run_id: str | None, variant: str | None, task: str | None,
    compare: list[str] | None, summary: str | None,
    format: str = "rich", output: Path | None = None,
): ...
```

Wire into `main.py` as `app.add_typer(timeline_app, name="timeline")`.

---

## 3B. Analysis & Visualization

### Files to Create

```
src/consortium/analysis/
├── __init__.py          # exists
├── statistics.py        # NEW: Friedman, Wilcoxon, effect sizes
├── pareto.py            # NEW: cost-quality Pareto frontier
├── heatmaps.py          # NEW: variant × dimension heatmaps
└── framework.py         # NEW: §10.2 decision framework generator
```

### statistics.py — Statistical Tests

| Analysis | Method | Function |
|----------|--------|----------|
| Variant ranking (overall) | Friedman test + Nemenyi post-hoc | `rank_variants_overall()` |
| Variant ranking (per complexity) | Stratified Friedman | `rank_variants_by_complexity()` |
| v1 vs each variant | Wilcoxon signed-rank (paired) | `compare_to_baseline()` |
| Pairwise comparisons | Wilcoxon + Bonferroni correction | `pairwise_comparisons()` |
| Effect of authority axis | centralized vs decentralized | `analyze_axis_effect("authority")` |
| Effect of roles axis | homogeneous vs specialized | `analyze_axis_effect("roles")` |
| Effect of dynamics axis | cooperative vs adversarial | `analyze_axis_effect("dynamics")` |
| Interaction effects | 3-way ANOVA-like on 2×2×2 | `interaction_effects()` |
| Variance analysis | CV per variant | `variance_analysis()` |
| Token efficiency | Quality per 1K tokens | `token_efficiency()` |

All functions take a DataFrame of scores from the `scores_median` table joined with `runs`.

### pareto.py — Cost-Quality Pareto

```python
def compute_pareto_frontier(
    scores_df: pd.DataFrame,
    cost_col: str = "total_cost_usd",
    quality_col: str = "overall_median",
) -> pd.DataFrame:
    """Identify Pareto-optimal variants (max quality, min cost)."""
    ...

def plot_pareto(
    scores_df: pd.DataFrame,
    output_path: Path,
    interactive: bool = True,  # Plotly HTML vs static PNG
) -> None:
    """Generate Pareto frontier chart."""
    ...
```

### heatmaps.py — Heatmap Generation

```python
def variant_dimension_heatmap(scores_df: pd.DataFrame, output_path: Path) -> None:
    """Variant × rubric dimension median scores heatmap (seaborn)."""
    ...

def complexity_variant_heatmap(scores_df: pd.DataFrame, output_path: Path) -> None:
    """Complexity × variant median scores heatmap."""
    ...
```

### framework.py — Decision Framework (§10.2)

```python
def generate_decision_framework(
    scores_df: pd.DataFrame,
    costs_df: pd.DataFrame,
    output_dir: Path,
) -> DecisionFramework:
    """Generate the §10.2 practitioner decision framework.
    
    For each (complexity_level, budget_priority, quality_priority):
    1. Filter results to matching complexity
    2. Rank variants by quality (weighted rubric median)
    3. Rank by cost-efficiency (quality / cost)
    4. Apply budget constraint
    5. Select Pareto-optimal variant
    6. Report expected quality range (IQR from experiment data)
    
    Outputs:
    - data/exports/tables/decision_framework.csv
    - data/exports/reports/decision_framework.md
    """
    ...

def validate_predictions(
    framework: DecisionFramework,
    predictions: list[dict],
) -> PredictionReport:
    """Compare empirical framework against §10.1 predicted findings."""
    ...
```

### CLI Command

```bash
consortium analyze --all                  # full suite
consortium analyze --ranking              # Friedman test
consortium analyze --pareto               # Pareto frontier
consortium analyze --heatmap              # heatmaps
consortium analyze --framework            # §10.2 generation
consortium analyze --predictions          # validate §10.1
consortium analyze --output-dir exports/  # output location
```

### cli/analyze.py

```python
app = typer.Typer()

@app.command()
def all(output_dir: Path = Path("data/exports")): ...

@app.command()
def ranking(output_dir: Path = Path("data/exports")): ...

@app.command()
def pareto(output_dir: Path = Path("data/exports"), interactive: bool = True): ...

@app.command()
def heatmap(output_dir: Path = Path("data/exports")): ...

@app.command()
def framework(output_dir: Path = Path("data/exports")): ...

@app.command()
def predictions(output_dir: Path = Path("data/exports")): ...
```

Wire into `main.py` as `app.add_typer(analyze_app, name="analyze")`.

### Output Artifacts

```
data/exports/
├── tables/
│   ├── variant_ranking_overall.csv
│   ├── variant_ranking_by_complexity.csv
│   ├── pairwise_significance.csv
│   ├── axis_effects.csv
│   └── decision_framework.csv
├── figures/
│   ├── pareto_frontier.html            # interactive Plotly
│   ├── pareto_frontier.png             # static for thesis
│   ├── heatmap_variant_dimension.png
│   ├── heatmap_complexity_variant.png
│   ├── cd_diagram_overall.png          # critical difference diagram
│   ├── cost_breakdown_by_variant.png
│   └── variance_comparison.png
└── reports/
    ├── pilot_report.md
    ├── evaluator_validation.md
    ├── full_analysis_report.md
    └── decision_framework.md
```

---

## 3C. Coherence Checker Integration

The coherence check template (`prompts/evaluation/coherence_check.j2`) already exists.
The `coherence_checks` DB table exists. What's missing is the pipeline integration.

### Add to `evaluation/pipeline.py`:

```python
async def run_coherence_checks(
    self,
    design_id: str,
    design_text: str,
    section_pairs: list[tuple[str, str]],
) -> list[CoherenceResult]:
    """Check internal consistency between design sections.
    
    For each pair of sections, asks the evaluator:
    "Do these two sections contradict each other?"
    
    Results stored in coherence_checks table.
    """
    ...
```

### Add to `cli/run.py`:

```python
@app.command()
def coherence(
    run_id: str | None = None,
    force: bool = False,
    batch: bool = False,
): ...
```

### Section pairs from `evaluator.yaml`:

The evaluator config already defines:
```yaml
coherence_check:
  enabled: true
  section_pairs:
    - ["data_architecture", "scalability_strategy"]
    - ["api_design", "security_design"]
    - ["error_handling", "trade_off_awareness"]
```

But the `EvaluatorConfig` model doesn't have `section_pairs` — it only has `CoherenceCheckConfig` with `enabled` and `prompt_template`. Need to add `section_pairs: list[list[str]]` to the config model.

---

## 3D. Pilot Runs & Evaluator Validation

### Pilot Scope

2 variants × 2 tasks × 2 reps = **8 runs**:

| | T1 (Simple, System) | T5 (Complex, Application) |
|---|---|---|
| **v1** (baseline) | 2 reps | 2 reps |
| **v2** (leader + reviewers) | 2 reps | 2 reps |

### Pilot Checklist

| # | Check | Pass Criteria |
|---|-------|---------------|
| P1 | Config loads without errors | `consortium validate` passes |
| P2 | v1 produces a coherent design | Manual inspection |
| P3 | v2 reviewer feedback is actionable | Reviews cite rubric dimensions |
| P4 | v2 revision improves on v1 | ≥1 dimension score improves |
| P5 | Evaluator scores parse correctly | All 3 runs produce valid JSON |
| P6 | Evaluator consistency | Krippendorff's α ≥ 0.7 |
| P7 | Token accounting matches API | Within 5% of reported |
| P8 | Cost accounting is accurate | Within 10% of actual |
| P9 | Timeline renders correctly | `consortium timeline` shows all runs |
| P10 | Checkpoint/resume works | Kill mid-run, resume, same result |
| P11 | Batch evaluation works | 8 × 3 = 24 evaluations succeed |
| P12 | Results stored in DB | `consortium db stats` shows all |

### Evaluator Validation

1. **Intra-rater reliability:** 3 evaluator runs per design → Krippendorff's α ≥ 0.7
2. **Inter-model agreement:** Sonnet 4.5 vs GPT-4.1 on 20 designs → Spearman ρ ≥ 0.8
3. **Human validation:** Researcher manually scores 20 designs → Spearman ρ ≥ 0.75

### Pilot Script

```bash
# scripts/pilot.sh
#!/bin/bash
set -euo pipefail

echo "=== Pilot Run ==="
consortium validate
consortium db init
consortium run experiment --variants v1,v2 --tasks t1,t5 --reps 2 --resume
consortium run evaluate --force
consortium run status
consortium timeline --summary cost-by-variant
echo "=== Pilot Complete ==="
```

---

## 3E. Full Experiment Execution

```bash
# scripts/full_experiment.sh
#!/bin/bash
set -euo pipefail

# Freeze config
git tag experiment-config-freeze
consortium validate --strict

# Run all 320 runs
consortium run experiment --resume 2>&1 | tee data/run_log.txt

# Monitor
consortium run status

# Evaluate all designs (3× each = 960 evaluations)
consortium run evaluate --force

# Coherence checks
consortium run coherence --all

# Verify completeness
consortium db stats

# Run analysis
consortium analyze --all --output-dir data/exports/

# Generate decision framework
consortium analyze --framework --output-dir data/exports/
```

### Cost Tracking

Expected costs (all batch API, 50% discount):
- Generation: ~$200 (320 runs × avg $0.63/run)
- Evaluation: ~$108 (960 evaluations × avg $0.11/eval)
- Coherence: ~$20 (320 × 3 pairs × ~$0.02/check)
- **Total: ~$330** (budget: $500)

---

## 3F. Decision Framework Generation (§10.2)

The framework generator (`analysis/framework.py`) produces the thesis's key deliverable:

```
┌─────────────┬──────────┬──────────┬────────────────────┬─────────────────┬────────────┐
│ Complexity  │ Budget   │ Quality  │ Recommended        │ Expected        │ Cost/Run   │
│             │ Priority │ Priority │ Variant            │ Quality (IQR)   │ (USD)      │
├─────────────┼──────────┼──────────┼────────────────────┼─────────────────┼────────────┤
│ Simple      │ Low      │ Good     │ v1b                │ 3.52–3.98       │ $0.22      │
│ Simple      │ Any      │ Maximum  │ v2b                │ 3.78–4.21       │ $0.46      │
│ Medium      │ Moderate │ High     │ v5                 │ 3.81–4.32       │ $0.46      │
│ Complex     │ Moderate │ High     │ v3b (rubric merge) │ 4.03–4.41       │ $0.88      │
│ Complex     │ Any      │ Maximum  │ v3c (dialectical)  │ 4.18–4.67       │ $0.88      │
│ Complex     │ Any      │ Robust   │ v8 (debate)        │ 4.15–4.62       │ $0.88      │
└─────────────┴──────────┴──────────┴────────────────────┴─────────────────┴────────────┘
```

Note: Values above are *predictions* from the thesis proposal. Actual values will come from experiment data.

### Prediction Validation

Compare empirical results against §10.1 predictions:
- P1: v1b matches consortium on simple tasks
- P2: v3 highest peak on complex tasks
- P3: v7 lowest variance
- P4: v4 highest variance
- P5: v5 outperforms v2 on specialist dimensions
- P6: Rubric > topology effect

---

## 3G. Tests

### Unit Tests

```
tests/unit/
├── test_config_loader.py           # YAML loading, env interpolation, validation
├── test_config_models.py           # Pydantic model validation edge cases
├── test_prompt_renderer.py         # Template rendering with all variable combos
├── test_providers.py               # Mock API round-trips for all 4 providers
├── test_agents.py                  # Agent _call_llm with mock provider
├── test_scoring.py                 # JSON extraction, median, alpha computation
├── test_batch_collector.py         # BatchCollector enqueue/flush
└── test_database.py                # Schema init, CRUD operations
```

### Integration Tests

```
tests/integration/
├── test_variant_v1.py              # Full v1 pipeline with mock provider
├── test_variant_v2.py              # v2 with parallel reviewers
├── test_variant_v3.py              # v3 with merge strategies
├── test_evaluation_pipeline.py     # 3× evaluation with mock evaluator
├── test_checkpoint_resume.py       # Kill mid-run, verify resume
└── test_experiment_runner.py       # Small matrix run end-to-end
```

### Test Fixtures

```
tests/fixtures/
├── sample_design.md                # A known-good design for evaluation tests
├── sample_evaluation.json          # Expected evaluator output
├── mock_responses/                 # Pre-recorded LLM responses per variant
│   ├── v1_generation.json
│   ├── v2_review.json
│   └── ...
└── mini_config/                    # Minimal config set for testing
    ├── experiment.yaml
    ├── models/test_model.yaml
    ├── variants/v1_test.yaml
    └── tasks/t1_test.yaml
```

---

## 4. Database Schema

(Unchanged from implementation — see `src/consortium/storage/database.py` for full DDL.)

Tables: `runs`, `designs`, `reviews`, `evaluations`, `scores_median`, `coherence_checks`, `traces`, `batches`, `schema_version`.

---

## 5. CLI Reference (Current + Planned)

```
consortium — LLM Consortium Experiment Framework

Commands (✅ = implemented, 🔴 = Phase 3):
  ✅ validate    Validate configuration files
  ✅ db          Database operations (init, stats, export)
  ✅ run         Execute experiments
     ✅ single      Run one variant×task×rep
     ✅ experiment   Run full matrix
     ✅ evaluate     Score completed designs
     ✅ batch        Run with batch APIs
     ✅ status       Show progress
     🔴 coherence   Run coherence checks
  🔴 timeline    View LLM call timelines
  🔴 analyze     Statistical analysis & visualization
     🔴 all          Full analysis suite
     🔴 ranking      Friedman test + rankings
     🔴 pareto       Cost-quality Pareto frontier
     🔴 heatmap      Variant × dimension heatmaps
     🔴 framework    Generate §10.2 decision framework
     🔴 predictions  Validate §10.1 predictions
```

---

## 6. Config Reference

### Environment Variables

```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
OLLAMA_HOST=http://nagarjunas-Mac-Studio.local:11434
```

### Config Resolution Order

1. Pydantic defaults (lowest)
2. YAML files in `configs/`
3. Environment variables via `${VAR:-default}`
4. CLI flags (highest)

---

## Phase 3 Summary for Handoff

**What's built:** Complete experiment pipeline — config, providers, agents, 8 variant orchestrators, engine, evaluation, batch, CLI. Ready to run experiments.

**What's needed:**

1. **Observability** (`observability/timeline.py`, `exporter.py`) — timeline builder + multi-format export + CLI command
2. **Analysis** (`analysis/statistics.py`, `pareto.py`, `heatmaps.py`, `framework.py`) — Friedman, Wilcoxon, Pareto, heatmaps, §10.2 generator + CLI command
3. **Coherence checker** — integrate into evaluation pipeline, add `section_pairs` to config model
4. **Pilot run** — execute 8 runs, validate evaluator, fix any bugs
5. **Full experiment** — 320 runs + 960 evaluations + coherence checks
6. **Tests** — unit + integration test suites
7. **Scripts** — pilot.sh, full_experiment.sh, setup.sh

**Estimated effort:** 2-3 weeks for code, then 1 week for experiment execution.
