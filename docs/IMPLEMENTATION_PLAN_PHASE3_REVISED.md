<!-- See /mnt/user-data/outputs/IMPLEMENTATION_PLAN_PHASE3_REVISED.md for full content -->
<!-- Copying full content to project docs -->
# Phase 3 Implementation Plan — Revised

> **Revision notes:** This plan supersedes the original Phase 3 proposal. It incorporates lessons from the first live run attempt (Feb 2025), where 3 of 8 variant orchestrators had critical bugs in feedback flow, the Anthropic provider rejected dual temperature/top_p, and .env loading was missing. These fixes are now merged but **no variant has been validated end-to-end except v1 t1 (post-fix)**. This plan treats data integrity and variant correctness as P0 prerequisites before any analysis work.

---

## Phase 3 Execution Order

```
P0: Smoke Tests → Clean Pilot
P1: Observability (3A)
P1: Coherence Checker (3C)
P2: Analysis & Visualization (3B)
P2: Decision Framework (3F)
P3: Full Experiment (3E)
P3: Scripts & Automation (3H)
P3: Unit + Integration Tests (3G)
```

**Rationale:** We must validate that all 8 variants produce correct prompt→feedback→revision flows before generating any data. Observability goes first because it's the debugging tool for everything else. Coherence checking is cheap and validates design quality. Analysis is pointless until we have scored data. The full experiment is the most expensive step and goes last.

---

## P0: Smoke Tests & Clean Pilot (prerequisite for everything)

### 0.1 — Variant Smoke Test Suite

**Problem:** v1, v7, and v8 had critical bugs (no review feedback, missing template vars, template/var mismatches). v2–v6 appeared correct on code review but have never been executed. We cannot trust any variant until it has been run at least once and the traces inspected.

**Action:** Run each variant once on the cheapest task (t1) and verify traces:

```bash
# For each variant v1..v8:
consortium run single <variant> t1 --rep 0 --force

# Then verify:
sqlite3 data/consortium.db "
  SELECT step, round, 
         CASE WHEN prompt_text LIKE '%Previous Review Feedback%' THEN 'HAS_FEEDBACK' 
              WHEN prompt_text LIKE '%Opposing position%' THEN 'HAS_POSITIONS'
              WHEN prompt_text LIKE '%Design Under Review%' THEN 'HAS_DESIGN'
              ELSE 'GENERATION_ONLY' END as prompt_type,
         length(response_text) as response_len
  FROM traces 
  WHERE run_id IN (SELECT run_id FROM runs WHERE variant_id='<variant>' AND task_id='t1')
  ORDER BY started_at
"
```

**Expected trace patterns per variant:**

| Variant | Expected Steps | Key Verification |
|---------|---------------|------------------|
| v1 | gen → review → revision → review → revision | Revision prompts contain `Previous Review Feedback` |
| v2 | gen → review ×K → revision → review ×K → revision | Reviews appear in revision prompt |
| v3 | gen ×K (parallel) → merge | Merge prompt contains all K designs |
| v4 | gen → review → (revision → review)* or ACCEPT | `VERDICT: ACCEPT/REJECT` in review response |
| v5 | gen → specialist_review ×N → revision → ... | Specialist focus dimensions in prompts |
| v6 | gen → review ×(K-1) → revision → review ×(K-1) → revision | Different leader each round |
| v7 | gen ×K → revision ×K → convergence_check → ... | `STATUS: CONVERGED/NOT_CONVERGED` in check |
| v8 | position ×K → rebuttal ×K → judge | Position/rebuttal use correct templates |

**Exit criteria:** All 8 variants produce traces matching expected patterns. Any failure → fix before proceeding.

### 0.2 — Evaluation Pipeline Smoke Test

```bash
consortium run evaluate --force
```

Verify:
- All 8 designs get 3 evaluator scores each
- `scores_median` table has 8 rows
- No JSON parse failures in logs

### 0.3 — Known Issues to Watch For

| Issue | Symptom | Fix Location |
|-------|---------|-------------|
| Anthropic temperature + top_p | 400 error on eval calls | `providers/anthropic.py` (already fixed) |
| v5 specialist `focus_dimensions` matching | Specialist template filters by `dim.id in focus_dimensions` — verify dim IDs in rubric match specialist config | `configs/variants/v5_specialist_panel.yaml` |
| v7 convergence never triggers | LLM doesn't output `STATUS: CONVERGED` in expected format | `consensus_synthesis.j2` — may need stricter formatting instructions |
| v4 adversary always accepts round 1 | fail-open default in `parse_verdict()` | `agents/adversary.py` — log warning if unparseable |
| v8 rebuttal template `{{ round }}` | Jinja2 `round` shadows built-in filter | Template var name collision — rename to `round_num` if needed |

---

## 3A — Observability & Timeline

### Files to Create

```
src/consortium/observability/
├── __init__.py          (exists)
├── timeline.py          NEW
└── exporter.py          NEW

src/consortium/cli/
├── timeline.py          NEW
```

### 3A.1 — timeline.py

```python
@dataclass
class TraceEvent:
    trace_id: str
    run_id: str
    agent_id: str
    step: str          # generation, review, revision, merge, etc.
    round_num: int
    started_at: datetime
    ended_at: datetime
    duration_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    model: str
    prompt_hash: str
    # Optional: prompt_text and response_text (large, load on demand)

@dataclass
class SwimLane:
    agent_id: str
    role: str
    events: list[TraceEvent]

@dataclass
class RunTimeline:
    run_id: str
    variant_id: str
    task_id: str
    total_duration_ms: float
    total_cost_usd: float
    total_tokens: int
    swim_lanes: list[SwimLane]
    parallel_groups: list[list[str]]  # groups of trace_ids that overlapped

@dataclass
class VariantSummary:
    variant_id: str
    task_id: str | None
    run_count: int
    mean_duration_ms: float
    mean_cost_usd: float
    mean_tokens: int
    cost_breakdown_by_step: dict[str, float]   # step → mean cost
    token_breakdown_by_step: dict[str, float]

@dataclass
class ComparisonView:
    task_id: str
    variants: list[VariantSummary]
```

**`TimelineBuilder` methods:**

- `build_run_timeline(run_id: str) → RunTimeline` — queries traces ordered by `started_at`, groups by agent_id, detects parallelism via overlapping `[started_at, ended_at]` intervals
- `build_variant_summary(variant_id: str, task_id: str | None = None) → VariantSummary` — aggregates across repetitions
- `build_comparison(variant_ids: list[str], task_id: str) → ComparisonView`

### 3A.2 — exporter.py

| Format | Method | Output | Use Case |
|--------|--------|--------|----------|
| Rich console | `export_rich(timeline)` | Printed table with color | Terminal debugging |
| JSON | `export_json(timeline, path)` | `.json` file | Programmatic analysis |
| Chrome Trace | `export_chrome_trace(timeline, path)` | Chrome tracing `.json` | `chrome://tracing` interactive view |
| CSV | `export_csv(timeline, path)` | `.csv` file | Spreadsheet import |

**Chrome Trace format** (most useful for debugging parallel variants):

```json
[
  {"name": "generation", "cat": "llm", "ph": "X", "ts": <epoch_us>, "dur": <us>,
   "pid": 1, "tid": <agent_id_hash>,
   "args": {"agent_id": "designer_0", "model": "gpt-4.1", "tokens": 4521, "cost": 0.04}}
]
```

### 3A.3 — CLI: `consortium timeline`

```bash
consortium timeline show --run-id <id>                    # single run
consortium timeline show --variant v3 --task t5           # aggregate
consortium timeline show --task t5 --compare v1,v2,v3     # comparison
consortium timeline summary --by cost                     # cost/token summary
consortium timeline summary --by tokens                   # token breakdown
--format rich|json|csv|chrome   --output <path>
```

### Wire into main.py

```python
from consortium.cli.timeline import app as timeline_app
app.add_typer(timeline_app, name="timeline", help="LLM call timeline views")
```

---

## 3C — Coherence Checker Integration

### 3C.1 — Config Model Update

**Modify** `CoherenceCheckConfig` in `config/models.py`:

```python
class CoherenceCheckConfig(BaseModel, frozen=True):
    enabled: bool = True
    prompt_template: str = "evaluation/coherence_check.j2"
    section_pairs: list[list[str]] = Field(default_factory=list)  # NEW
```

**Runtime resolution:** If `evaluator.coherence_check.section_pairs` is non-empty, use it. Otherwise fall back to `rubric.coherence_pairs`. If both empty, skip coherence checking.

### 3C.2 — Pipeline Integration

**Add to** `evaluation/pipeline.py`:

```python
@dataclass
class CoherenceResult:
    check_id: str
    design_id: str
    section_a: str
    section_b: str
    contradicts: bool
    explanation: str
    confidence: float     # 0.0–1.0
    input_tokens: int
    output_tokens: int
    cost_usd: float

async def run_coherence_checks(
    self,
    design_id: str,
    design_text: str,
    section_pairs: list[tuple[str, str]],
) -> list[CoherenceResult]:
    """Check internal consistency between design sections.

    For each (section_a, section_b) pair:
    1. Render coherence_check.j2 with both sections extracted from design
    2. Call evaluator LLM
    3. Parse JSON response with defensive fallback
    4. Persist to coherence_checks table
    """
```

**Defensive JSON parsing** (lesson from adversary verdict parsing):

```python
def _parse_coherence_response(text: str) -> dict:
    """Parse coherence check response, with regex fallback."""
    try:
        return _extract_json(text)
    except ValueError:
        # Fallback: look for CONTRADICTS/CONSISTENT keyword
        if re.search(r"CONTRADICTS", text, re.IGNORECASE):
            return {"contradicts": True, "explanation": text, "confidence": 0.5}
        return {"contradicts": False, "explanation": text, "confidence": 0.5}
```

### 3C.3 — CLI: `consortium run coherence`

```bash
consortium run coherence                    # all completed, unevaluated
consortium run coherence --run-id <id>      # single run
consortium run coherence --all --force      # re-check everything
consortium run coherence --dry-run          # report what would be checked
```

---

## 3B — Analysis & Visualization

### Prerequisites

- `scores_median` table populated (from evaluation pipeline) ✅ exists
- `runs` table with `total_cost_usd`, `total_input_tokens`, `total_output_tokens` ✅ exists
- Dependencies: `pandas`, `scipy`, `seaborn`, `plotly` (add to `pyproject.toml`)

### Files to Create

```
src/consortium/analysis/
├── __init__.py          (exists)
├── loader.py            NEW — DataFrame construction from DB
├── statistics.py        NEW — statistical tests
├── pareto.py            NEW — Pareto frontier
├── heatmaps.py          NEW — heatmap generation
└── framework.py         NEW — §10.2 decision framework

src/consortium/cli/
├── analyze.py           NEW
```

### 3B.1 — loader.py (Data Loading)

**Why a separate loader:** Multiple analysis modules need the same DataFrame. Centralizing avoids repeated DB queries and ensures consistent joins/filters.

```python
def load_scores_dataframe(db: Database) -> pd.DataFrame:
    """Build the master analysis DataFrame.
    
    Joins: scores_median ← designs ← runs
    
    Columns:
        run_id, variant_id, sub_variant, task_id, repetition,
        complexity, design_type,
        overall_median, dimension_medians (exploded to one col per dim),
        krippendorff_alpha, disagreement_count,
        total_cost_usd, total_input_tokens, total_output_tokens,
        duration_seconds
    
    Handles:
        - Missing data: rows with NULL scores excluded + warning logged
        - Sparse matrix: reports which (variant, task) cells are missing
    """

def load_costs_dataframe(db: Database) -> pd.DataFrame:
    """Cost and token data from runs table, one row per run."""

def check_completeness(df: pd.DataFrame) -> CompletenessReport:
    """Check the variant × task × rep matrix for gaps.
    
    Returns:
        CompletenessReport with:
        - expected_runs: 8 × 8 × 5 = 320
        - actual_runs: count of rows
        - missing_cells: list of (variant, task, rep) tuples
        - coverage_pct: float
        - warnings: list[str] for Friedman requirements
    """
```

**Important:** Friedman test requires **complete blocks** — every variant must have a score for every task. If v8 failed on t3, either:
- Exclude t3 from the Friedman analysis entirely
- Exclude v8 from the comparison
- Document the gap

`check_completeness()` must be called before any statistical test and its warnings surfaced to the user.

### 3B.2 — statistics.py

All functions receive a DataFrame from `loader.py` and return typed result objects.

```python
@dataclass
class RankingResult:
    test_name: str                    # "Friedman"
    statistic: float
    p_value: float
    rankings: dict[str, float]        # variant → mean rank
    post_hoc: pd.DataFrame | None     # Nemenyi pairwise p-values
    significant: bool                 # p < 0.05
    n_tasks: int
    n_variants: int

@dataclass
class PairwiseResult:
    variant_a: str
    variant_b: str
    test_name: str                    # "Wilcoxon signed-rank"
    statistic: float
    p_value_raw: float
    p_value_corrected: float          # after Bonferroni
    effect_size: float                # r = Z / sqrt(N)
    significant: bool
    direction: str                    # "a > b" or "a < b" or "no difference"

@dataclass  
class AxisEffectResult:
    axis: str                         # "authority", "roles", "dynamics"
    group_a: str                      # e.g., "centralized"
    group_b: str                      # e.g., "decentralized"
    mean_a: float
    mean_b: float
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    significant: bool

def rank_variants_overall(df: pd.DataFrame) -> RankingResult:
    """Friedman test across all tasks + Nemenyi post-hoc if significant.
    
    Requires complete blocks: warns/errors if any (variant, task) cell is empty.
    Uses task-level median across repetitions as the observation.
    """

def rank_variants_by_complexity(df: pd.DataFrame) -> dict[str, RankingResult]:
    """Stratified Friedman: one test per complexity level (simple, medium, complex).
    Returns dict keyed by complexity."""

def compare_to_baseline(
    df: pd.DataFrame, baseline: str = "v1"
) -> list[PairwiseResult]:
    """Wilcoxon signed-rank: each variant vs v1 baseline.
    Bonferroni correction with m=7 comparisons."""

def pairwise_comparisons(df: pd.DataFrame) -> list[PairwiseResult]:
    """All-pairs Wilcoxon + Bonferroni correction.
    m = C(8,2) = 28 comparisons → α_corrected = 0.05/28 ≈ 0.0018."""

def analyze_axis_effect(
    df: pd.DataFrame, 
    axis: str,  # "authority" | "roles" | "dynamics"
) -> AxisEffectResult:
    """Mann-Whitney U test: group variants by 2×2×2 matrix axis.
    
    authority: centralized (v1,v2,v4,v5) vs decentralized (v3,v6,v7,v8)
    roles: homogeneous (v1,v2,v7,v8) vs specialized (v3,v4,v5,v6)  
    dynamics: cooperative (v1,v2,v3,v5,v6,v7) vs adversarial (v4,v8)
    
    Note: dynamics axis is unbalanced (6 vs 2). Report this limitation.
    """

def variance_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Coefficient of variation per variant. Tests prediction P3 (v7 lowest) 
    and P4 (v4 highest variance)."""

def token_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """Quality per 1K tokens: overall_median / (total_tokens / 1000).
    Identifies cost-effective variants."""
```

### 3B.3 — pareto.py

```python
def compute_pareto_frontier(
    df: pd.DataFrame,
    cost_col: str = "total_cost_usd",
    quality_col: str = "overall_median",
) -> pd.DataFrame:
    """Identify Pareto-optimal rows (max quality, min cost).
    
    A row is Pareto-optimal if no other row has both 
    higher quality AND lower cost.
    
    Returns: DataFrame with only Pareto-optimal rows + 
    'pareto_optimal' boolean column added to input df.
    """

def plot_pareto(
    df: pd.DataFrame,
    output_path: Path,
    interactive: bool = True,  # Plotly HTML vs matplotlib PNG
    color_by: str = "variant_id",
    facet_by: str | None = None,  # e.g., "complexity"
) -> None:
    """Generate Pareto frontier chart.
    
    X-axis: mean cost per run (from runs.total_cost_usd)
    Y-axis: mean overall quality (from scores_median.overall_median)
    Each point = one variant (aggregated across tasks/reps)
    Color = variant_id
    Pareto frontier drawn as connected line
    """
```

### 3B.4 — heatmaps.py

```python
def variant_dimension_heatmap(
    df: pd.DataFrame, 
    output_path: Path,
    annotate: bool = True,
) -> None:
    """8 variants × N dimensions heatmap.
    Cell value = mean median score across tasks and reps.
    Annotated with actual values. Seaborn with diverging colormap."""

def complexity_variant_heatmap(
    df: pd.DataFrame, 
    output_path: Path,
) -> None:
    """3 complexity levels × 8 variants heatmap.
    Cell value = mean overall_median for that (complexity, variant)."""

def task_variant_heatmap(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """8 tasks × 8 variants heatmap. Full experiment matrix."""
```

### 3B.5 — CLI: `consortium analyze`

```bash
consortium analyze all                        # full suite
consortium analyze doctor                     # check data completeness (NEW)
consortium analyze ranking                    # Friedman + post-hoc
consortium analyze baseline                   # vs v1 comparisons
consortium analyze pairwise                   # all-pairs Wilcoxon
consortium analyze axes                       # 2×2×2 axis effects
consortium analyze pareto                     # cost-quality frontier
consortium analyze heatmap                    # all heatmaps
consortium analyze efficiency                 # token efficiency
consortium analyze framework                  # §10.2 generation
consortium analyze predictions                # §10.1 validation
--output-dir data/exports/  --format png|html|csv|md
```

**`consortium analyze doctor`** (new — data completeness check):

```
$ consortium analyze doctor

Experiment Completeness Report
==============================
Expected:  320 runs (8 variants × 8 tasks × 5 reps)
Completed: 312 runs (97.5%)
Failed:    5 runs
Missing:   3 runs

Missing cells:
  v7 × t7 × rep3  (status: failed — convergence timeout)
  v7 × t7 × rep4  (status: failed — convergence timeout)
  v8 × t3 × rep2  (status: failed — judge parse error)

Evaluations:  936/960 (97.5%)
  Missing:    v7-t7-rep3, v7-t7-rep4, v8-t3-rep2 (no designs to evaluate)

Friedman test readiness:
  ⚠ t7: missing v7 data (2/5 reps) — will use available reps for median
  ⚠ t3: missing v8 data (1/5 reps) — will use available reps for median
  ✅ All other (variant, task) cells have ≥3 reps

Recommendations:
  → Run: consortium run single v7 t7 --rep 3 --force
  → Run: consortium run single v7 t7 --rep 4 --force
  → Run: consortium run single v8 t3 --rep 2 --force
```

---

## 3F — Decision Framework (§10.2)

> This is the thesis's primary deliverable. It deserves a real spec.

### What the Framework IS

A lookup table that answers: *"Given my task complexity, budget, and quality requirements, which consortium variant should I use?"*

### Input Data

1. **Quality matrix:** `scores_median.overall_median` and `scores_median.dimension_medians` per (variant, task, rep)
2. **Cost matrix:** `runs.total_cost_usd` per (variant, task, rep)
3. **Task metadata:** `tasks.complexity` (simple/medium/complex), `tasks.design_type` (system/application)

### Framework Generation Algorithm

```python
@dataclass
class FrameworkRow:
    complexity: str              # simple | medium | complex
    budget_priority: str         # low | moderate | any
    quality_priority: str        # good | high | maximum | robust
    recommended_variant: str
    expected_quality_iqr: tuple[float, float]  # 25th–75th percentile
    expected_cost_iqr: tuple[float, float]
    confidence: str              # high | medium | low
    evidence: str                # "Based on N runs, Wilcoxon p=X vs baseline"

@dataclass
class DecisionFramework:
    rows: list[FrameworkRow]
    pareto_variants: list[str]   # Pareto-optimal set
    generation_date: str
    n_runs_analyzed: int
    warnings: list[str]

def generate_decision_framework(
    scores_df: pd.DataFrame,
    costs_df: pd.DataFrame,
    output_dir: Path,
) -> DecisionFramework:
    """
    For each complexity level:
      1. Compute mean quality and cost per variant (across tasks of that complexity)
      2. Identify Pareto-optimal variants for that complexity
      3. Assign budget_priority buckets:
         - "low": cheapest Pareto-optimal variant
         - "moderate": best quality among cost < 2× cheapest
         - "any": highest quality regardless of cost
      4. Assign quality_priority:
         - "good": median ≥ 3.5
         - "high": median ≥ 4.0
         - "maximum": highest observed median
         - "robust": lowest CV (most consistent)
      5. Compute IQR from repetition data
      6. Compute confidence:
         - "high": Wilcoxon p < 0.01 vs next-best
         - "medium": p < 0.05
         - "low": p ≥ 0.05 (not significantly different from alternatives)
    
    Output:
      - data/exports/tables/decision_framework.csv
      - data/exports/reports/decision_framework.md
    """
```

### Prediction Validation (§10.1)

The thesis proposal makes 6 predictions. Validation is mechanical:

```python
@dataclass
class PredictionResult:
    prediction_id: str        # P1..P6
    description: str
    supported: bool
    p_value: float | None
    evidence: str

PREDICTIONS = {
    "P1": "v1 matches consortium quality on simple tasks",
    "P2": "v3 achieves highest peak quality on complex tasks",
    "P3": "v7 has lowest variance across repetitions",
    "P4": "v4 has highest variance across repetitions",
    "P5": "v5 outperforms v2 on specialist dimensions (security, scalability)",
    "P6": "Rubric awareness matters more than topology",
}

def validate_predictions(
    scores_df: pd.DataFrame,
    costs_df: pd.DataFrame,
) -> list[PredictionResult]:
    """Test each prediction against experiment data.
    
    P1: Wilcoxon v1 vs {v2..v8} on simple tasks → p > 0.05 means v1 is competitive
    P2: Rank v3 on complex tasks → must be rank 1 with significant difference
    P3: CV(v7) < CV(all others) → compare with bootstrap CI
    P4: CV(v4) > CV(all others)
    P5: v5 scores on security/scalability dims > v2 scores → Wilcoxon per dim
    P6: Correlation between rubric_in_prompt and quality > correlation between topology and quality
    """
```

---

## 3D — Pilot Runs

### Scope: 2 variants × 2 tasks × 2 reps = 8 runs

| | T1 (Simple, System) | T5 (Complex, Application) |
|---|---|---|
| **v1** (baseline) | 2 reps | 2 reps |
| **v2** (leader + reviewers) | 2 reps | 2 reps |

### Pilot Checklist

| # | Check | Pass Criteria | How to Verify |
|---|-------|---------------|---------------|
| P1 | Config loads | `consortium validate` passes | CLI |
| P2 | v1 self-review present | Revision prompts contain `Previous Review Feedback` | `traces.prompt_text` |
| P3 | v2 reviewer feedback in revision | Reviewer text appears in revision prompt | `traces.prompt_text` |
| P4 | Evaluator JSON parses | All 24 evals produce valid scores | No `evaluation_failed` in logs |
| P5 | Median computation correct | `scores_median` has 8 rows, all `overall_median` ∈ [1.0, 5.0] | DB query |
| P6 | Evaluator consistency | Krippendorff α ≥ 0.5 (relaxed for pilot) | `scores_median.krippendorff_alpha` |
| P7 | Cost tracking accurate | `runs.total_cost_usd` > 0 for all runs | DB query |
| P8 | Timeline renders | `consortium timeline show --variant v1 --task t1` works | CLI |
| P9 | Checkpoint/resume | Kill mid-run, resume, verify completion | Manual |
| P10 | No template errors | Zero `UndefinedError` in logs | Grep logs |

---

## 3E — Full Experiment Execution

### Scale: 8 variants × 8 tasks × 5 reps = 320 runs

### Execution Strategy

1. **Run by complexity tier** (cheap tasks first to catch issues early):
   - Tier 1: simple tasks (t1, t4) — 80 runs, ~$50
   - Tier 2: medium tasks (t2, t3) — 80 runs, ~$60
   - Tier 3: complex tasks (t5, t6, t7, t8) — 160 runs, ~$180

2. **Evaluate after each tier** — catch scoring issues before burning budget

3. **Use batch APIs where possible** — Anthropic and OpenAI batch APIs give 50% discount

### Cost Estimates

| Component | Runs | Cost/Run | Total |
|-----------|------|----------|-------|
| Generation (simple) | 80 | ~$0.40 | $32 |
| Generation (medium) | 80 | ~$0.50 | $40 |
| Generation (complex) | 160 | ~$0.75 | $120 |
| Evaluation (3× each) | 960 | ~$0.11 | $106 |
| Coherence (3 pairs each) | 960 | ~$0.02 | $19 |
| **Total** | | | **~$317** |
| **Budget** | | | **$500** |
| **Buffer** | | | **$183** (for retries, re-evaluations) |

### Error Handling & Resume

The `ExperimentRunner` already supports `--resume` (skip completed runs). Scripts must additionally handle:

```bash
# Check for and retry failed runs
consortium run status --failed
consortium run experiment --variants <failed_variants> --tasks <failed_tasks> --resume --force
```

---

## 3G — Tests

### Unit Tests

```
tests/unit/
├── test_config_loader.py           # YAML loading, env interpolation
├── test_config_models.py           # Pydantic validation edge cases
├── test_prompt_renderer.py         # Template rendering all variable combos
├── test_providers.py               # Mock API round-trips for all 4 providers
├── test_agents.py                  # Agent _call_llm with mock provider
├── test_scoring.py                 # JSON extraction, median, alpha computation
├── test_coherence_parser.py        # Defensive JSON + regex fallback (NEW)
├── test_batch_collector.py         # BatchCollector enqueue/flush
├── test_database.py                # Schema init, CRUD, migration v1→v2
├── test_timeline.py                # TimelineBuilder with mock DB (NEW)
├── test_statistics.py              # Friedman/Wilcoxon on synthetic data (NEW)
├── test_pareto.py                  # Pareto frontier edge cases (NEW)
├── test_framework.py               # Framework generation with synthetic data (NEW)
└── test_loader.py                  # DataFrame loading + completeness check (NEW)
```

### Integration Tests (NEW — P0 priority)

> **These would have caught every bug we hit in the first run.**

```
tests/integration/
├── conftest.py                     # Shared fixtures: mock provider, temp DB, mini config
├── test_variant_e2e.py             # Each variant end-to-end with mock provider
├── test_evaluation_e2e.py          # Generate → evaluate → scores_median
├── test_coherence_e2e.py           # Generate → coherence check
└── test_template_vars.py           # Verify every template renders without UndefinedError
```

**`test_variant_e2e.py`** — the most critical test:

```python
class MockProvider(LLMProvider):
    """Returns canned responses. Tracks all requests for assertion."""
    
    def __init__(self):
        self.requests: list[LLMRequest] = []
    
    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(
            content="Mock design with all required sections...",
            input_tokens=100, output_tokens=200,
            cost_usd=0.001, model="mock",
        )

@pytest.mark.parametrize("variant_id", ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"])
async def test_variant_produces_final_design(variant_id, tmp_db, mini_config):
    """Each variant must produce exactly one final design without errors."""
    engine = OrchestratorEngine(mini_config, tmp_db, prompts_dir)
    result = await engine.run(variant_id, "t1", repetition=0)
    
    assert result.status == "completed"
    assert any(d.is_final for d in result.designs)

@pytest.mark.parametrize("variant_id", ["v1", "v2", "v4", "v5"])
async def test_revision_includes_review_feedback(variant_id, tmp_db, mini_config):
    """Variants with review loops must pass feedback to revision prompts."""
    provider = MockProvider()
    # ... run variant ...
    
    revision_requests = [r for r in provider.requests if "Previous Review Feedback" in r.messages[0]["content"]]
    assert len(revision_requests) > 0, f"{variant_id} revision has no review feedback"
```

**`test_template_vars.py`** — prevent template/orchestrator mismatches:

```python
@pytest.mark.parametrize("template,required_vars", [
    ("generation/design_system.j2", ["system_name", "problem_statement", "hard_constraints", "use_cases"]),
    ("review/general_review.j2", ["design_text", "system_name", "complexity", "design_type"]),
    ("review/adversarial_review.j2", ["design_text", "system_name", "complexity", "design_type"]),
    ("review/specialist_review.j2", ["design_text", "specialty", "focus_dimensions"]),
    ("debate/position_assignment.j2", ["system_name", "complexity", "position", "perspective", "hard_constraints", "use_cases"]),
    ("debate/rebuttal.j2", ["round", "position", "perspective", "own_design", "other_positions"]),
    ("debate/judge_ruling.j2", ["positions", "system_name", "complexity"]),
    ("synthesis/merge_rubric_guided.j2", ["designs", "system_name", "complexity"]),
    ("synthesis/consensus_synthesis.j2", ["designs", "system_name", "complexity", "round"]),
])
def test_template_renders_with_required_vars(template, required_vars, renderer):
    """Every template must render without error when given its required variables."""
    vars = {v: f"mock_{v}" for v in required_vars}
    # Add list/dict types where needed
    for v in ["hard_constraints", "use_cases", "focus_dimensions"]:
        if v in vars:
            vars[v] = ["mock_item"]
    for v in ["designs", "positions", "other_positions"]:
        if v in vars:
            vars[v] = [{"agent_id": "mock", "text": "mock", "position": "mock", "perspective": "mock", "design": "mock", "final_design": "mock"}]
    for v in ["rubric_dimensions"]:
        if v in vars:
            vars[v] = [{"name": "mock", "weight": 1.0, "description": "mock", "anchors": {}}]
    
    result = renderer.render(template, **vars)
    assert len(result) > 0
```

### Test Fixtures

```
tests/fixtures/
├── mock_responses/
│   ├── design_response.txt         # Realistic design output
│   ├── review_response.txt         # Realistic review output
│   ├── evaluation_response.json    # Valid evaluator JSON
│   └── coherence_response.json     # Valid coherence JSON
├── mini_config/                    # Minimal config for testing
│   ├── experiment.yaml
│   ├── evaluator.yaml
│   ├── models/mock_model.yaml
│   ├── variants/v1_test.yaml
│   ├── tasks/t1_test.yaml
│   └── rubrics/test_rubric.yaml
└── sample_data/
    ├── sample_scores.csv           # Synthetic scores for statistics tests
    └── sample_costs.csv            # Synthetic costs for Pareto tests
```

---

## 3H — Scripts

### scripts/smoke_test.sh

```bash
#!/bin/bash
set -euo pipefail

echo "=== Variant Smoke Tests ==="
consortium validate

for v in v1 v2 v3 v4 v5 v6 v7 v8; do
    echo "--- Testing $v ---"
    consortium run single $v t1 --rep 0 --force 2>&1 | tail -5
    
    # Verify traces exist
    count=$(sqlite3 data/consortium.db "SELECT COUNT(*) FROM traces WHERE run_id IN (SELECT run_id FROM runs WHERE variant_id='$v' AND task_id='t1')")
    echo "$v: $count traces recorded"
    
    if [ "$count" -eq 0 ]; then
        echo "ERROR: $v produced no traces!"
        exit 1
    fi
done

echo "=== Evaluating all smoke test designs ==="
consortium run evaluate --force

echo "=== Smoke Tests Complete ==="
consortium db stats
```

### scripts/pilot.sh

```bash
#!/bin/bash
set -euo pipefail

echo "=== Pilot Run (v1,v2 × t1,t5 × 2 reps) ==="
consortium validate --strict
consortium db init

consortium run experiment --variants v1,v2 --tasks t1,t5 --reps 2 --resume 2>&1 | tee data/pilot_run.log

echo "=== Evaluating ==="
consortium run evaluate --force 2>&1 | tee -a data/pilot_run.log

echo "=== Coherence Checks ==="
consortium run coherence --all 2>&1 | tee -a data/pilot_run.log

echo "=== Timeline ==="
consortium timeline summary --by cost

echo "=== Data Check ==="
consortium analyze doctor

echo "=== Pilot Complete ==="
```

### scripts/full_experiment.sh

```bash
#!/bin/bash
set -euo pipefail

LOG_DIR="data/logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=== Full Experiment ==="
consortium validate --strict

# Tier 1: Simple tasks
echo "--- Tier 1: Simple Tasks ---"
consortium run experiment --tasks t1,t4 --resume 2>&1 | tee "$LOG_DIR/tier1_generation.log"
consortium run evaluate 2>&1 | tee "$LOG_DIR/tier1_evaluation.log"
consortium analyze doctor 2>&1 | tee "$LOG_DIR/tier1_doctor.log"

# Tier 2: Medium tasks
echo "--- Tier 2: Medium Tasks ---"
consortium run experiment --tasks t2,t3 --resume 2>&1 | tee "$LOG_DIR/tier2_generation.log"
consortium run evaluate 2>&1 | tee "$LOG_DIR/tier2_evaluation.log"
consortium analyze doctor 2>&1 | tee "$LOG_DIR/tier2_doctor.log"

# Tier 3: Complex tasks
echo "--- Tier 3: Complex Tasks ---"
consortium run experiment --tasks t5,t6,t7,t8 --resume 2>&1 | tee "$LOG_DIR/tier3_generation.log"
consortium run evaluate 2>&1 | tee "$LOG_DIR/tier3_evaluation.log"
consortium analyze doctor 2>&1 | tee "$LOG_DIR/tier3_doctor.log"

# Coherence
echo "--- Coherence Checks ---"
consortium run coherence --all 2>&1 | tee "$LOG_DIR/coherence.log"

# Retry failures
FAILED=$(consortium run status --failed --count 2>/dev/null || echo "0")
if [ "$FAILED" -gt 0 ]; then
    echo "--- Retrying $FAILED failed runs ---"
    consortium run experiment --resume --force 2>&1 | tee "$LOG_DIR/retry.log"
    consortium run evaluate 2>&1 | tee "$LOG_DIR/retry_eval.log"
fi

# Analysis
echo "--- Full Analysis ---"
consortium analyze all --output-dir data/exports/ 2>&1 | tee "$LOG_DIR/analysis.log"

echo "=== Experiment Complete ==="
consortium analyze doctor
```

---

## Output Artifacts

```
data/exports/
├── tables/
│   ├── variant_ranking_overall.csv
│   ├── variant_ranking_by_complexity.csv
│   ├── pairwise_significance.csv
│   ├── baseline_comparisons.csv
│   ├── axis_effects.csv
│   ├── variance_analysis.csv
│   ├── token_efficiency.csv
│   ├── completeness_report.csv
│   └── decision_framework.csv          # §10.2
├── figures/
│   ├── pareto_frontier.html             # interactive Plotly
│   ├── pareto_frontier.png              # static for thesis
│   ├── heatmap_variant_dimension.png
│   ├── heatmap_complexity_variant.png
│   ├── heatmap_task_variant.png
│   ├── cost_breakdown_by_variant.png
│   └── variance_comparison.png
└── reports/
    ├── completeness_report.md
    ├── pilot_report.md
    ├── full_analysis_report.md
    ├── prediction_validation.md         # §10.1
    └── decision_framework.md            # §10.2
```

---

## Dependencies to Add

```toml
# pyproject.toml — add to [project.dependencies]
pandas = ">=2.0"
scipy = ">=1.11"
seaborn = ">=0.13"
plotly = ">=5.18"
```

---

## Summary of Changes vs Original Plan

| Item | Original Plan | This Revision |
|------|---------------|---------------|
| **P0 smoke tests** | Not mentioned | Added as hard prerequisite |
| **`analyze doctor`** | Not mentioned | Added for data completeness checks |
| **Sparse data handling** | Not mentioned | `check_completeness()` + Friedman guard |
| **Integration tests** | Not mentioned | Added with MockProvider, template var tests |
| **`loader.py`** | Implied in statistics.py | Extracted as separate module |
| **Framework spec** | 1-line placeholder | Full algorithm + data structures |
| **Prediction validation** | 1-line placeholder | Full spec with test methods per prediction |
| **Coherence parser** | "Parse JSON" | Defensive parsing with regex fallback |
| **Scripts** | Linear happy-path | Tiered execution, error handling, retry logic |
| **Cost source for Pareto** | Not specified | Explicit: `runs.total_cost_usd` |
| **v8 template bugs** | Not mentioned | Documented in known issues + fixed |
| **Execution order** | 3A → 3C → 3B | P0 → 3A → 3C → 3B → 3F → 3E → 3H → 3G |