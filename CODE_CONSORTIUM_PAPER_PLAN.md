# LLM Consortium for Software Engineering — Paper #2 Plan

> Natural extension of *"LLM Consortium for Software Design Refinement"* (arXiv:2606.01490).
> Paper #1 measured **subjective design quality** via an LLM-judge ensemble. Paper #2 measures
> **objective, executable software engineering** via test execution — no LLM judge in the loop.

---

## 1. Thesis & Research Questions

**Thesis:** Do the multi-agent collaboration-topology findings from Paper #1 transfer from
subjective design quality to *objective* software engineering, where a solution either passes the
hidden tests or it doesn't?

- **RQ1 (transfer):** Does a model consortium beat the best single model (Sonnet-alone) on pass@1?
- **RQ2 (topology):** Do cross-model review (v2b) and structural-adversarial (v4b) still win on code?
  Does parallel-merge still break (expected to be *worse* — syntactically broken merges)?
- **RQ3 (cost-effectiveness):** pass@1-per-dollar — can a *cheap/free local* model (gpt-oss) lift a
  strong model into a Pareto-better region? (the "is it worth it" figure)
- **RQ4 (reliability):** does the consortium reduce run-to-run variance (Paper #1's CV analysis →
  pass@1 stability across reps)?
- **RQ5 (harness-consortium, novel):** on SWE-bench, can a consortium of *(harness × model)* agents
  (Claude Code, Codex CLI, Aider/SWE-agent …) beat any single harness+model? Turns the
  "harness-dependence" confound into the independent variable.

Why this is strong: it removes Paper #1's biggest criticism (LLM-as-judge subjectivity) by using
**test execution as ground truth**, and directly answers a practical question with money attached.

---

## 2. Locked Decisions

| Decision | Choice |
|---|---|
| **Headline benchmarks** | **LiveCodeBench** (function/competitive) · **BigCodeBench-Hard** (practical) · **SWE-bench Verified** (repo-level) |
| **Calibration anchor** | **HumanEval+** — repro check only (prove our harness matches published ~90%), not a headline result |
| **Model roster** | **Claude Sonnet 4.x**, **GPT-4.1**, **gpt-oss-120b (local Ollama / Vertex)** |
| **Baselines** | All 3 models solo (single-shot) → reproduce published pass@1 |
| **Consortium** | Heterogeneous topologies mixing the 3 families |
| **Exec feedback** | Each benchmark uses its **canonical protocol** (function-level = single-shot baselines so numbers match leaderboards; SWE-bench inherently uses execution/reproduction). Consortium adds *topology* within that protocol. |
| **Pilot scale** | Subsets (~50/benchmark), **2 reps**, core variants. Gate = single-model reproduces leaderboard before scaling. |
| **SWE-bench solver** | Agentless-style localize→repair→validate; consortium at the repair step |
| **Harness-consortium** | Sequenced last (Phase 4) — heavy integration & cost |

### Why these benchmarks (headroom check, June 2026 leaderboards)

The user's "pick ~70–80% or lower" instinct rules out HumanEval+ (Sonnet at ceiling: 97.6% orig,
~90% plus). Chosen suite has real headroom and a large model spread:

| Benchmark | Sonnet 4.x | GPT-4.1 | gpt-oss-120b |
|---|---|---|---|
| HumanEval+ (calibration) | ~88–92% | ~85% | ~80% |
| **LiveCodeBench** | ~55–64% | ~40–48% | ~58–60% |
| **BigCodeBench-Hard** | ~35–45% | ~35% | ~30% |
| **SWE-bench Verified** | 77.2% / 79.6% | 54.6% | low (~30–45%, harness-dep.) |

(Fallback: BigCodeBench-**Full** ~45–60% if Hard starves gpt-oss too much.)

---

## 3. Architecture: reuse vs. new

The framework is ~80% reusable. The **only** thing that fundamentally changes is evaluation:
LLM-rubric scoring → **sandboxed test execution**.

```
Paper #1:  task → topology → design.md  → LLM-judge ×3 → median rubric score (subjective)
Paper #2:  problem → topology → code/patch → SANDBOX runs hidden tests → pass/fail (objective)
```

| Layer | File(s) | Change |
|---|---|---|
| Config system | `config/models.py`, `config/loader.py` | **Extend** — add `task_type: coding`, `CodingProblemConfig` |
| Engine + matrix | `orchestrator/engine.py` | **Reuse as-is** |
| 12 topologies | `orchestrator/variants/*` | **Reuse** — agents emit code instead of design |
| Agents | `agents/*` | **Reuse classes**; new code prompt templates + code-extraction post-step |
| Providers + batching + rate limits | `providers/*`, `async_batcher/*` | **Reuse as-is** (all families already wired) |
| DB | `storage/database.py` | **Add** `code_results` table (schema v4) |
| **Evaluation** | `evaluation/pipeline.py` | **Replace** with `evaluation/code_pipeline.py` (selected by task type) |
| Execution sandbox | — | **New** `execution/` module |
| Analysis/stats | `analysis/statistics.py` | **Add** paired-binary stats (McNemar, bootstrap pass@k) |

---

## 4. New Components (detail)

### 4.1 Coding problem schema (`config/models.py`)
```python
class CodingProblemConfig(BaseModel, frozen=True):
    id: str
    benchmark: str            # "livecodebench" | "bigcodebench" | "swebench" | "humanevalplus"
    prompt: str               # spec / docstring / issue text
    entry_point: str = ""     # function name (function-level)
    visible_tests: str = ""   # example tests shown in prompt (function-level)
    # hidden tests live in the official harness, NOT in the prompt
    repo: str = ""            # SWE-bench: owner/name
    base_commit: str = ""     # SWE-bench
    difficulty: str = ""      # for stratified analysis
```
`TaskConfig` gains `task_type: "design" | "coding"`; the engine branches on it for template
selection and which eval pipeline to use.

### 4.2 Code extraction (`agents/code_extract.py`)
Pull the solution from a model response: last/largest fenced ```python block for function-level;
unified-diff parsing for SWE-bench. Robust to chatter around the code.

### 4.3 Sandboxed executor (`execution/`)  ← security-critical
- `execution/base.py` — `Sandbox` protocol: `run(code, tests, limits) -> ExecResult`.
- `execution/docker_sandbox.py` — Docker container, **no network**, CPU/mem/time caps, read-only FS
  except a temp workdir. Default for all execution of model-generated code.
- `execution/evalplus_runner.py` — wraps the official **EvalPlus** harness (HumanEval+/MBPP+/BigCodeBench? )
- `execution/livecodebench_runner.py` — wraps the official **LiveCodeBench** runner (date-windowed).
- `execution/bigcodebench_runner.py` — wraps the official **BigCodeBench** harness.
- `execution/swebench_runner.py` — wraps the official **SWE-bench** Docker harness (apply patch →
  run FAIL_TO_PASS + PASS_TO_PASS).
- `ExecResult`: `passed: bool, n_pass, n_total, error_type` (compile/wrong-answer/timeout/runtime),
  `stdout_tail, exec_ms`.

**Principle:** use the *official* harness per benchmark so single-model baselines reproduce
published numbers (validates our pipeline; satisfies "numbers must match").

### 4.4 Code evaluation pipeline (`evaluation/code_pipeline.py`)
Parallel to the rubric pipeline; selected when `task_type == coding`. For each final code artifact:
extract code → run official harness in sandbox → persist objective result. Supports a Docker pool
for parallelism. Writes `code_results`.

### 4.5 New table (`storage/database.py`, schema v4)
```sql
CREATE TABLE IF NOT EXISTS code_results (
    result_id     TEXT PRIMARY KEY,
    design_id     TEXT NOT NULL REFERENCES designs(design_id),  -- the final code artifact
    run_id        TEXT NOT NULL REFERENCES runs(run_id),
    benchmark     TEXT NOT NULL,
    problem_id    TEXT NOT NULL,
    passed        BOOLEAN NOT NULL,
    n_pass        INTEGER, n_total INTEGER,
    error_type    TEXT,           -- compile|wrong_answer|timeout|runtime|none
    exec_ms       REAL,
    harness_meta  TEXT,           -- JSON: harness version, image, etc.
    created_at    TEXT DEFAULT (datetime('now'))
);
```

### 4.6 Metrics & statistics (`analysis/code_stats.py`)
- **pass@1** (per condition/benchmark), **pass@k** (uses reps), **pass@1/$** (cost-effectiveness).
- **McNemar's test** — paired binary outcomes, consortium vs baseline on the *same* problems
  (the correct test here, not Wilcoxon-on-continuous).
- **Bootstrap CIs** on pass@1 and on the consortium−baseline delta.
- **Per-problem win/loss matrices** — where does the consortium fix bugs the baseline misses, and
  where does it regress? (the "Frankenstein effect for code" check).
- **Error taxonomy** shift (compile vs wrong-answer vs timeout) baseline → consortium.
- Reuse Paper #1's Pareto / break-even / budget-sim machinery with pass@1 swapped for quality.

### 4.7 Scoring vs. selection — two different things (clarification)
- **Scoring (benchmark's job, identical for every condition):** single model *or* consortium, each
  submits **exactly one** final answer; the benchmark's **hidden** tests run on it and dictate
  pass/fail. **No condition ever sees the hidden tests** — that's what makes consortium pass@1
  comparable to single-model pass@1.
- **Selection (consortium's *internal* job, only for multi-candidate topologies):** `c-select` and
  the harness-consortium generate **several** candidates, so they must pick **one** to submit
  *before* scoring. They choose using only **allowed** signals — a **generated reproduction test**
  (from the issue), the **repo's own existing test suite**, **candidate-agreement voting**, or a
  **cross-model judge** — **never** the hidden eval tests (SWE-bench FAIL_TO_PASS / PASS_TO_PASS stay
  sealed until final scoring).
- **Optional upper bound:** we may *additionally* report **pass@k** ("solved if any candidate
  passes") — there the hidden tests directly decide and no internal selection is needed, but pass@k
  is easier than single-model pass@1, so it's a clearly-separate "any-pass / oracle" number, not the
  headline.

---

## 5. Variant → code mapping & model assignments

### Single-model baselines (reproduce leaderboards)
| Condition | Topology | Model |
|---|---|---|
| `base-sonnet` | v1a single-shot | Sonnet 4.x |
| `base-gpt41`  | v1a single-shot | GPT-4.1 |
| `base-gptoss` | v1a single-shot | gpt-oss-120b (local) |

### Consortium variants (heterogeneous)
| Condition | Topology (Paper #1) | Composition |
|---|---|---|
| `c-xreview` | v2b cross-model review | Sonnet writes → gpt-oss reviews → Sonnet revises |
| `c-adv`     | v4b structural-adversarial | Sonnet writes → GPT-4.1 demands rewrites |
| `c-select`  | v3 parallel + **select-best** | Sonnet+GPT-4.1+gpt-oss each solve → judge/vote selects (literal code-merge replaced by best-of-N selection) |
| `c-panel`   | v5 specialist panel | correctness / edge-cases / complexity reviewers (mixed models) → leader revises |
| `c-debate`  | v8 debate *(optional)* | two models argue a solution → judge picks |

**Code-specific opportunity:** unlike design, code has natural selection signals —
cross-model **agreement/voting** and **self-generated tests**. Primary arm keeps "pure
collaboration" (no hidden-test execution in loop) per the canonical-protocol principle; `c-select`
uses a cross-model judge (no hidden tests) as the selector.

### New prompt templates (`prompts/`)
`generation/solve_coding.j2`, `review/code_review.j2`, `review/structural_code_adversarial.j2`,
`review/specialist_code_review.j2`, `synthesis/select_best_code.j2`, `debate/code_*.j2`.
Mirror Paper #1's prompt *style* (opinionated, rejection criteria) but for code correctness,
edge cases, and complexity.

---

## 6. Phased Build

- **Phase 0 — Coding scaffolding (shared, benchmark-agnostic):** `task_type` + `CodingProblemConfig`;
  `code_extract`; `execution/` (Docker sandbox + `ExecResult`); `code_results` table (schema v4);
  `evaluation/code_pipeline.py`; new `consortium bench` CLI verb. Engine/variants/providers untouched.
- **Phase 1 — LiveCodeBench (+ HumanEval+ calibration):** loader → official runner; code prompt
  templates; **gate: single-model baselines reproduce published pass@1**; run core consortium
  variants; pass@1 + McNemar + bootstrap.
- **Phase 2 — BigCodeBench-Hard:** loader + harness on the same pipeline.
- **Phase 3 — SWE-bench Verified (model-consortium):** Agentless localize→repair→validate;
  consortium at repair step; official SWE-bench Docker harness for scoring.
- **Phase 4 — Harness-consortium (novel):** pluggable solver backends (Claude Code / Codex CLI /
  Aider / SWE-agent); selection-topology over candidate patches. Open design Q: *selection signal*
  (repo regression tests / generated reproduction test / patch-agreement vote / cross-model review).
- **Phase 5 — Analysis + paper:** pass@1 bars, cost-effectiveness Pareto, per-problem win/loss,
  error taxonomy, harness×model grid; reuse IEEE LaTeX scaffold in `final_paper/latex/`.

---

## 7. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| **Executing model-generated code** | Docker, no network, CPU/mem/time caps, read-only FS, ephemeral containers |
| **Ceiling effect hides gains** | Suite chosen for headroom (≤~77%); HumanEval+ demoted to calibration |
| **Baselines don't match leaderboards** | Use official harnesses + canonical generation protocol; treat repro as a go/no-go gate |
| **Contamination inflates baselines** | LiveCodeBench date-windowing; SWE-bench Verified |
| **SWE-bench cost/complexity** | Sequenced after cheap phases; Agentless (fixed pipeline) not open agent; subset for pilot; gpt-oss is free locally |
| **Code-merge produces broken code** | Replace literal merge with best-of-N selection for `c-select` |
| **Harness-consortium selection w/o hidden tests** | regression tests / generated repro test / agreement vote / cross-model review |

## 8. Rough pilot cost
Function-level (LCB + BCB-Hard, ~100 problems × ~8 conditions × 2 reps ≈ 1.6k solves): low $ (short
code, few calls); gpt-oss free. SWE-bench dominates (long agentic context × Docker) — keep to ~50
instances for the pilot. Hard gate on the cheap phases before spending on SWE-bench.

---

## 9. Status
- [x] Decisions locked (suite, roster, reps, harness-consortium sequencing)
- [ ] Phase 0 — scaffolding
- [ ] Phase 1 — LiveCodeBench + calibration
- [ ] Phase 2 — BigCodeBench-Hard
- [ ] Phase 3 — SWE-bench Verified
- [ ] Phase 4 — harness-consortium
- [ ] Phase 5 — analysis + paper
