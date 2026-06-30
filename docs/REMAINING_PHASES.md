# Remaining-phase integration plan (Paper #2)

Status of the build as of this session — what runs today and the precise plan to
finish Phases 2–4. Phase 1 (function-level: generate via topologies → score with
the official EvalPlus harness → pass@1/McNemar) is **implemented and validated**.

## What works today (Phase 0 + Phase 1)

- **Provider**: `do_inference` (OpenAI-compatible) routes the whole roster through
  DigitalOcean serverless inference: `do-sonnet` (claude-4.6-sonnet),
  `do-gpt52` (gpt-5.2), `do-gptoss` (gpt-oss-120b). See `configs/models/do_*.yaml`.
- **Generation**: `consortium bench run <benchmark> --conditions ... --limit N --reps R`
  builds an in-memory experiment (roster × conditions × problems) and runs the
  existing v1–v8 orchestrators, emitting code instead of designs. Conditions live
  in `benchmarks/conditions.py` (baselines + `c-xreview` v2, `c-adv` v4).
- **Scoring**: `consortium bench grade <benchmark>` runs the **official EvalPlus
  harness in Docker** (`consortium-evalplus`, `--network none`) and writes per-design
  pass/fail to `code_results`. Validated: canonical solutions score 99.4% (plus).
- **Analysis**: `consortium bench report <benchmark>` → pass@1, pass@k, bootstrap
  CIs, and exact McNemar vs a baseline. (`analysis/code_stats.py`.)
- **Benchmarks live now**: `humanevalplus`, `mbppplus` (both via EvalPlus; the
  Docker image already caches both datasets).

The whole loop is `make pilot BENCH=humanevalplus`.

## Phase 2 — BigCodeBench-Hard (headroom, practical library use)

Same shape as EvalPlus; only the harness package and dataset differ.

1. **Loader** (`benchmarks/loaders.py`): add `load_bigcodebench(limit, split="hard")`
   using `from bigcodebench.data import get_bigcodebench` → map each record to a
   `CodingProblemConfig(benchmark="bigcodebench", prompt=..., entry_point=...)`.
   Register under `_LOADERS["bigcodebench"]`.
2. **Harness image** (`docker/bigcodebench.Dockerfile`): `pip install bigcodebench`,
   pre-cache the dataset, same `--network none` run contract. NOTE: BigCodeBench
   pulls many scientific libs — the image is large; build once.
3. **Scorer** (`evaluation/bigcodebench_scorer.py`): mirror `EvalPlusScorer` but
   invoke `bigcodebench.evaluate --subset hard --samples ...`; parse its
   `*_eval_results.json` (`status == "pass"`). Group by `(variant_id, repetition)`,
   pad missing problems, write `code_results` with `reportable=True`.
4. Add `bigcodebench` to `bench grade` dispatch. Everything downstream (report,
   stats) already works because it reads `code_results`.

Expected pass@1 (headroom): Sonnet ~35–45%, gpt-5.2 ~35%, gpt-oss ~30%.

## Phase 1b — LiveCodeBench (contamination-free, date-windowed)

Heavier than EvalPlus: stdin/stdout *and* functional problems, date windows.

1. **Loader**: `datasets.load_dataset("livecodebench/code_generation_lite", version_tag=...)`,
   filter by release window (e.g. problems after the models' training cutoffs),
   map to `CodingProblemConfig(benchmark="livecodebench", prompt=question_content,
   difficulty=..., metadata for the test format)`.
2. **Harness image** (`docker/livecodebench.Dockerfile`): install `livecodebench`
   (the `lcb_runner`), cache the dataset.
3. **Scorer** (`evaluation/livecodebench_scorer.py`): write generations in the
   `lcb_runner` custom-eval JSONL format, run its evaluator in Docker
   (`python -m lcb_runner.runner.custom_evaluator ...`), parse pass@1. Handle both
   the call-based and stdin/stdout test harnesses LiveCodeBench uses.
4. Date-windowing is the contamination control — record the window in `harness_meta`.

Expected pass@1 (headroom): Sonnet ~55–64%, gpt-5.2 mid, gpt-oss ~58–60%.

## Phase 3 — SWE-bench Verified (repo-level)

Different solver shape (patches, not function bodies) but the same scoring seam.

1. **Solver = Agentless** (localize → repair → validate), consortium at the *repair*
   step. Generation produces a unified diff; reuse `agents/code_extract.extract_diff`.
2. **Loader**: `datasets.load_dataset("princeton-nlp/SWE-bench_Verified")` →
   `CodingProblemConfig(benchmark="swebench", repo=..., base_commit=..., prompt=issue)`.
3. **Scorer** (`evaluation/swebench_scorer.py`): wrap the official SWE-bench Docker
   harness (`python -m swebench.harness.run_evaluation`), apply patch → run
   FAIL_TO_PASS + PASS_TO_PASS, parse the report → `code_results`.
4. **Selection without hidden tests** (for multi-candidate arms): generated
   reproduction test / repo's own suite / patch-agreement vote / cross-model judge —
   never the hidden FAIL_TO_PASS set. See plan §4.7.

## Phase 4 — Harness-consortium (novel, RQ5)

A consortium of *(harness × model)* solver backends (Claude Code, Codex CLI,
Aider/SWE-agent), each producing a candidate patch; a selection topology picks one.

1. Define a `SolverBackend` protocol: `solve(problem) -> patch`. Wrap each external
   CLI harness behind it (subprocess, isolated workdir).
2. Run N backends per instance → N candidate patches.
3. Selection topology (reuse v3-select / a cross-model judge) chooses one patch
   using only allowed signals; score the chosen patch with the SWE-bench harness.
4. This turns the "harness dependence" confound into the independent variable.

## Cross-cutting TODО

- **c-select (v3 heterogeneous parallel)**: the framework's `parallel_leaders` is
  count-based (one model). For 3 *different* models in parallel + judge-selects, use
  the `participants` mechanism (individually-modeled agents) plus a code-aware judge
  template (`synthesis/select_best_code.j2`). Add a coding branch to the merger/judge
  agents (mirrors the reviewer/adversary coding branches already added).
- **Pricing**: `configs/models/do_*.yaml` use published list prices for the
  cost-effectiveness figure; refine `do-gpt52` if exact gpt-5.2 rates differ.
