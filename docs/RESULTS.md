# Paper #2 — pilot results (HumanEval+)

Objective software-engineering evaluation of the consortium topologies: the same
multi-agent collaboration framework from Paper #1, but emitting **real code**
scored by **execution against hidden tests** (pass@1) — no LLM judge.

## Setup

- **Models (all via DigitalOcean serverless inference, OpenAI-compatible):**
  `do-sonnet` = anthropic-claude-4.6-sonnet, `do-gpt52` = openai-gpt-5.2,
  `do-gptoss` = openai-gpt-oss-120b.
- **Generation (Phase A):** each topology emits a single final solution per
  problem. Temperature 0 (greedy) for single-sample pass@1. 1 rep.
- **Scoring (Phase B):** the official **EvalPlus** harness runs the hidden
  base+plus tests inside a network-disabled Docker container; `passed` = the
  hardened HumanEval+ *plus* status. **No model ever sees the hidden tests.**
- **Harness validation:** canonical solutions score **99.4%** plus (163/164),
  confirming our scoring reproduces the official protocol.

## Single-model baselines (calibration gate)

| Condition | Model | pass@1 | 95% CI |
|---|---|---:|---|
| base-sonnet | claude-4.6-sonnet | **94.5%** | [90.9, 97.6] |
| base-gptoss | gpt-oss-120b | **93.3%** | [89.0, 97.0] |
| base-gpt52 | gpt-5.2 | **90.9%** | [86.0, 95.1] |

All three land in the published leaderboard range for strong 2026 models on
HumanEval+ — the **pipeline reproduces leaderboard pass@1**, so the gate passes.

**Key finding:** all three models cluster at ~91–95%. **HumanEval+ is saturated
(ceiling)** for this roster, exactly as the plan predicted — so it serves as a
*calibration anchor*, not a headline. A meaningful consortium-vs-single-model
comparison needs a benchmark with headroom (LiveCodeBench / BigCodeBench-Hard /
SWE-bench Verified; integration status + blockers in `REMAINING_PHASES.md`).

## Consortium topologies (heterogeneous)

Both run on the full 164-problem set; McNemar is exact, paired per-problem vs base-sonnet.

| Condition | Topology | pass@1 | 95% CI | vs sonnet (fix / regress) | McNemar p |
|---|---|---:|---|---|---:|
| base-sonnet | single-shot | 94.5% | [90.9, 97.6] | — | — |
| **c-xreview** | sonnet writes → gpt-oss reviews → sonnet revises | **87.8%** | [82.9, 92.7] | +3 / −14 | **0.0127** |
| c-adv* | sonnet writes → gpt-5.2 attacks → sonnet revises | 91.0%* | [83.6, 97.0] | +2 / −5* | 0.45* |

\* c-adv on a partial 67/164 at time of writing (run still completing); refreshed on the final re-grade.

**Headline finding — collaboration *hurts* at ceiling (the "Frankenstein effect for code").**
Cross-model review (`c-xreview`) significantly *degrades* a strong model: it
**regressed 14** problems sonnet had solved while fixing only **3** (net −11,
McNemar p = 0.013). The cause is **over-revision** — verified manually: every
regression still contained valid extracted code (0 extraction failures), and the
revisions systematically *bloated* correct solutions (e.g. HumanEval/63: 214 →
761 chars) with extra handling that introduced bugs. When the base solution is
already correct (94.5% baseline), a reviewer's "improvements" have little to fix
and much to break.

This mirrors Paper #1's finding that naive collaboration can reduce quality, now
shown **objectively** (correct code made incorrect) rather than via an LLM judge.
It also sharpens the motivation for headroom benchmarks: collaboration should pay
off where the base model actually fails (LiveCodeBench / BigCodeBench-Hard /
SWE-bench), and HumanEval+ ceiling is precisely the regime where it cannot.

## Reproduce

```
make setup                     # deps + EvalPlus Docker harness
make validate-harness          # canonical ~99% (no LLM calls)
make baselines BENCH=humanevalplus   # generate + grade + report
make consortium BENCH=humanevalplus  # consortium arms
```
