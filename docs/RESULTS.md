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

_(filled in when the consortium runs grade; both run on the full 164-problem set.)_

- **c-xreview** (v2 cross-model review): sonnet writes → gpt-oss reviews → sonnet revises.
- **c-adv** (v4 structural-adversarial): sonnet writes → gpt-5.2 demands rewrites → sonnet revises.

Reported per condition: pass@1, bootstrap CI, and **exact McNemar** vs base-sonnet
(paired per-problem: where the consortium fixes problems the baseline misses vs.
regresses). At ceiling we expect small, likely non-significant deltas; the value
is the validated end-to-end consortium path and the per-problem win/loss matrix.

## Reproduce

```
make setup                     # deps + EvalPlus Docker harness
make validate-harness          # canonical ~99% (no LLM calls)
make baselines BENCH=humanevalplus   # generate + grade + report
make consortium BENCH=humanevalplus  # consortium arms
```
