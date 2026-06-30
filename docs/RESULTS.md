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
| **c-xreview** | sonnet writes → gpt-oss reviews → sonnet revises | **87.8%** | [82.9, 92.7] | +3 / −14 | **0.013** |
| c-adv | sonnet writes → gpt-5.2 attacks → sonnet revises | 91.5% | [87.2, 95.7] | +4 / −9 | 0.27 |

**Headline finding — collaboration *hurts* at ceiling (the "Frankenstein effect for code").**
Both heterogeneous topologies *regress* relative to the strong single model, and
cross-model review does so significantly: `c-xreview` falls **94.5% → 87.8%**,
**breaking 14** problems sonnet had solved while fixing only **3** (McNemar
p = 0.013). The cause is **over-revision** — verified manually: every regression
still contained valid extracted code (0 extraction failures), and revisions
systematically *bloated* correct solutions (e.g. HumanEval/63: 214 → 761 chars)
with extra handling that introduced bugs.

`c-adv` regresses less (91.5%, n.s.) for a structural reason: its adversarial
**accept/reject gate can stop early** — when the first solution is accepted there
is no revision and therefore no regression risk — whereas `c-xreview` always runs
a review→revise cycle that always risks over-revision. So the *gating* topology is
more conservative than the *mandatory-revision* topology, exactly where the base is
already correct.

This mirrors Paper #1's finding that naive collaboration can reduce quality, now
shown **objectively** (correct code made incorrect) rather than via an LLM judge.
It also sharpens the thesis: collaboration should pay off where the base model
actually *fails*, so the decisive test is on headroom benchmarks (LiveCodeBench /
BigCodeBench-Hard / SWE-bench) — HumanEval+ ceiling is precisely the regime where
it cannot help. Those are scaffolded with exact blockers in `REMAINING_PHASES.md`.

## Reproduce

```
make setup                     # deps + EvalPlus Docker harness
make validate-harness          # canonical ~99% (no LLM calls)
make baselines BENCH=humanevalplus   # generate + grade + report
make consortium BENCH=humanevalplus  # consortium arms
```
