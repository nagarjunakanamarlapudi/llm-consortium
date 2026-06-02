# LLM Consortium

[![arXiv](https://img.shields.io/badge/arXiv-2606.01490-b31b1b.svg)](https://arxiv.org/abs/2606.01490)

A controlled experiment comparing **12 multi-agent LLM collaboration topologies** for software architecture design — and the framework used to run it.

📄 **Paper:** [*LLM Consortium for Software Design Refinement: A Controlled Experiment on Multi-Agent Collaboration Topologies*](https://arxiv.org/abs/2606.01490) — N. Kanamarlapudi, P. K · arXiv:2606.01490 [cs.SE]

## What it is

We cross three dimensions of multi-agent collaboration — **Authority** (centralized vs. decentralized), **Roles** (homogeneous vs. specialized), and **Dynamics** (cooperative vs. adversarial) — in a 2×2×2 factorial design, deriving 12 variant topologies. Across **520 runs** (12 variants × 8 design tasks × 5 repetitions), each design is scored on a 12-dimension rubric by a weighted three-evaluator ensemble (2×Opus + 2×Sonnet + 1×GPT-OSS).

## Key findings

1. **Structural adversarial review (v4b) ranks #1** (ensemble 4.637) — demanding rewrite mandates beats accepting patches.
2. **Cross-model review (v2b) ranks #2** (4.606) — a different model family surfaces complementary blind spots.
3. **Evaluator diversity is itself a finding** — model families weight design qualities differently; an ensemble is more robust.
4. **Parallel merge is fundamentally broken** — token starvation and the "Frankenstein effect."

## Repository layout

- `final_paper/` — the paper: PDF, DOCX, and the arXiv LaTeX source (`latex/`) + submission zip.
- `src/` — the consortium orchestration framework (variants, providers, evaluation).
- `configs/`, `prompts/` — experiment configuration and prompt templates.
- `data/` — experiment databases and outputs (gitignored; ~1.4 GB, reproduces the paper).

## Citation

```bibtex
@misc{kanamarlapudi2026llmconsortium,
  title         = {LLM Consortium for Software Design Refinement: A Controlled
                   Experiment on Multi-Agent Collaboration Topologies},
  author        = {Kanamarlapudi, Nagarjuna and K, Praveen},
  year          = {2026},
  eprint        = {2606.01490},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SE},
  url           = {https://arxiv.org/abs/2606.01490}
}
```
