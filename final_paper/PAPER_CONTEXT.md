# LLM Consortium Project Context

## What This Project Is

An empirical research study comparing 12 multi-agent LLM topology variants for software architecture design generation. The experiment ran 520 controlled runs (12 variants x 8 design tasks x 5 repetitions) evaluated by three independent evaluator models on a 12-dimensional rubric. Two papers are being produced: an IEEE-format conference paper and a detailed analysis report.

---

## Key Files

### Primary Deliverables (in workspace folder)
- `LLM_Consortium_IEEE_Paper.docx` — IEEE conference paper (~1429 KB)
- `LLM_Consortium_Analysis_Report.docx` — Detailed analysis report (~1177 KB)

### Generator Scripts (in session working directory)
- `/sessions/optimistic-dazzling-shannon/ieee_paper.js` — Generates IEEE paper DOCX (693 lines, Node.js using docx library)
- `/sessions/optimistic-dazzling-shannon/generate_report.js` — Generates analysis report DOCX (656 lines, Node.js using docx library)
- `/sessions/optimistic-dazzling-shannon/premium_figures.py` — Generates 11 premium figures with v2c→v2b remap
- `/sessions/optimistic-dazzling-shannon/cube_final3.py` — Design space 3D cube figure
- `/sessions/optimistic-dazzling-shannon/redraw_workflows_v8.py` — Workflow topology diagram

### Figure Files
All in `analysis_output/` folder. Key ones used in papers:
- `fig7_variant_workflows.png` — Full-width workflow topology diagram (Fig. 1 in IEEE paper)
- `fig_ensemble_ranking.png` — Ensemble ranking bar chart (Fig. 3)
- `fig1_quality_boxplot.png` — Quality boxplot showing variance (Fig. 4)
- `fig2_quality_by_complexity.png` — Complexity interaction (Fig. 5)
- `fig3_dimension_heatmap.png` — Delta heatmap (Fig. 6)
- `fig4_pareto_frontier.png` — Pareto frontier (Fig. 7)
- `fig9_crossmodel_mechanism.png` — Cross-model/adversarial mechanism (Fig. 8)
- `fig8_design_space.png` — 3D design space cube (Fig. 9)
- `fig_evaluator_agreement.png` — Cross-validation agreement (Fig. 10)

---

## Experiment Details

### 12 Variants
| ID | Description | 2x2x2 Cell |
|----|-------------|------------|
| v1 | Single-model baseline | Centralized, Homogeneous, Cooperative |
| v1a | Single-model + self-review | Centralized, Homogeneous, Cooperative |
| v2a | Two-pass same-model review | Centralized, Homogeneous, Cooperative |
| v2b | Cross-model review (was v2c) | Centralized, Homogeneous, Cooperative |
| v3a | Parallel merge (3 generators) | Decentralized, Homogeneous, Cooperative |
| v3b | Parallel merge (2 generators) | Decentralized, Homogeneous, Cooperative |
| v3c | Parallel merge + review | Decentralized, Homogeneous, Cooperative |
| v4 | Adversarial with soft rejection | Centralized, Homogeneous, Adversarial |
| v4b | Structural adversarial (rewrite mandates) | Centralized, Homogeneous, Adversarial |
| v5 | Specialist panel | Decentralized, Specialized, Cooperative |
| v6 | Debate (two models argue) | Decentralized, Homogeneous, Adversarial |
| v8 | Orchestrated pipeline | Centralized, Specialized, Cooperative |

### 2x2x2 Factorial Design
Three dimensions: Authority (centralized vs. decentralized), Roles (homogeneous vs. specialized), Dynamics (cooperative vs. adversarial)

### Evaluator Ensemble
Weighted: (2x Opus + 2x Sonnet + 1x GPT-OSS) / 5
Models: `openai/gpt-oss-120b-maas`, `claude-opus-4-6`, `claude-sonnet-4-6`

### 8 Design Tasks (varying complexity)
Simple: URL shortener, markdown note app, weather dashboard
Medium: recipe manager, fitness tracker
Complex: real-time chat, collaborative editor, SaaS billing platform

---

## Four Core Findings

1. **Structural adversarial review (v4b) ranks #1** (ensemble 4.637) — Rewrite mandates, not patches, produce the highest quality designs
2. **Cross-model review (v2b) ranks #2** (ensemble 4.606) — Different model families surface blind spots that same-model review misses
3. **Evaluator disagreement is informative** — Different model families weight design qualities differently; multi-evaluator ensemble is more robust
4. **Parallel merge is fundamentally broken** — Token starvation causes merger to compress designs into undersized skeletons

---

## Four Hypotheses (H1-H4)

- **H1**: Parallel merge should outperform sequential → **REFUTED** (v3a/v3b/v3c rank bottom-third)
- **H2**: Adversarial dynamics will show higher variance → **PARTIALLY SUPPORTED** (v6 shows wider spread, v4b does not)
- **H3**: Specialist panel will outperform homogeneous → **PARTIALLY SUPPORTED** (v5 ranks mid-pack, beaten by simpler topologies)
- **H4**: Baseline will perform adequately on simple tasks → **REFUTED** (v4b and v2b dominate even on simple tasks)

---

## Statistical Results

- **Friedman test**: Variant choice has statistically significant effect (χ² = 220.3, p < 10⁻⁴⁰)
- **Wilcoxon signed-rank with Bonferroni**: Pairwise comparisons confirm top-tier separation
- **ANOVA**: η² = 0.64 with all variants, drops to 0.145 without v3 — 85% unexplained attributed to task difficulty and run-to-run variation
- **Cohen's d**: v4b vs v4 = 1.47 (very large effect)

---

## IEEE Paper Structure (8 sections, 10 figures, 5 tables)

| Section | Title |
|---------|-------|
| I | Introduction |
| II | Related Work |
| III | Methodology (includes TABLE I: Variant Configs, TABLE II: Design Tasks, Prompt Excerpts) |
| IV | Results (TABLE III: Rankings, TABLE IV: Statistical Comparisons) |
| V | Analysis: Key Mechanisms |
| VI | Discussion (includes TABLE V: Practitioner Recommendations, Hypotheses H1-H4) |
| VII | Conclusion |
| VIII | Cross-Evaluator Robustness and Implications |

---

## Key Technical Implementation Details

### DOCX Generation
Both papers use Node.js `docx` library. Run with:
```bash
node ieee_paper.js       # outputs to mnt/llm-consortium/
node generate_report.js  # outputs to mnt/llm-consortium/
```

### IEEE Paper Layout
- US Letter (8.5" x 11"), two-column layout with 0.625" margins
- Full-width workflow diagram via `SectionType.CONTINUOUS` breaks (switches from 2-col → 1-col → 2-col)
- Section headings: centered, uppercase, bold, Times New Roman 10pt
- Sub-headings: italic, bold, Times New Roman 10pt
- Section separators: blank line + horizontal rule before each major section

### Helper Functions (ieee_paper.js)
- `sectionSpacer()` — Returns array: [empty paragraph, horizontal rule paragraph]
- `sectionHeading(text)` — Centered uppercase bold heading
- `subHeading(text)` — Italic bold sub-heading
- `bodyText(text)` — Standard body paragraph
- `bodyMixed(segments)` — Mixed bold/plain text paragraph
- `imgPara(filename, w, h)` — Image paragraph from analysis_output/
- `figCaption(text)` — Centered italic figure caption
- `tableCaption(text)` — Centered bold table caption
- `headerCell(text)` / `dataCell(text)` — Table cell helpers

### Figure Generation
```bash
python premium_figures.py     # 11 analysis figures
python cube_final3.py         # 3D design space
python redraw_workflows_v8.py # Workflow diagram
```

---

## Important Design Decisions (User Directives)

1. **DOCX only, no PDFs** as primary deliverable
2. **Plain English first, stats as parenthetical references** — e.g., "differences are real and not due to chance (Friedman χ² = 220.3, p < 10⁻⁴⁰)"
3. **Practitioner-guide tone** — Not academic obsession with one finding; balanced coverage of all 4 findings
4. **Both papers must stay synchronized** — No significant drift between IEEE paper and analysis report
5. **Tables for structured data** (variant configs, tasks, recommendations), inline for prose (IEEE convention)
6. **v7 removed entirely** — User decided singling out unexplored v7 as limitation is bad when many 2x2x2 cells are unexplored
7. **Rubric is NOT a limitation** — Using one rubric is a deliberate experimental control, not a flaw
8. **v2c renamed to v2b** throughout
9. **v4b coverage balanced** — Was overemphasized; now all 4 findings get equal weight
10. **Visual breathing room** — Blank line + horizontal rule before each major section heading

---

## Recent Changes (Latest Session)

- Increased section heading spacing and added `sectionSpacer()` (blank line + HR) before every major section
- Removed "IEEE Conference Paper —" from page header
- Converted variant configurations, design tasks, and practical recommendations to tables
- Simplified all statistical jargon to plain English
- Reduced v4b overemphasis — merged Analysis + Empirical Evidence sections, removed 3 redundant v4b figures
- Made workflow diagram full-page width via continuous section breaks
- Added hypotheses section (H1-H4) and prompt excerpts section

---

## How to Regenerate Papers

```bash
# From /sessions/optimistic-dazzling-shannon/
node ieee_paper.js        # → mnt/llm-consortium/LLM_Consortium_IEEE_Paper.docx
node generate_report.js   # → mnt/llm-consortium/LLM_Consortium_Analysis_Report.docx
```

Both scripts read images from `mnt/llm-consortium/analysis_output/` and output DOCX files to `mnt/llm-consortium/`.
