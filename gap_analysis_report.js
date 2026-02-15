const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, PageNumber, PageBreak, LevelFormat
} = require("docx");

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };
const cellMargins = { top: 60, bottom: 60, left: 100, right: 100 };

// Helper for header cells
function hCell(text, width) {
  return new TableCell({
    borders, width: { size: width, type: WidthType.DXA },
    shading: { fill: "1B3A5C", type: ShadingType.CLEAR },
    margins: cellMargins,
    verticalAlign: "center",
    children: [new Paragraph({ children: [new TextRun({ text, bold: true, color: "FFFFFF", font: "Arial", size: 18 })] })]
  });
}

// Helper for regular cells
function cell(text, width, opts = {}) {
  const fill = opts.fill || undefined;
  const shading = fill ? { fill, type: ShadingType.CLEAR } : undefined;
  return new TableCell({
    borders, width: { size: width, type: WidthType.DXA },
    shading, margins: cellMargins,
    children: [new Paragraph({ children: [new TextRun({ text, font: "Arial", size: 18, bold: opts.bold, color: opts.color })] })]
  });
}

// Severity tag helper
function severityCell(severity, width) {
  const colors = {
    "CRITICAL": { fill: "FADBD8", color: "922B21" },
    "HIGH": { fill: "FDEBD0", color: "935116" },
    "MEDIUM": { fill: "FEF9E7", color: "7D6608" },
    "LOW": { fill: "D5F5E3", color: "1E8449" },
    "PASS": { fill: "D4EFDF", color: "196F3D" },
  };
  const c = colors[severity] || { fill: "F2F3F4", color: "2C3E50" };
  return cell(severity, width, { fill: c.fill, color: c.color, bold: true });
}

function makeTable(headers, rows, colWidths) {
  const totalWidth = colWidths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: totalWidth, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [
      new TableRow({ children: headers.map((h, i) => hCell(h, colWidths[i])) }),
      ...rows.map(row => new TableRow({
        children: row.map((c, i) => {
          if (typeof c === "object" && c._type === "severity") return severityCell(c.text, colWidths[i]);
          return cell(String(c), colWidths[i]);
        })
      }))
    ]
  });
}

function sev(text) { return { _type: "severity", text }; }

function heading(text, level) {
  return new Paragraph({ heading: level, spacing: { before: 300, after: 150 }, children: [new TextRun({ text, font: "Arial" })] });
}

function para(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 120 },
    children: [new TextRun({ text, font: "Arial", size: 22, bold: opts.bold, italics: opts.italics, color: opts.color })]
  });
}

function bulletItem(text, ref) {
  return new Paragraph({
    numbering: { reference: ref, level: 0 },
    spacing: { after: 80 },
    children: [new TextRun({ text, font: "Arial", size: 22 })]
  });
}

// ===== BUILD DOCUMENT =====
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, font: "Arial", color: "1B3A5C" },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: "2C3E50" },
        paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Arial", color: "34495E" },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 2 } },
    ]
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1200, bottom: 1440, left: 1200 }
      }
    },
    headers: {
      default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "LLM Consortium \u2014 Thesis Gap Analysis", font: "Arial", size: 16, color: "999999", italics: true })] })] })
    },
    footers: {
      default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Page ", font: "Arial", size: 16, color: "999999" }), new TextRun({ children: [PageNumber.CURRENT], font: "Arial", size: 16, color: "999999" })] })] })
    },
    children: [
      // ===== TITLE PAGE =====
      new Paragraph({ spacing: { before: 3000 } }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [new TextRun({ text: "LLM CONSORTIUM", font: "Arial", size: 52, bold: true, color: "1B3A5C" })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 }, children: [new TextRun({ text: "THESIS GOALS vs. IMPLEMENTATION", font: "Arial", size: 36, color: "2C3E50" })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 }, children: [new TextRun({ text: "Comprehensive Gap Analysis Report", font: "Arial", size: 28, color: "7F8C8D", italics: true })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 }, children: [new TextRun({ text: "February 2026", font: "Arial", size: 22, color: "95A5A6" })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Prepared by Claude \u2014 Automated Code Review", font: "Arial", size: 20, color: "95A5A6" })] }),

      new Paragraph({ children: [new PageBreak()] }),

      // ===== EXECUTIVE SUMMARY =====
      heading("1. Executive Summary", HeadingLevel.HEADING_1),
      para("This report presents a thorough gap analysis comparing the thesis proposal (\"LLM Consortium for Software Design Refinement\") against the current codebase implementation. The analysis covers all 8 variant orchestrators, the agent hierarchy, evaluation pipeline, configuration/task definitions, prompt architecture, analysis modules, and observability tooling."),
      para("Overall assessment: The codebase is architecturally sound and covers approximately 80\u201385% of the thesis requirements. The core experiment pipeline (config \u2192 agents \u2192 orchestrators \u2192 evaluation \u2192 storage) is complete and functional. However, there are several critical gaps that must be addressed before the experiment can produce results that fully validate the thesis claims.", { bold: true }),

      heading("Summary of Findings", HeadingLevel.HEADING_2),
      makeTable(
        ["Category", "Status", "Key Gaps"],
        [
          ["Task Portfolio (T1\u2013T8)", sev("CRITICAL"), "5 of 8 tasks mismatch thesis names/types"],
          ["Variant Sub-Variants", sev("HIGH"), "v1a/b, v2a/b/c, v3a/b/c not differentiated"],
          ["Experiment Controls", sev("HIGH"), "No run randomization, no seed management"],
          ["Critical Blocker Detection", sev("HIGH"), "Not implemented in evaluation pipeline"],
          ["Convergence Metrics (v7)", sev("HIGH"), "Epsilon-based convergence not implemented"],
          ["Coherence Pass (v6)", sev("MEDIUM"), "Missing dedicated phase per thesis spec"],
          ["Krippendorff\u2019s Alpha", sev("MEDIUM"), "Simplified proxy, not full computation"],
          ["Missing Analysis Metrics", sev("MEDIUM"), "Convergence speed, novelty/diversity, blocker count, Cohen\u2019s d"],
          ["Test Coverage", sev("MEDIUM"), "No tests for loader, heatmaps, exporter modules"],
          ["Rubrics & Prompts", sev("PASS"), "Both rubrics complete with anchors; 5-component prompt structure verified"],
          ["Agent Architecture", sev("PASS"), "All 7 agent types implemented correctly"],
          ["Database Schema", sev("PASS"), "All 8 required tables present with proper indexing"],
          ["Analysis Pipeline", sev("PASS"), "Friedman, Wilcoxon, Pareto, heatmaps, \u00A710.2 framework all implemented"],
          ["Observability", sev("PASS"), "Timeline builder, Chrome trace, multi-format export complete"],
        ],
        [3600, 1200, 5040]
      ),

      new Paragraph({ children: [new PageBreak()] }),

      // ===== GAP #1: TASK PORTFOLIO =====
      heading("2. Critical Gaps", HeadingLevel.HEADING_1),
      heading("2.1 Task Portfolio Mismatch (CRITICAL)", HeadingLevel.HEADING_2),
      para("The thesis specifies 8 tasks with specific names, complexity levels, and design types. Five of the eight configured tasks do not match the thesis specification. This is the most critical issue because it directly undermines the experimental claims."),

      makeTable(
        ["ID", "Thesis Task", "Configured Task", "Thesis Type", "Config Type", "Match?"],
        [
          ["T1", "URL Shortener", "URL Shortener Service", "System", "System", sev("PASS")],
          ["T2", "Rate Limiter Service", "Real-Time Chat App", "Application", "Application", sev("CRITICAL")],
          ["T3", "Notification System", "E-Commerce Platform", "System", "Application", sev("CRITICAL")],
          ["T4", "Task Queue / Job Scheduler", "Notification Service", "Application", "System", sev("CRITICAL")],
          ["T5", "Multi-Tenant SaaS Billing", "Trading Platform", "System", "Application", sev("CRITICAL")],
          ["T6", "Collaborative Editor", "Video Streaming", "System", "System", sev("MEDIUM")],
          ["T7", "Access Control Module", "AV Fleet Mgmt", "Application", "System", sev("CRITICAL")],
          ["T8", "Event-Driven Orders", "ML Platform", "Application", "Application", sev("MEDIUM")],
        ],
        [600, 1800, 1800, 1200, 1200, 1200]
      ),

      para(""),
      para("Impact: The thesis\u2019s complexity hypotheses (Section 6.2) make predictions about specific tasks (e.g., T2 Rate Limiter as a concurrency-focused simple task, T5 SaaS Billing as a financial-correctness complex task). With different tasks, these predictions cannot be tested as specified. The 2\u00D72\u00D72 balance of system/application types is also disrupted.", { italics: true }),
      para("Recommendation: Either update the task configs to match the thesis, or revise the thesis proposal to reflect the new task choices. The configured tasks are well-specified and valid; the issue is solely the mismatch with documented claims."),

      // ===== GAP #2: VARIANT SUB-VARIANTS =====
      heading("2.2 Missing Variant Sub-Variants (HIGH)", HeadingLevel.HEADING_2),
      para("The thesis defines explicit sub-variants for v1, v2, and v3 that enable key experimental comparisons. None of these are implemented as distinct code paths:"),
      makeTable(
        ["Variant", "Sub-Variant", "Thesis Purpose", "Implementation Status"],
        [
          ["v1", "v1a: Single-shot (no loop)", "Absolute quality floor", sev("CRITICAL")],
          ["v1", "v1b: Self-refinement loop", "Fair baseline", sev("PASS")],
          ["v2", "v2a: Same-model review", "Isolate feedback source variable", sev("HIGH")],
          ["v2", "v2b: Cross-model review", "Test training distribution diversity", sev("HIGH")],
          ["v2", "v2c: Multi-model panel", "Maximum reviewer diversity", sev("HIGH")],
          ["v3", "v3a: Naive merge", "Merge strategy baseline", sev("HIGH")],
          ["v3", "v3b: Rubric-guided merge", "Structured synthesis", sev("PASS")],
          ["v3", "v3c: Dialectical merge", "Thesis-antithesis approach", sev("PASS")],
        ],
        [800, 2200, 2800, 2000]
      ),
      para(""),
      para("Impact: Without v1a, there is no absolute quality floor. Without v2a/b/c differentiation, the pairwise comparison \"v2a vs v2b\" (isolating cross-model review benefit) cannot be performed. The v3 merge templates exist (naive, rubric-guided, dialectical) but there\u2019s no config-level mechanism to select between them per sub-variant."),
      para("Recommendation: Add a sub_variant field to the variant configs. For v1, add a skip_review: true option. For v2, add model diversity enforcement in the factory. For v3, the merge template is already configurable\u2014just need v3a/v3b/v3c config variants."),

      // ===== GAP #3: EXPERIMENT CONTROLS =====
      heading("2.3 Missing Experiment Controls (HIGH)", HeadingLevel.HEADING_2),
      para("The thesis protocol (Section 7.3) requires three critical experimental controls that are not implemented:"),

      bulletItem("Run order randomization: The ExperimentRunner iterates through the (variant, task, rep) matrix in a fixed nested-loop order. The thesis requires randomization to prevent systematic effects from API rate changes, model updates, or time-of-day effects.", "bullets"),
      bulletItem("Seed management: No random seeds are set or tracked per repetition. The thesis specifies that where APIs support it, seeds should be set for reproducibility and paired comparisons.", "bullets"),
      bulletItem("Evaluator blinding: Evaluator receives no variant metadata (this IS correctly implemented).", "bullets"),
      para(""),
      para("Impact: Without randomization, early runs (v1, t1) may benefit from fresher API quotas while later runs (v8, t8) may hit rate limits. Without seeds, repetition pairing for Wilcoxon tests is weaker. The thesis acknowledges this as a limitation but the code doesn\u2019t even attempt it where APIs support it (e.g., Anthropic\u2019s seed parameter)."),

      // ===== GAP #4: CRITICAL BLOCKERS =====
      heading("2.4 Critical Blocker Detection Missing (HIGH)", HeadingLevel.HEADING_2),
      para("The thesis scoring methodology (Section 5.3) explicitly requires that Critical Blockers be flagged independently of the numeric score: \"A design at 3.8 with zero blockers may beat 4.2 with one blocker.\" The evaluation pipeline computes per-dimension scores and medians but has no mechanism to detect or flag critical blockers."),
      para("The evaluate_design.j2 template does ask for \"blockers\" in the JSON output schema, but the pipeline\u2019s scoring logic does not aggregate, persist, or surface these. The Blocker Count metric (Section 8.1) is listed as a primary metric but cannot be computed."),
      para("Recommendation: Add blocker extraction to the evaluation pipeline\u2019s JSON parsing. Persist to a dedicated column or table. Surface in analysis."),

      // ===== GAP #5: CONVERGENCE =====
      heading("2.5 V7 Convergence: Epsilon Not Implemented (HIGH)", HeadingLevel.HEADING_2),
      para("The thesis specifies that v7 (Consensus Convergence) should use an epsilon-based convergence criterion (\u03B5 = 0.3, max 4 rounds): \"if score delta < \u03B5 across agents, stop.\" The implementation instead relies on the LLM outputting a STATUS: CONVERGED string from the consensus_synthesis.j2 template, parsed via regex."),
      para("This means convergence is determined by the LLM\u2019s subjective judgment rather than a measurable numeric threshold. The LLM might declare convergence when scores actually differ by > 0.3, or continue iterating when designs are nearly identical."),
      para("Recommendation: Implement intermediate scoring (call the evaluator between rounds) and compute score delta programmatically. Fall back to LLM-based convergence only if scoring is too expensive."),

      new Paragraph({ children: [new PageBreak()] }),

      // ===== MEDIUM GAPS =====
      heading("3. Medium-Priority Gaps", HeadingLevel.HEADING_1),

      heading("3.1 V6 Rotating Leader: Missing Phase Structure & Coherence Pass", HeadingLevel.HEADING_2),
      para("The thesis specifies v6 as a multi-phase workflow: Phase 1 (domain model + module structure) \u2192 Phase 2 (error handling + data flow) \u2192 Phase 3 (testing + operability) \u2192 Phase 4 (coherence pass). The implementation uses simple round-robin rotation with generic review prompts. There is no phase-specific focus and no dedicated coherence pass."),
      para("Impact: Without phase structure, v6 is functionally just \"v2 with rotating leadership\" rather than the thesis\u2019s intended \"epistemic reset with phase-specific expertise.\" The coherence Achilles heel predicted by the thesis cannot be properly measured."),

      heading("3.2 Krippendorff\u2019s Alpha: Proxy Implementation", HeadingLevel.HEADING_2),
      para("The evaluation pipeline implements a variance-based proxy for Krippendorff\u2019s alpha (1 - within_var/total_var) rather than the true ordinal alpha. The code itself documents this: \"For the full experiment analysis, use the krippendorff package.\" The thesis target is \u03B1 > 0.7 for evaluator validation."),
      para("Recommendation: Install the krippendorff Python package and replace the proxy computation for the final experiment runs."),

      heading("3.3 Missing Analysis Metrics", HeadingLevel.HEADING_2),
      para("Four metrics from the thesis\u2019s primary metrics table (Section 8.1) are not implemented:"),
      makeTable(
        ["Metric", "Thesis Description", "Status", "Difficulty"],
        [
          ["Convergence Speed", "Rounds to reach score plateau (< 0.1 improvement/round)", sev("HIGH"), "Requires per-round evaluation scoring"],
          ["Per-Dimension Delta", "Per-dimension score improvement over v1b baseline", sev("MEDIUM"), "Straightforward from existing data"],
          ["Blocker Count", "Number of Critical Blockers flagged by evaluator", sev("HIGH"), "Requires blocker extraction (see 2.4)"],
          ["Novelty / Diversity", "Cosine similarity of section embeddings across 5 runs", sev("MEDIUM"), "Requires embedding computation"],
          ["Cohen\u2019s d", "Effect size for mean differences", sev("LOW"), "Rank-biserial used instead (acceptable)"],
        ],
        [1800, 3500, 1200, 1300]
      ),

      heading("3.4 Test Coverage Gaps", HeadingLevel.HEADING_2),
      para("Several modules lack test coverage entirely:"),
      makeTable(
        ["Module", "Tests?", "Risk"],
        [
          ["analysis/loader.py", "None", "Data integrity for all downstream analysis"],
          ["analysis/heatmaps.py", "None", "Visualization correctness"],
          ["observability/exporter.py", "None", "Export format correctness"],
          ["Checkpoint/resume flow", "None", "Crash recovery reliability"],
          ["Template variable completeness", "None", "Runtime template errors (caused bugs in first run)"],
        ],
        [2800, 1000, 4000]
      ),
      para(""),
      para("The Phase 3 Revised plan correctly identifies integration tests as P0 priority. The existing unit tests for statistics, pareto, framework, and timeline are real tests (not stubs) with synthetic data."),

      new Paragraph({ children: [new PageBreak()] }),

      // ===== WHAT'S WORKING WELL =====
      heading("4. What\u2019s Working Well", HeadingLevel.HEADING_1),
      para("The following areas are fully implemented and align with thesis requirements:"),

      heading("4.1 Agent Architecture", HeadingLevel.HEADING_2),
      para("All 7 agent types (Designer, Reviewer, Specialist, Adversary, Merger, Judge, and base) are cleanly implemented with proper separation of concerns. The BaseAgent._call_llm() method handles request construction, provider invocation, trace recording, and limit checking automatically. Feedback incorporation in DesignerAgent properly handles both generation and revision modes."),

      heading("4.2 Evaluation Pipeline", HeadingLevel.HEADING_2),
      para("The pipeline correctly implements: 3x evaluator runs per design with median aggregation, per-dimension scoring with JSON parsing, evaluator blinding (no variant metadata passed), coherence checking with section-pair contradiction detection and batch support, and disagreement flagging (range > 1.5)."),

      heading("4.3 Rubrics & Prompts", HeadingLevel.HEADING_2),
      para("Both rubrics (11 system dimensions, 12 application dimensions) are complete with full 1\u20135 anchors. Generation prompts follow the 5-component structure (task definition, positive examples, negative examples, required deliverables, meta-instructions). All four merge strategies are implemented as separate templates. The evaluation prompt is correctly shared across all variants."),

      heading("4.4 Statistical Analysis", HeadingLevel.HEADING_2),
      para("The analysis module implements: Friedman test with Nemenyi post-hoc, Wilcoxon signed-rank with Bonferroni correction, Mann-Whitney U for axis analysis, Pareto frontier computation with Plotly visualization, four types of heatmaps, the \u00A710.2 decision framework generator with confidence scoring, and all 8 prediction validations (P1\u2013P8). The rank-biserial correlation is used as effect size (acceptable substitute for Cohen\u2019s d in nonparametric tests)."),

      heading("4.5 Database & Storage", HeadingLevel.HEADING_2),
      para("All 8 required tables are present (runs, designs, reviews, evaluations, scores_median, coherence_checks, traces, batches) with proper indexing and WAL mode. Token and cost accounting is tracked at both the run and trace level."),

      heading("4.6 Observability", HeadingLevel.HEADING_2),
      para("The timeline builder correctly queries traces, groups by agent into swim lanes, detects parallelism via overlapping timestamps, and exports to Rich console, JSON, CSV, and Chrome trace formats. The CLI provides both single-run and aggregate views."),

      new Paragraph({ children: [new PageBreak()] }),

      // ===== VARIANT-BY-VARIANT =====
      heading("5. Variant-by-Variant Assessment", HeadingLevel.HEADING_1),
      makeTable(
        ["Variant", "Status", "Workflow", "Key Gap"],
        [
          ["v1 Baseline", sev("MEDIUM"), "gen \u2192 self-review \u2192 revision loop", "Missing v1a (single-shot, no loop)"],
          ["v2 Leader+Rev", sev("MEDIUM"), "gen \u2192 K reviews \u2192 revision loop", "No v2a/b/c model diversity enforcement"],
          ["v3 Parallel+Merge", sev("MEDIUM"), "K parallel gen \u2192 merge", "No iterative refinement post-merge; sub-variants template-only"],
          ["v4 Adversarial", sev("PASS"), "gen \u2192 accept/reject gate loop", "Verdict defaults to accept on parse failure"],
          ["v5 Specialist", sev("PASS"), "gen \u2192 N specialist reviews \u2192 revision", "Specialists are generic (not hardcoded 5 roles)"],
          ["v6 Rotating", sev("HIGH"), "Round-robin leader \u2192 reviews", "No phase structure; no coherence pass"],
          ["v7 Consensus", sev("HIGH"), "K parallel \u2192 peer review \u2192 converge", "No epsilon metric; LLM-based convergence only"],
          ["v8 Debate", sev("PASS"), "positions \u2192 rebuttals \u2192 judge", "Rebuttal count edge case if max_rounds=1"],
        ],
        [1600, 1000, 2800, 2400]
      ),

      new Paragraph({ children: [new PageBreak()] }),

      // ===== PAIRWISE COMPARISONS =====
      heading("6. Pairwise Comparison Readiness", HeadingLevel.HEADING_1),
      para("The thesis defines 8 targeted pairwise comparisons (Section 7.4). Here is the readiness status for each:"),
      makeTable(
        ["Comparison", "Variable Isolated", "Can Execute?", "Blocker"],
        [
          ["v1b vs v2a", "Feedback source: scores vs critique", sev("HIGH"), "v1a/v1b and v2a not differentiated"],
          ["v2a vs v2b", "Same-model vs cross-model review", sev("CRITICAL"), "v2a/v2b not implemented"],
          ["v2 vs v4", "Cooperative vs adversarial dynamics", sev("PASS"), "Both implemented"],
          ["v2 vs v5", "General vs specialist reviewers", sev("PASS"), "Both implemented"],
          ["v3 vs v7", "Central merger vs peer convergence", sev("PASS"), "Both implemented (v7 convergence is LLM-based)"],
          ["v3 vs v8", "Random vs forced opposition", sev("PASS"), "Both implemented"],
          ["v4 vs v8", "Critique-only vs build+critique", sev("PASS"), "Both implemented"],
          ["v5 vs v6", "Reviewer vs leader specialization", sev("MEDIUM"), "v6 lacks phase-specific focus"],
        ],
        [1500, 2300, 1300, 2700]
      ),

      new Paragraph({ children: [new PageBreak()] }),

      // ===== PRIORITIZED RECOMMENDATIONS =====
      heading("7. Prioritized Recommendations", HeadingLevel.HEADING_1),
      heading("P0: Must Fix Before Experiment", HeadingLevel.HEADING_2),
      bulletItem("Align task portfolio with thesis (or update thesis): 5 tasks need renaming/retyping to match documented claims about task complexity interactions.", "numbers"),
      bulletItem("Add run order randomization: Shuffle the (variant, task, rep) matrix in ExperimentRunner before execution. Simple random.shuffle() on the pending list.", "numbers"),
      bulletItem("Implement v1a single-shot variant: Add a skip_review flag to v1 config. Without v1a, there is no absolute quality floor.", "numbers"),
      bulletItem("Fix v4 adversary default-to-accept: When verdict parsing fails, default to \"reject\" (fail-closed) rather than \"accept\" (fail-open). Log a warning.", "numbers"),

      heading("P1: Should Fix Before Experiment", HeadingLevel.HEADING_2),
      bulletItem("Implement v2a/v2b/v2c model diversity: Add model assignment logic in factory.py to enforce same-model vs cross-model reviewer configurations.", "numbers"),
      bulletItem("Add v3a/v3b/v3c config variants: Three variant config files pointing to different merge templates.", "numbers"),
      bulletItem("Implement critical blocker extraction: Parse the \"blockers\" field from evaluator JSON and persist/aggregate.", "numbers"),
      bulletItem("Add seed management: Pass seed parameter to Anthropic/OpenAI APIs where supported. Track seed in runs table.", "numbers"),
      bulletItem("Replace Krippendorff\u2019s alpha proxy: Install krippendorff package and use proper ordinal alpha computation.", "numbers"),

      heading("P2: Should Fix Before Thesis Submission", HeadingLevel.HEADING_2),
      bulletItem("Implement v6 phase structure and coherence pass: Add phase-specific prompts and a dedicated coherence verification round.", "numbers"),
      bulletItem("Implement v7 epsilon-based convergence: Add intermediate scoring between rounds and compare against \u03B5 = 0.3 threshold.", "numbers"),
      bulletItem("Add per-dimension delta analysis: Compute score improvement per rubric dimension over v1b baseline.", "numbers"),
      bulletItem("Add convergence speed metric: Track quality trajectory across rounds for iterative variants.", "numbers"),
      bulletItem("Add novelty/diversity metric: Compute embeddings of design sections and measure cosine similarity across repetitions.", "numbers"),
      bulletItem("Add missing test coverage: Loader, heatmaps, exporter, checkpoint/resume, template variable completeness.", "numbers"),

      new Paragraph({ children: [new PageBreak()] }),

      // ===== CONCLUSION =====
      heading("8. Conclusion", HeadingLevel.HEADING_1),
      para("The LLM Consortium codebase is a well-engineered implementation that covers the majority of the thesis\u2019s ambitious scope. The core pipeline (configuration \u2192 agent orchestration \u2192 evaluation \u2192 storage \u2192 analysis) is complete and functional. All 8 variant orchestrators run, all 7 agent types are properly implemented, the evaluation pipeline handles multi-run scoring with median aggregation, and the analysis module provides the statistical tests needed for the thesis\u2019s conclusions."),
      para("The critical gaps center on experimental validity rather than functionality: the task portfolio mismatch means the thesis\u2019s specific predictions cannot be tested as written, the missing sub-variants prevent key pairwise comparisons, and the absence of run randomization and seed management weakens reproducibility claims. These are addressable issues\u2014most require configuration changes or modest code additions rather than architectural rework."),
      para("With the P0 and P1 fixes applied (estimated 3\u20135 days of work), the pipeline would be ready for the full 320-run experiment. The P2 items can be addressed during analysis and thesis writing without blocking experiment execution."),
    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("/sessions/pensive-peaceful-tesla/mnt/llm-consortium/gap_analysis_report.docx", buffer);
  console.log("Document created successfully.");
});
