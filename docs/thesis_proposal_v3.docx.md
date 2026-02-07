

**LLM CONSORTIUM FOR**

**SOFTWARE DESIGN REFINEMENT**

*A Multi-Agent Framework for Collaborative Design Generation,*

*Review, and Iterative Refinement Using Large Language Models*

**Thesis Proposal & Experiment Plan**

{{AUTHOR\_NAME}}  
{{UNIVERSITY / DEPARTMENT}}  
{{DATE}}

# **Table of Contents**

# **Abstract**

Large Language Models (LLMs) have demonstrated remarkable capability in generating software system designs and application architectures. However, a single model operating in isolation is susceptible to mode collapse, anchoring bias, and blind spots that degrade design quality, particularly for complex systems. This thesis investigates whether multi-LLM consortia, where multiple models collaborate through structured workflows of generation, review, critique, and synthesis, can systematically produce higher-quality software designs than a single model iterating alone.

We define a rigorous experimental framework comprising 8 consortium variants organized along a 2×2×2 design space (Authority × Roles × Dynamics), a standardized prompt and rubric system for both system-level and application-level design, and an automated LLM-based evaluation pipeline. Each variant is tested across 8 design tasks of varying complexity, with 5 repetitions per task per variant, yielding 320 experimental runs. We measure quality (weighted rubric score), cost-efficiency (quality per token), variance (reliability), and coherence (internal consistency).

Our hypotheses predict that: (1) parallel-generation variants escape mode collapse and achieve highest peak quality on complex tasks, (2) adversarial dynamics improve robustness but increase variance, (3) specialist reviewers outperform generalist reviewers on per-dimension quality, and (4) the quality of the evaluation rubric is the single largest lever, outweighing the collaboration topology. The contribution of this work is a decision framework practitioners can use to select the optimal consortium topology given their task complexity, quality threshold, and token budget.

# **1\. Introduction**

## **1.1 Problem Statement**

Software system design is among the highest-leverage activities in engineering. A well-designed system is maintainable, scalable, and resilient; a poorly designed one accrues compounding technical debt. LLMs have shown strong ability to generate system designs when given structured prompts. However, single-model generation suffers from three fundamental limitations:

* Mode collapse: The model commits to an architectural approach early and optimizes locally, unable to explore fundamentally different solutions.

* Anchoring bias: Self-refinement polishes surface issues while leaving structural flaws intact, because the model that created the flaw shares the reasoning that led to it.

* Blind spots: A single model’s training distribution creates systematic gaps—it may consistently undertreat error handling, security, or operability.

Human engineering organizations solve these problems through collaboration: architecture review boards, code reviews, red-team exercises, and cross-functional specialist input. This thesis asks whether similar collaborative structures, implemented as multi-LLM workflows, can produce analogous improvements.

## **1.2 Research Questions**

This thesis addresses the following research questions:

1. RQ1: Under what conditions does multi-LLM collaboration outperform single-LLM iteration for software design tasks, when controlling for total token budget?

2. RQ2: Which collaboration topology (centralized vs. decentralized authority, homogeneous vs. specialized roles, cooperative vs. adversarial dynamics) produces the best quality-to-cost ratio?

3. RQ3: How does task complexity interact with consortium topology? Do different topologies excel at different complexity levels?

4. RQ4: Is the collaboration topology or the evaluation rubric the primary driver of design quality improvement?

## **1.3 Contributions**

* A 2×2×2 taxonomy of multi-LLM collaboration topologies for design tasks, with 8 concrete variant implementations.

* A standardized prompt and evaluation framework for both system design and application design, including rubrics with weighted dimensions and concrete anchors.

* An automated experiment pipeline producing 320 experimental runs with reproducible results.

* A practitioner-facing decision framework: given task complexity X, budget Y, and quality threshold Z, use variant V.

* Empirical evidence on whether the evaluation rubric or the collaboration topology is the stronger lever for quality improvement.

## **1.4 Scope and Limitations**

This thesis focuses on software design documents (system architecture and application architecture) as the output artifact. It does not extend to code generation, testing, deployment, or other SDLC phases, though the framework could be adapted for those. We evaluate design quality through LLM-based rubric scoring, validated against a human expert subset. We do not claim that LLM-evaluated quality perfectly correlates with real-world system success; rather, we demonstrate relative ordering between variants under a consistent evaluation methodology.

# **2\. The 2×2×2 Design Space**

## **2.1 Axis Definitions**

Every multi-agent variant is classified along three orthogonal axes. This creates 8 cells, of which we populate 6 with concrete variants (plus 1 baseline outside the matrix). Two cells remain empty as future work.

| Axis | Option A | Option B |
| :---- | :---- | :---- |
| **Authority** | Centralized — one agent owns the final design and arbitrates conflicts | Decentralized — no single agent has final say; authority is shared, rotated, or synthesized |
| **Roles** | Homogeneous — all agents are general-purpose with the same capabilities | Specialized — agents have distinct expertise, personas, or forced positions |
| **Dynamics** | Cooperative — agents help, improve, and build on each other’s work | Adversarial — agents challenge, oppose, reject, or argue against each other |

## **2.2 Matrix Placement**

The following table shows where each variant falls in the 2×2×2 matrix. v1 (baseline) sits outside the matrix as the experimental control.

|  |  | Homo \+ Coop | Homo \+ Adv | Spec \+ Coop | Spec \+ Adv |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Centralized** | 🏛️ | v2 Leader \+ Reviewers | v4 Adversarial Reviewer | v5 Specialist Panel | ⬜ Future work |
| **Decentralized** | 🌐 | v3 Parallel \+ Merge, v7 Consensus | v8 Structured Debate | v6 Rotating Leader | ⬜ Future work |

*Note: v3 and v7 share the Decentralized/Homogeneous/Cooperative cell but differ in synthesis mechanism. v3 uses a centralized merger agent; v7 uses decentralized peer convergence. This distinction is itself a key experimental comparison.*

# **3\. Variant Specifications**

This chapter provides the complete specification for each of the 8 variants. For each variant, we define: the agents involved, the step-by-step workflow, the research question it addresses, our prediction, the key risk, and any sub-variants.

## **3.1 v1 — Single Leader (Baseline)**

**Matrix Position:** Outside matrix (experimental control)

**Agents:** 1 design agent \+ 1 evaluator

**Sub-variants:** v1a: single shot, no refinement (absolute floor). v1b: self-refinement with evaluator loop, N rounds (fair baseline).

**Workflow Steps:**

5. Leader generates full design from task prompt \+ design rubric.

6. Evaluator scores the design against the evaluation rubric, producing per-dimension scores and qualitative weaknesses.

7. Leader reads evaluator feedback and revises the design.

8. Repeat steps 2–3 for N rounds (N matched to other variants’ iteration count).

9. Record final evaluator score.

**Research Question:** What is the quality ceiling of a single model iterating with structured feedback? This is the cost floor and the bar every consortium variant must beat at equal token budget.

**Prediction:** Performs surprisingly well on simple/medium tasks. Falls short on complex tasks due to mode collapse — the model optimizes locally around its initial approach and cannot escape its early architectural commitment.

**Key Risk:** Anchoring bias: if the initial design has a structural flaw, self-refinement may polish the surface without fixing the foundation.

## **3.2 v2 — Leader \+ Reviewers**

**Matrix Position:** Centralized × Homogeneous × Cooperative

**Agents:** 1 leader \+ K reviewers (K=2–3) \+ 1 evaluator

**Sub-variants:** v2a: same-model self-review. v2b: cross-model review. v2c: multi-model panel review.

**Workflow Steps:**

10. Leader generates full design from task prompt \+ rubric.

11. K reviewers independently critique the design (qualitative feedback, not just scores).

12. Leader receives all K critiques simultaneously.

13. Leader revises the design, addressing reviewer feedback.

14. Evaluator scores the revised design.

15. Optionally repeat steps 2–5 for additional rounds.

**Research Question:** Does external qualitative critique add value beyond structured self-evaluation (v1b)? Does cross-model review beat same-model review?

**Prediction:** Modest improvement over v1b (5–10%) on dimensions where the leader had blind spots. Cross-model review outperforms self-review specifically on error handling and edge cases.

**Key Risk:** Reviewer feedback may be generic (‘improve error handling’) rather than actionable.

## **3.3 v3 — Parallel Leaders \+ Merge**

**Matrix Position:** Decentralized × Homogeneous × Cooperative

**Agents:** K parallel leaders (K=3) \+ 1 merger agent \+ 1 evaluator

**Sub-variants:** v3a: naive merge (section-by-section best-of). v3b: rubric-guided merge. v3c: dialectical merge (must resolve contradictions).

**Workflow Steps:**

1. K leaders independently generate full designs in parallel (no communication).

2. Merger agent receives all K designs.

3. Merger identifies the strongest approach per dimension across all designs.

4. Merger synthesizes one coherent design, resolving contradictions explicitly.

5. Evaluator scores the merged design.

6. Optionally: merged design goes through a v2-style review round.

**Research Question:** Does independent parallel generation escape mode collapse? Does architectural diversity in inputs produce a better synthesis than any single design? How critical is the merge strategy?

**Prediction:** Highest peak quality on complex tasks where the solution space is large. The merge step is the bottleneck — a bad merger destroys good inputs. Dialectical merge outperforms naive merge by 15–20%.

**Key Risk:** Frankenstein risk: merged design may combine individually good ideas that are architecturally incompatible.

## **3.4 v4 — Leader \+ Adversarial Reviewer**

**Matrix Position:** Centralized × Homogeneous × Adversarial

**Agents:** 1 leader \+ 1 adversarial reviewer \+ 1 evaluator

**Sub-variants:** Tuning levers: rejection threshold, max rounds (3–5), adversarial intensity.

**Workflow Steps:**

1. Leader generates full design from task prompt \+ rubric.

2. Adversarial reviewer attempts to reject: must cite specific rubric dimensions below threshold.

3. If rejected: leader receives rejection rationale and must revise.

4. If accepted: design proceeds to final evaluation.

5. Repeat steps 2–4 up to N rounds (hard cap \= 5).

6. Final evaluator scores accepted design (or last revision if cap reached).

**Research Question:** Does adversarial pressure produce more robust designs than cooperative feedback (v2)? Does the threat of binary rejection force the leader to address weaknesses it would otherwise rationalize away?

**Prediction:** Highest variance variant. When it works, produces the most defensively robust designs. When it fails, wastes tokens on unproductive rejection loops.

**Key Risk:** Infinite rejection loop if threshold too strict; rubber-stamping if too lenient. Threshold tuning is the key hyperparameter.

## **3.5 v5 — Specialist Panel**

**Matrix Position:** Centralized × Specialized × Cooperative

**Agents:** 1 leader \+ N specialist reviewers (N=3–5) \+ 1 evaluator

**Workflow Steps:**

1. Leader generates full design from task prompt \+ rubric.

2. Each specialist reviews ONLY their domain: Domain Modeling, Reliability, Testability, Security, Operability.

3. Leader receives all specialist critiques simultaneously.

4. Leader revises the design, weighing specialist feedback per area.

5. Evaluator scores the revised design.

**Research Question:** Does role specialization in reviewers outperform general-purpose review (v2)? Do specialist reviewers catch domain-specific flaws that generalist reviewers miss?

**Prediction:** Highest per-dimension scores in specialist areas (e.g., Testability specialist lifts testability from 3→4.5). Risk of conflicting recommendations across specialists. Overall 10–15% improvement over v2.

**Key Risk:** Specialist recommendations may conflict (security wants isolation; operability wants simple access). Leader must arbitrate trade-offs.

## **3.6 v6 — Rotating Leader**

**Matrix Position:** Decentralized × Specialized × Cooperative

**Agents:** K agents (K=3–4) rotating through phases \+ 1 evaluator

**Workflow Steps:**

1. Phase 1: Agent A leads (domain model \+ module structure). B,C review.

2. Phase 2: Agent B leads (error handling \+ data flow). A,C review.

3. Phase 3: Agent C leads (testing \+ operability). A,B review.

4. Optional Phase 4: coherence pass — any agent flags cross-phase contradictions.

5. Evaluator scores the final integrated design plus a separate coherence metric.

**Research Question:** Does epistemic reset (fresh eyes at each phase) prevent anchoring bias? What is the coherence cost of leadership rotation?

**Prediction:** Strong individual section quality. Weakest cross-section consistency. Coherence score will be the Achilles heel.

**Key Risk:** Coherence degradation: each leader improves their phase while potentially violating assumptions from earlier phases.

## **3.7 v7 — Consensus Convergence**

**Matrix Position:** Decentralized × Homogeneous × Cooperative

**Agents:** K peer agents (K=3) with equal authority \+ 1 evaluator

**Sub-variants:** Convergence criteria: ε \= 0.3, hard cap \= 4 rounds.

**Workflow Steps:**

1. K agents independently generate full designs (same starting point as v3).

2. Each agent reviews the other K−1 designs and writes qualitative critique.

3. Each agent revises their own design based on peer feedback.

4. Evaluator scores all K revised designs.

5. Convergence check: if score delta \< ε across agents, stop. Else repeat from step 2\.

6. Final output \= highest-scoring design at convergence.

7. Record pre-convergence best individual score vs. post-convergence score.

**Research Question:** Can models self-organize toward quality without centralized authority? Does peer pressure drive convergence to excellence or to mediocrity (committee effect)?

**Prediction:** Lowest variance across runs (most reliable quality). Rarely the highest peak — committee effect pushes toward safe designs. Post-convergence score often lower than best pre-convergence individual.

**Key Risk:** Groupthink: models may converge on shared biases rather than complementary strengths.

## **3.8 v8 — Structured Debate**

**Matrix Position:** Decentralized × Homogeneous × Adversarial

**Agents:** 2 debater agents \+ 1 judge agent \+ 1 evaluator

**Sub-variants:** Position pairs: event-sourcing vs. CRUD, microservices vs. monolith, SQL vs. NoSQL, sync vs. async.

**Workflow Steps:**

1. Assign opposing architectural positions (e.g., event-sourcing vs. CRUD).

2. Debater A produces full design defending Position A.

3. Debater B produces full design defending Position B.

4. Both debaters write rebuttals: why the other approach is worse for THIS problem.

5. Judge reads all four documents (2 designs \+ 2 rebuttals).

6. Judge produces final design: may pick winner, synthesize, or find a third way.

7. Evaluator scores the judge’s final output.

**Research Question:** Does forced architectural disagreement explore the solution space more effectively than random parallel work (v3)? Do rebuttals surface trade-offs that cooperative review misses?

**Prediction:** Strongest on tasks with genuine architectural trade-offs. Weakest when one position is clearly inferior. The rebuttal step is the key value-add.

**Key Risk:** Position assignment matters enormously. If one position is clearly wrong, half the budget is wasted on a strawman.

# **4\. Prompt Architecture**

The quality of a consortium’s output is bounded by the quality of its prompts. This chapter defines the three-layer prompt architecture used across all variants.

## **4.1 Layer 1: The Design Generation Prompt**

Every design agent (leader, parallel leader, debater) receives the same structured generation prompt. This prompt consists of five components:

* Task Definition: the system or application to design, the problem statement, hard constraints, and prioritized use cases. These are parameterized per design task (see Chapter 6).

* Positive Examples (3–4): complete worked examples of good design patterns — clean separation of concerns, honest trade-off analysis, evolutionary architecture. These prime the model with the quality standard we expect.

* Negative Examples (3–4): anti-patterns with explanations — resume-driven architecture, vague hand-waving, ignoring operational reality. These establish guardrails against common LLM failure modes.

* Required Deliverables (12–14 sections): an exhaustive checklist of what the design must contain — requirements clarification, capacity estimation, high-level architecture, data model, API design, critical path deep-dive, scalability strategy, reliability design, security, observability, trade-off summary table, and evolution path. The model cannot skip sections.

* Meta-Instructions: behavioral directives (“be opinionated,” “be specific,” “right-size the solution,” “think about Day 2”) that shape the model’s reasoning approach.

Two versions of this prompt exist: one for system design (infrastructure-level) and one for application design (code-architecture-level). Both follow the same 5-component structure but differ in deliverable sections and examples.

## **4.2 Layer 2: The Review / Critique Prompt**

Reviewer agents (in v2, v4, v5, v6, v7) receive a critique prompt that includes:

* The complete design document to review.

* The evaluation rubric (same rubric used by the final evaluator) so the reviewer knows what dimensions matter and how they’re weighted.

* Role-specific instructions: for general reviewers (v2), “provide qualitative feedback on all dimensions.” For specialist reviewers (v5), “you are the Testability Specialist — review ONLY the testing architecture, dependency inversion, and faking strategy.” For adversarial reviewers (v4), “you must ACCEPT or REJECT. To reject, cite specific rubric dimensions scoring below 3.”

* Output format: structured critique with specific, actionable feedback per dimension. Not scores, but observations and suggestions.

## **4.3 Layer 3: The Evaluation Prompt**

The evaluator agent is the final scoring authority. It receives:

* The final design document.

* The full evaluation rubric with 11–12 weighted dimensions (see Chapter 5).

* Scoring instructions: for each dimension, compare against the “What 5 Looks Like” and “What 1 Looks Like” anchors, assign 1–5, justify with 2–3 sentences citing specific passages.

* Output format: dimension score table, weighted total, letter grade, top 3 strengths, top 3 weaknesses, actionable improvements, and any Critical Blockers.

Critically, the evaluator prompt is identical across all variants. This ensures that quality comparisons between variants are measured on the same scale.

## **4.4 Prompt Consistency Controls**

To ensure experimental validity:

* All agents across all variants use the same base model temperature (configurable, default 0.7 for generation, 0.3 for evaluation).

* The task definition section is identical across variants — only the workflow structure differs.

* System prompts (role definitions) are the only element that varies between agents within a variant.

* All prompts are version-controlled and frozen before the experiment begins.

# **5\. Evaluation Rubric**

The evaluation rubric is the backbone of this experiment. It must be precise enough to produce consistent scores across runs, and rich enough to capture meaningful quality differences. Two rubrics are used: one for system design, one for application design.

## **5.1 System Design Rubric (11 Dimensions)**

| \# | Dimension | Weight | What ‘5’ Looks Like (summary) |
| :---- | :---- | :---- | :---- |
| 1 | **Requirements Understanding** | 8% | Restates, clarifies ambiguities, makes assumptions |
| 2 | **Capacity & Estimation Rigor** | 8% | Clear math with stated assumptions, consistent numbers |
| 3 | **Architectural Clarity** | 12% | Clear responsibilities, explicit communication, diagram matches prose |
| 4 | **Data Model Quality** | 10% | Entities well-defined, storage justified, indexing aligned to queries |
| 5 | **API Design Quality** | 8% | Intuitive, consistent, versioned, error handling specified |
| 6 | **Scalability & Performance** | 12% | Real bottlenecks identified, concrete strategy, targets tied to math |
| 7 | **Reliability & Fault Tolerance** | 12% | Every SPOF mitigated, failure modes enumerated, RPO/RTO stated |
| 8 | **Security Posture** | 8% | Auth model defined, encryption strategy, input validation, secrets mgmt |
| 9 | **Operability** | 10% | Observability stack defined, alerts identified, deployment strategy clear |
| 10 | **Trade-off Honesty** | 7% | Every decision includes alternatives, rationale, and downside accepted |
| 11 | **Right-Sizing / Simplicity** | 5% | Complexity proportional to problem, no unnecessary abstractions |

## **5.2 Application Design Rubric (12 Dimensions)**

The application design rubric follows the same structure but focuses on code-architecture concerns: Domain Model Quality (14%), Module Structure & Dependencies (12%), Testability (12%), Error Handling (10%), Interface & Contract Design (10%), Cross-Cutting Concerns (6%), State Management (6%), Concurrency Safety (6%), and others. Full rubric with anchors is included in Appendix A.

## **5.3 Scoring Methodology**

For each dimension:

7. The evaluator LLM reads the relevant section(s) of the design.

8. It compares against the “What 5 Looks Like” and “What 1 Looks Like” anchors.

9. It assigns an integer score from 1–5.

10. It writes 2–3 sentences justifying the score with specific citations from the design.

11. If a section is entirely missing, it scores 1 and notes the absence.

After scoring all dimensions:

* Weighted total: sum(score\_i × weight\_i) yields a score out of 5.0.

* Grade mapping: 4.5–5.0 \= A, 3.5–4.4 \= B, 2.5–3.4 \= C, 1.5–2.4 \= D, 1.0–1.4 \= F.

* Critical Blockers are flagged independently of the score (e.g., circular dependencies, unenforced invariants around money).

## **5.4 Evaluator Consistency Validation**

Since the evaluator is itself an LLM, we must validate its reliability:

* Intra-evaluator reliability: Run the same design through the evaluator 5 times at temperature 0.3. If the standard deviation exceeds 0.5 points, the evaluator prompt must be refined.

* Inter-evaluator agreement: Use 3 different evaluator models (e.g., Claude Sonnet, GPT-4o, Gemini Pro). Compute Krippendorff’s alpha across their scores. Target: α \> 0.7.

* Human validation: A human expert scores a random 10% subset (32 designs). Compute Spearman rank correlation between human and LLM scores. Target: ρ \> 0.75.

# **6\. Design Tasks (Test Suite)**

To generalize our findings, we test each variant across 8 design tasks spanning three complexity levels (Simple, Medium, Complex) and two design types (System, Application). This yields a task suite that exercises different strengths and weaknesses of each consortium variant.

## **6.1 Task Portfolio**

| ID | Task | Complexity | Type | Key Challenge |
| :---- | :---- | :---- | :---- | :---- |
| **T1** | URL Shortener | Simple | System | CRUD, redirect, analytics. Well-understood, one reasonable approach. |
| **T2** | Rate Limiter Service | Simple | Application | Token bucket / sliding window. Concurrency focus. |
| **T3** | Notification System | Medium | System | Multi-channel (email, SMS, push). Event-driven, preference routing. |
| **T4** | Task Queue / Job Scheduler | Medium | Application | Reliability focus: at-least-once delivery, retry, dead-letter. |
| **T5** | Multi-Tenant SaaS Billing | Complex | System | Metering, invoicing, plan management. Financial correctness critical. |
| **T6** | Real-Time Collaborative Editor | Complex | System | CRDT/OT, conflict resolution, presence. Distributed systems challenge. |
| **T7** | Rule-Based Access Control Module | Complex | Application | RBAC/ABAC hybrid. Policy engine, audit trail. Security critical. |
| **T8** | Event-Driven Order Processing | Complex | Application | Saga pattern, compensating transactions. Reliability \+ domain modeling. |

## **6.2 Complexity Hypothesis**

We hypothesize that consortium topology interacts with task complexity:

* Simple tasks (T1, T2): v1b (single leader with self-refinement) matches or exceeds all consortium variants at lower cost. The solution space is small enough that mode collapse is not a problem.

* Medium tasks (T3, T4): v2 and v5 (leader \+ reviewers/specialists) provide measurable improvement by catching blind spots in error handling and edge cases.

* Complex tasks (T5–T8): v3 (parallel \+ merge) and v8 (debate) achieve highest peak quality by generating genuinely different architectural approaches. The diversity of initial designs is the key lever.

## **6.3 Task Specification Template**

Each task is specified using the following template, which maps directly to the {{PLACEHOLDERS}} in the design generation prompt:

* System/Application Name

* Problem Statement (2–4 sentences)

* Hard Constraints: performance targets, compliance requirements, team size, timeline, existing tech stack

* Key Use Cases: prioritized P0/P1/P2 list

* Expected Complexity Drivers: what makes this task hard (concurrency, distributed state, financial correctness, etc.)

Complete task specifications for all 8 tasks are provided in Appendix B.

# **7\. Experiment Protocol**

## **7.1 Run Configuration**

| Parameter | Value |
| :---- | :---- |
| **Variants** | 8 (v1, v2, v3, v4, v5, v6, v7, v8) \+ sub-variants for v1, v2, v3 |
| **Design tasks** | 8 (T1–T8, spanning Simple/Medium/Complex) |
| **Repetitions per (variant, task)** | 5 (for variance measurement) |
| **Total experimental runs** | 8 variants × 8 tasks × 5 reps \= 320 runs (+ sub-variant runs) |
| **Max revision rounds per run** | 3 (matched across all variants for fair comparison) |
| **Generation model temperature** | 0.7 (for diversity in generation) |
| **Evaluator model temperature** | 0.3 (for consistency in evaluation) |
| **Evaluator runs per design** | 3 (median taken as final score) |
| **Design generation model(s)** | {{PRIMARY\_MODEL}} (e.g., Claude Sonnet 4.5 / GPT-4o / Gemini 2.5 Pro) |
| **Evaluator model** | {{EVALUATOR\_MODEL}} (recommend different family from generation model) |

## **7.2 Execution Pipeline**

Each experimental run follows this pipeline:

7. Task Selection: select task T\_i and load its specification.

8. Variant Initialization: instantiate the selected variant’s agent constellation (leader, reviewers, specialists, etc.) with their respective prompts.

9. Design Generation: execute the variant’s workflow (see Chapter 3 for per-variant steps). Record all intermediate outputs.

10. Token Accounting: record total input \+ output tokens consumed across all agents, all rounds.

11. Evaluation: feed the final design to the evaluator 3 times. Take the median per-dimension score.

12. Coherence Check (for multi-agent variants): run a separate coherence evaluation pass asking “do sections X and Y contradict each other?” for all section pairs.

13. Artifact Storage: save the final design, all intermediate designs, all critiques, all evaluator outputs, and token counts to the experiment database.

## **7.3 Controls and Validity**

To ensure experimental validity, the following controls are applied:

* Token budget control: for each (variant, task) pair, record total tokens consumed. All analyses include a cost-adjusted comparison (quality per 1M tokens).

* Prompt freezing: all prompts (generation, review, evaluation) are version-controlled and frozen before the first run. No prompt changes during the experiment.

* Randomization: the order of (variant, task, repetition) combinations is randomized to prevent systematic effects from API rate changes, model updates, or time-of-day effects.

* Evaluator blinding: the evaluator receives only the final design document with no metadata about which variant produced it.

* Seed management: where the API supports it, random seeds are set per repetition for reproducibility.

## **7.4 Pairwise Comparisons**

Beyond the overall variant ranking, we conduct targeted pairwise comparisons that isolate a single experimental variable each:

| Comparison | Variable Isolated | Hypothesis |
| :---- | :---- | :---- |
| **v1b vs. v2a** | Feedback source: rubric scores vs. qualitative critique (same model) | Qualitative critique provides more actionable revision guidance |
| **v2a vs. v2b** | Reviewer identity: same-model vs. cross-model (same format) | Different training distribution surfaces different blind spots |
| **v2 vs. v4** | Feedback dynamics: cooperative vs. adversarial (same topology) | Adversarial pressure produces more robust designs |
| **v2 vs. v5** | Reviewer specialization: general vs. domain-expert (same topology) | Specialists catch deeper flaws in their domain |
| **v3 vs. v7** | Synthesis mechanism: central merger vs. peer convergence | Central synthesis → higher peak; peer convergence → more consistency |
| **v3 vs. v8** | Diversity source: random vs. forced opposition | Forced disagreement explores solution space more thoroughly |
| **v4 vs. v8** | Adversarial structure: critique-only vs. build+critique | Constructive adversarial \> destructive adversarial |
| **v5 vs. v6** | Specialization target: reviewer expertise vs. leader expertise | Specialized leaders \> specialized reviewers |

## **7.5 How Repetitions Feed Into Pairwise Comparisons**

Each (variant, task) pair is executed 5 times, producing 5 independent quality scores. For pairwise comparisons, we do not select a single repetition. Instead, we compare the full distributions using a three-layered analysis:

**Layer 1 — Per-Task Descriptive Comparison.** For each task T\_i, report the median and interquartile range (IQR) of the 5 scores for each variant in the pair. Example: “On T5 (SaaS Billing), v2a median \= 4.2 (IQR 4.0–4.4) vs. v1b median \= 3.9 (IQR 3.7–4.1).” This layer tells the qualitative story of where the improvement shows up.

**Layer 2 — Pooled Statistical Test (All Tasks).** Compute the paired score difference (variant\_A – variant\_B) for each of the 5 repetitions across all 8 tasks, yielding 40 paired observations. Apply the Wilcoxon signed-rank test on these 40 differences to determine whether the improvement is statistically significant. Apply Bonferroni correction for the 8 pairwise comparisons (adjusted α \= 0.05/8 \= 0.00625). This is the primary test for determining whether one variant reliably outperforms another.

**Layer 3 — Stratified by Complexity.** Repeat the Layer 2 test separately for each complexity level: Simple (2 tasks × 5 reps \= 10 pairs), Medium (2 tasks × 5 reps \= 10 pairs), Complex (4 tasks × 5 reps \= 20 pairs). This tests the interaction hypothesis: does the consortium advantage depend on task complexity? A variant may show no significant improvement on Simple tasks but a large, significant improvement on Complex tasks.

**Repetition Pairing Strategy.** Repetitions are paired by index: repetition 1 of variant A is compared to repetition 1 of variant B for the same task. To ensure this pairing is meaningful, both variants use the same random seed per repetition (where supported by the API). If seeds are not supported, we acknowledge this as a limitation and supplement with an unpaired Mann-Whitney U test as a robustness check. The pairing ensures that any variation due to API stochasticity is shared across the pair, isolating the effect of the variant topology.

**Effect Size Reporting.** Beyond p-values, we report effect size using the rank-biserial correlation (r) for each Wilcoxon test, and Cohen’s d for the mean difference. A statistically significant result with a tiny effect size (d \< 0.2) would indicate the improvement is real but practically negligible — important for the cost-efficiency analysis, since even a real but small improvement may not justify the token cost premium of a consortium variant.

## **7.6 End-to-End Pipeline: Who Does What**

To eliminate any confusion about which actor performs which function in the experiment, the following diagram traces a single experimental run from design generation through to thesis-level interpretation. Each stage has exactly one responsible actor, and the handoff between stages is always structured data (never ambiguous natural language instructions).

| Stage | Actor | What It Does | Output | Judgment Involved? |
| :---- | :---- | :---- | :---- | :---- |
| **1** | **Design LLM(s)** | Generate, review, critique, revise designs per the variant’s workflow (Ch. 3\) | Final design document (text) | Yes — creative/architectural judgment. This is the experimental treatment. |
| ▼ | *handoff: design doc* | *Blinded: no variant metadata attached* |  |  |
| **2** | **Evaluator LLM** | Score design against rubric (Ch. 5). Run 3 times per design; take median. | Per-dimension scores \+ weighted total (numbers) | Yes — qualitative judgment to assign scores. This is the only step where an LLM makes an evaluative judgment. |
| ▼ | *handoff: scores CSV* | *Structured data: variant, task, rep, dim1..dim11, total* |  |  |
| **3** | **Python Script** | Compute descriptive stats, run Wilcoxon signed-rank tests, compute Cohen’s d, generate box plots and Pareto charts (Ch. 8\) | p-values, effect sizes, charts, rankings | **No — pure arithmetic.** Deterministic math on numbers from Stage 2\. No LLM, no human subjectivity. |
| ▼ | *handoff: stats \+ charts* | *Tables, plots, p-values ready for thesis* |  |  |
| **4** | **Human Researcher** | Interpret results, write case studies, build decision framework, write thesis narrative (Ch. 10\) | Thesis chapters, decision framework, conclusions | Yes — interpretive judgment. Only a human decides what the findings mean and what to recommend. |

**Key principle:** LLM judgment happens exactly twice — once to generate the design (Stage 1\) and once to score it (Stage 2). Everything downstream of Stage 2 is deterministic computation or human interpretation. The pairwise comparison in Section 7.4 lives entirely in Stage 3: it is a statistical test on numbers, not a judgment call by any LLM or human. This separation ensures that the comparison methodology itself introduces no additional subjectivity beyond what the evaluator LLM already contributes.

# **8\. Metrics & Analysis Plan**

## **8.1 Primary Metrics**

| Metric | How to Measure | Why It Matters |
| :---- | :---- | :---- |
| **Quality Score** | Weighted rubric score (evaluator LLM), median of 3 evaluator runs, per design | Primary outcome variable — did the variant produce a better design? |
| **Cost (tokens)** | Total input \+ output tokens across all agents, all rounds | Fair comparison requires equal-cost normalization |
| **Cost-Efficiency** | Quality score ÷ total tokens consumed (quality per 1M tokens) | The Pareto frontier — the practitioner-facing finding |
| **Variance** | Std. deviation of quality score across 5 repetitions per (variant, task) | Is the variant reliably good, or just occasionally lucky? |
| **Coherence** | Separate evaluator pass: section-pair contradiction check (binary per pair) | Critical for v3, v6, v7 where multiple agents build different sections |
| **Convergence Speed** | Rounds to reach score plateau (\< 0.1 improvement per round) | How quickly does iterating stop being worth the cost? |
| **Per-Dimension Delta** | Per-dimension score improvement over v1b baseline | Identifies WHERE each variant adds value, not just overall |
| **Blocker Count** | Number of Critical Blockers flagged by evaluator | A design at 3.8 with zero blockers may beat 4.2 with one blocker |
| **Novelty / Diversity** | Across 5 runs: architectural similarity score (cosine similarity of section embeddings) | Does the variant produce diverse solutions or the same thing every time? |

## **8.2 Analysis Plan**

The analysis proceeds in four stages:

**Stage 1: Descriptive Statistics**

* For each (variant, complexity level): mean quality score, std. deviation, min, max, and token cost.

* Box plots of quality scores per variant, faceted by complexity level.

* Cost-quality scatter plot with Pareto frontier highlighted.

**Stage 2: Statistical Tests**

* Friedman test across all variants (non-parametric repeated measures) to confirm that variant choice matters.

* Pairwise Wilcoxon signed-rank tests for each comparison in Section 7.4, with Bonferroni correction for multiple comparisons.

* Two-way interaction analysis: variant × complexity to test whether certain variants excel at certain complexity levels.

**Stage 3: Cost-Adjusted Analysis**

* Pareto frontier: which variants are on the quality-vs-cost frontier?

* Break-even analysis: at what quality threshold does each consortium variant justify its cost premium over v1b?

* Token budget simulation: if you had a fixed budget of N tokens, which variant allocation maximizes expected quality?

**Stage 4: Qualitative Analysis**

* For the top-scoring and bottom-scoring design in each (variant, task) pair: manual inspection of what worked and what failed.

* Thematic coding of evaluator feedback to identify recurring strengths/weaknesses per variant.

* Case studies: select 3 designs where a consortium variant dramatically outperformed v1b, and 3 where it didn’t, and analyze why.

## **8.3 Reporting**

The final report includes:

* A variant ranking table (overall and per-complexity-level).

* The cost-quality Pareto frontier chart.

* A heatmap: variant × rubric dimension showing where each variant adds (or loses) value relative to v1b.

* The practitioner decision framework: given complexity X, budget Y, quality threshold Z, use variant V.

* A recommendation on whether the evaluation rubric or the collaboration topology is the stronger quality lever (RQ4).

# **9\. Project Plan & Timeline**

## **9.1 Phase Breakdown**

| Phase | Activity | Deliverable | Duration |
| :---- | :---- | :---- | :---- |
| **Phase 1** | Prompt Engineering & Rubric Finalization | Frozen prompt templates \+ evaluation rubrics | Weeks 1–3 |
| **Phase 2** | Orchestrator Development | Python framework: agent orchestration, API wrappers, token accounting | Weeks 3–6 |
| **Phase 3** | Pilot Runs (v1 \+ v2 only, T1 \+ T5) | Validate pipeline end-to-end, tune evaluator, fix bugs | Weeks 6–7 |
| **Phase 4** | Evaluator Validation | Intra-rater reliability, inter-model agreement, human validation subset | Weeks 7–8 |
| **Phase 5** | Full Experiment Execution | 320 runs \+ sub-variant runs. Approx. 2–3 days of API calls. | Weeks 8–10 |
| **Phase 6** | Analysis & Visualization | Statistical tests, Pareto frontier, heatmaps, case studies | Weeks 10–12 |
| **Phase 7** | Thesis Writing | Complete thesis document with all chapters | Weeks 12–16 |
| **Phase 8** | Review & Defense Prep | Revisions based on advisor feedback, defense slides | Weeks 16–18 |

## **9.2 Cost Estimate**

Cost depends heavily on which model serves as the primary design generator and which serves as the evaluator. The experiment is model-agnostic by design: all prompts use a standard API interface, so any model can fill any role. Below we present pricing for six candidate models, per-run token estimates by variant class, and seven budget scenarios spanning from under $100 (local inference) to over $1,000 (premium cloud).

**Table 9.2a — Model Pricing (per 1M tokens, as of February 2026\)**

| Model | Input | Output | Batch | Context | Notes |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Claude Sonnet 4.5** | $3.00 | $15.00 | 50% off | 200K (1M beta) | Best coding benchmarks; recommended primary model |
| **Claude Opus 4.5** | $5.00 | $25.00 | 50% off | 200K (1M beta) | Strongest reasoning; ideal evaluator or cross-model reviewer |
| **GPT-4o** | $2.50 | $10.00 | 50% off | 128K | Strong multimodal; good for cross-model reviewer roles |
| **GPT-4.1** | $2.00 | $8.00 | 50% off | 1M native | Cheapest major proprietary; 1M context enables long consortium threads |
| **Gemini 3 Pro** | $2.00 | $12.00 | TBD (preview) | 1M native | ≤200K std; \>200K doubles to $4/$18. Preview pricing may drop at GA. |
| **GPT-OSS 120B (local)** | **$0.00** | **$0.00** | N/A | 128K | Open-weight MoE (117B/5.1B active). Runs locally on Mac via Metal. Apache 2.0. |

**Table 9.2b — Per-Run Token Estimates and Cost by Variant Class**

| Variant Class | \~In Tok | \~Out Tok | Sonnet 4.5 | Opus 4.5 | GPT-4o | Gemini 3 | GPT-4.1 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| v1 (single, 3 rounds) | 30K | 20K | $0.39 | $0.65 | $0.28 | $0.30 | $0.22 |
| v2/v4/v5 (leader \+ reviewers) | 70K | 40K | $0.81 | $1.35 | $0.58 | $0.62 | $0.46 |
| v3/v7/v8 (parallel \+ synthesis) | 120K | 80K | $1.56 | $2.60 | $1.10 | $1.20 | $0.88 |
| *Evaluator (per design, 3× median)* | 15K ×3 | 2K ×3 | $0.23 | $0.38 | $0.17 | $0.16 | $0.14 |

**Table 9.2c — Total Budget Scenarios (320 core runs \+ 960 evaluator runs \+ pilot/overhead)**

| Scenario | Design Gen. | Evaluator | Core Cost | With Overhead |
| :---- | :---- | :---- | :---- | :---- |
| **A. Budget-min (local \+ API eval)** | GPT-OSS 120B | Sonnet 4.5 | **\~$74** | **\~$120** |
| **B. Local \+ Gemini eval (cheapest)** | GPT-OSS 120B | Gemini 3 Pro | **\~$51** | **\~$82** |
| **C. GPT-4.1 gen \+ Sonnet eval** | GPT-4.1 | Sonnet 4.5 | **\~$252** | **\~$400** |
| **D. Gemini gen \+ Sonnet eval** | Gemini 3 Pro | Sonnet 4.5 | **\~$316** | **\~$505** |
| **E. Sonnet-only (recommended)** | Sonnet 4.5 | Sonnet 4.5 | **\~$390** | **\~$625** |
| **F. Mixed (Sonnet gen \+ Opus eval)** | Sonnet 4.5 | Opus 4.5 | **\~$437** | **\~$700** |
| **G. Opus-only (premium)** | Opus 4.5 | Opus 4.5 | **\~$648** | **\~$1,037** |

*“With Overhead” includes a 1.6× multiplier for pilot runs, evaluator validation (Phase 4), sub-variant exploration, failed/retried calls, and prompt iteration during Phase 3\. Weighted average per-run cost computed from variant mix: 2 single-agent variants × 5 reps \+ 3 leader-reviewer variants × 5 reps \+ 3 parallel-synthesis variants × 5 reps \= 320 runs.*

**GPT-OSS 120B as a local option.** OpenAI’s open-weight GPT-OSS 120B (117B parameters, 5.1B active via MoE, Apache 2.0 license) can run locally on Apple Silicon via the Metal backend. Running the 320 design-generation runs on a local Mac eliminates all API costs for Stage 1, reducing the total experiment to evaluator-only API spend. The tradeoff is speed: local inference on consumer hardware is significantly slower than cloud APIs, so the execution timeline for Phase 5 would approximately double. With Gemini 3 Pro as the evaluator (Scenario B), the entire experiment could run for under $82 with overhead. With Sonnet 4.5 as the evaluator (Scenario A), under $120. GPT-OSS 120B benchmarks near parity with OpenAI o4-mini on reasoning tasks, making it a credible design generator for this experiment.

**Cross-provider scenarios.** GPT-4.1 offers the cheapest per-token cloud pricing at $2/$8 with a native 1M context window, making it ideal for multi-agent variants where long conversation threads accumulate tokens (v3/v7/v8). Gemini 3 Pro is competitive at $2/$12 but still in preview — pricing may drop \~25% at GA. Mixing providers (e.g., GPT-4.1 for generation, Sonnet 4.5 for evaluation) is architecturally straightforward since all communication is via text, though it introduces a confound if cross-model variants (v2b) are part of the comparison.

**Cost optimization levers.** Prompt caching (available on Anthropic and OpenAI) reduces input costs by up to 90% for the system prompt and rubric, which are identical across all 320 runs. The Anthropic Batch API provides a 50% discount for non-time-sensitive workloads with 24-hour turnaround — ideal for the evaluation phase. Google offers context caching at 75% savings on repeated prompts. Combining caching \+ batch: Scenario E could drop from \~$625 to \~$300. These optimizations are noted as budget contingencies rather than baseline assumptions.

**Recommended budget: $500–$700** (Scenario E or F with overhead), allowing headroom for cross-model evaluator validation runs required in Phase 4\. Budget-constrained path: Scenario A or B using local GPT-OSS 120B for generation, keeping total spend under $120.

## **9.3 Risk Mitigation**

* API outages/rate limits: build retry logic with exponential backoff; budget 2× the minimum time for Phase 5\.

* Model updates mid-experiment: pin model versions where API supports it (e.g., claude-sonnet-4-5-20250929). If not available, document model version per run.

* Evaluator unreliability: Phase 4 catches this before the full experiment. If α \< 0.7, refine the evaluator prompt before proceeding.

* Unexpected cost overrun: implement hard token caps per run and per variant. Abort runs that exceed 2× expected budget.

# **10\. Expected Contributions & Outcomes**

## **10.1 Predicted Findings**

Based on our theoretical analysis, we predict the following results (to be validated or refuted by the experiment):

6. For simple tasks (T1, T2), v1b matches all consortium variants at 3–5× lower cost. The consortium overhead is not justified.

7. For complex tasks (T5–T8), v3 (parallel \+ dialectical merge) achieves the highest peak quality score, outperforming v1b by 15–25%.

8. v7 (consensus convergence) produces the lowest variance across runs but the post-convergence score is lower than the best pre-convergence individual in 60%+ of cases (committee effect).

9. v4 (adversarial) has the highest variance: it produces the best reliability/error-handling scores when it works, and the worst cost-efficiency when it doesn’t.

10. v5 (specialist panel) outperforms v2 (general reviewers) by 10–15%, with improvements concentrated in the dimensions matching specialist expertise.

11. The evaluation rubric is the single largest quality lever. Replacing a mediocre rubric with a good one (controlling for variant) improves quality more than changing from v1b to the best consortium variant (controlling for rubric). This is RQ4 and potentially the most important finding.

## **10.2 Decision Framework (Target Deliverable)**

The ultimate practical deliverable of this thesis is a decision matrix that practitioners can use:

| Task Complexity | Budget Priority | Quality Priority | Recommended Variant | Expected Quality |
| :---- | :---- | :---- | :---- | :---- |
| Simple | Low cost | Good enough | **v1b** | 3.5–4.0 |
| Simple | Any | Maximum | **v2b** | 3.8–4.2 |
| Medium | Moderate | High | **v5** | 3.8–4.3 |
| Medium | Any | Maximum | **v5 \+ v2 combo** | 4.0–4.5 |
| Complex | Moderate | High | **v3b (rubric merge)** | 4.0–4.4 |
| Complex | Any | Maximum | **v3c (dialectical)** | 4.2–4.7 |
| Complex (trade-offs) | Any | Maximum | **v8 (debate)** | 4.2–4.7 |

*Note: This table represents our predicted outcomes. The actual decision framework will be populated with empirical data from the experiment.*

## **10.3 Academic Contribution**

This work contributes to the emerging field of multi-agent LLM systems by providing the first systematic, controlled comparison of consortium topologies for a structured generation task (software design). Unlike prior work that explores multi-agent debate or self-refinement in isolation, we test 8 variants in a unified framework with controlled variables, reproducible protocols, and a consistent evaluation methodology. The 2×2×2 taxonomy itself is a contribution, providing a language for describing and comparing multi-agent collaboration structures.

# **Appendix A: Full Evaluation Rubrics**

The complete system design and application design rubrics with full anchor descriptions for scores 1 through 5 on all dimensions are provided as separate reference documents (see accompanying files: system\_design\_prompt\_template.md and application\_design\_prompt\_template.md).

# **Appendix B: Task Specifications**

Complete task specifications for all 8 design tasks (T1–T8), including problem statements, hard constraints, use cases, and complexity drivers, are provided as separate parameterized prompt files. {{TO BE COMPLETED DURING PHASE 1}}

# **Appendix C: Raw Prompt Templates**

All prompt templates used across the experiment, including: generation prompt, review prompt (general), review prompt (specialist × 5 specialties), adversarial review prompt, merger prompt (3 strategies), debate position assignment prompt, judge prompt, and evaluator prompt. {{TO BE FINALIZED DURING PHASE 1}}

# **Appendix D: Experiment Database Schema**

The experiment database stores all inputs, outputs, and metadata for every run. Schema includes: runs (variant, task, repetition, timestamp, model\_version), designs (run\_id, round, agent\_role, full\_text, token\_count), evaluations (design\_id, evaluator\_run, dimension\_scores, overall\_score, blockers), and coherence\_checks (design\_id, section\_pair, contradicts\_flag, explanation). {{TO BE IMPLEMENTED DURING PHASE 2}}