import { useState } from "react";

const variants = {
  v1: {
    id: "v1", name: "Single Leader (Baseline)", color: "#6B7280",
    authority: "N/A", roles: "N/A", dynamics: "N/A",
    agents: "1 design agent + 1 evaluator",
    steps: [
      "Leader generates full design from prompt + rubric",
      "Evaluator scores the design against rubric",
      "Leader reads evaluator feedback and revises",
      "Repeat steps 2-3 for N rounds",
      "Final evaluator score recorded"
    ],
    researchQ: "What is the quality ceiling of a single model iterating with structured feedback? This is the cost floor and the bar every consortium variant must beat.",
    prediction: "Performs surprisingly well on simple/medium tasks. Falls short on complex tasks due to mode collapse — the model optimizes locally around its initial approach and cannot escape it.",
    costProfile: "Lowest cost. ~2N × single-generation tokens (N revision rounds).",
    riskFlag: "Anchoring bias — if the initial design has a structural flaw, self-refinement may polish the surface without fixing the foundation."
  },
  v2: {
    id: "v2", name: "Leader + Reviewers", color: "#3B82F6",
    authority: "Centralized", roles: "Homogeneous", dynamics: "Cooperative",
    agents: "1 leader + K reviewers (K=2-3 recommended) + 1 evaluator",
    steps: [
      "Leader generates full design from prompt + rubric",
      "K reviewers independently critique the design (qualitative feedback, not just scores)",
      "Leader receives all K critiques simultaneously",
      "Leader revises the design, addressing reviewer feedback",
      "Evaluator scores the revised design",
      "Optionally repeat steps 2-5 for additional rounds"
    ],
    researchQ: "Does external qualitative critique add value beyond structured self-evaluation (v1)? Does cross-model review beat same-model review? Sub-variants: v2a (self-review), v2b (cross-model), v2c (multi-model panel).",
    prediction: "Modest improvement over v1b (5-10%) on dimensions where the leader had blind spots. Cross-model review (v2b) outperforms self-review (v2a) specifically on error handling and edge cases, where different training distributions surface different failure scenarios.",
    costProfile: "Medium. ~(1 + K + 1) × single-generation per round.",
    riskFlag: "Reviewer feedback may be generic ('improve error handling') rather than actionable. Quality depends heavily on reviewer prompt engineering."
  },
  v3: {
    id: "v3", name: "Parallel Leaders + Merge", color: "#8B5CF6",
    authority: "Decentralized → Centralized", roles: "Homogeneous", dynamics: "Cooperative",
    agents: "K parallel leaders (K=3) + 1 merger agent + 1 evaluator",
    steps: [
      "K leaders independently generate full designs in parallel (no communication)",
      "Merger agent receives all K designs",
      "Merger identifies the strongest approach per dimension across all designs",
      "Merger synthesizes one coherent design, resolving contradictions explicitly",
      "Evaluator scores the merged design",
      "Optionally: merged design goes through v2-style review round"
    ],
    researchQ: "Does independent parallel generation escape mode collapse? Does architectural diversity in inputs produce a better synthesis than any single design? How critical is the merge strategy (naive vs. rubric-guided vs. dialectical)?",
    prediction: "Highest peak quality on complex tasks where the solution space is large. The merge step is the bottleneck — a bad merger destroys good inputs. Dialectical merge (v3c) outperforms naive merge (v3a) by 15-20%.",
    costProfile: "Highest cost. ~(K + 1) × single-generation minimum. Worth it only for complex tasks.",
    riskFlag: "Frankenstein risk — merged design may combine individually good ideas that are architecturally incompatible. Coherence evaluation is critical."
  },
  v4: {
    id: "v4", name: "Leader + Adversarial Reviewer", color: "#EF4444",
    authority: "Centralized", roles: "Homogeneous", dynamics: "Adversarial",
    agents: "1 leader + 1 adversarial reviewer + 1 evaluator",
    steps: [
      "Leader generates full design from prompt + rubric",
      "Adversarial reviewer attempts to reject: must cite specific rubric dimensions < threshold",
      "If rejected: leader receives rejection rationale and must revise to address cited weaknesses",
      "If accepted: design proceeds to final evaluation",
      "Repeat steps 2-4 up to N rounds (hard cap = 5)",
      "Final evaluator scores accepted design (or last revision if cap reached)"
    ],
    researchQ: "Does adversarial pressure produce more robust designs than cooperative feedback (v2)? Does the threat of rejection force the leader to address weaknesses it would otherwise rationalize away?",
    prediction: "Highest variance variant. When it works (adversary catches real flaws, leader fixes them), produces the most defensively robust designs. When it fails (adversary nitpicks or rejects endlessly), wastes tokens on an unproductive loop.",
    costProfile: "Variable. Best case ~3 rounds, worst case hits the cap. ~2N × single-generation.",
    riskFlag: "Infinite rejection loop if adversary threshold is too strict. Rubber-stamping if too lenient. Threshold tuning is the key hyperparameter."
  },
  v5: {
    id: "v5", name: "Specialist Panel", color: "#F59E0B",
    authority: "Centralized", roles: "Specialized", dynamics: "Cooperative",
    agents: "1 leader + N specialist reviewers (N=3-5) + 1 evaluator",
    steps: [
      "Leader generates full design from prompt + rubric",
      "Each specialist reviews ONLY their domain (e.g., Domain Modeling Specialist reviews entities/invariants only)",
      "Specialists: Domain Modeling, Reliability & Error Handling, Testability, Security, Operability",
      "Leader receives all specialist critiques simultaneously",
      "Leader revises the design, weighing specialist feedback per area",
      "Evaluator scores the revised design"
    ],
    researchQ: "Does role specialization in reviewers outperform general-purpose review (v2)? Do specialist reviewers catch domain-specific flaws that generalist reviewers miss? Is the per-dimension score improvement concentrated in the specialist's area?",
    prediction: "Highest per-dimension scores in specialist areas (e.g., Testability specialist lifts testability from 3→4.5). But risk of conflicting recommendations across specialists. Overall score improvement of 10-15% over v2, concentrated in weaker dimensions.",
    costProfile: "Medium-high. ~(1 + N + 1) × partial-generation (specialists write less than full reviewers).",
    riskFlag: "Specialist recommendations may conflict (security wants strict isolation, operability wants simple debugging access). Leader must arbitrate trade-offs."
  },
  v6: {
    id: "v6", name: "Rotating Leader", color: "#10B981",
    authority: "Decentralized", roles: "Specialized", dynamics: "Cooperative",
    agents: "K agents (K=3-4) rotating through phases + 1 evaluator",
    steps: [
      "Phase 1: Agent A leads (e.g., domain model + module structure). Agents B,C review.",
      "Phase 2: Agent B leads (e.g., error handling + data flow). Agent A becomes reviewer. Agent C reviews.",
      "Phase 3: Agent C leads (e.g., testing + operability). Agents A,B review.",
      "Phase 4 (optional): Final coherence pass — any agent flags cross-phase contradictions",
      "Evaluator scores the final integrated design",
      "Additional metric: coherence score (do phases contradict each other?)"
    ],
    researchQ: "Does epistemic reset (fresh eyes at each phase) prevent anchoring bias? Does phase-based specialization produce deeper treatment of each area than a single leader addressing everything?",
    prediction: "Strong individual section quality. Weakest cross-section consistency. Coherence score will be the Achilles heel — each leader 'improves' their phase while potentially violating assumptions made in earlier phases.",
    costProfile: "Medium. ~K × partial-generation + K × review per phase.",
    riskFlag: "Coherence degradation is the primary risk. Requires explicit coherence evaluation metric beyond standard rubric."
  },
  v7: {
    id: "v7", name: "Consensus Convergence", color: "#06B6D4",
    authority: "Decentralized", roles: "Homogeneous", dynamics: "Cooperative",
    agents: "K peer agents (K=3) with equal authority + 1 evaluator",
    steps: [
      "K agents independently generate full designs (same starting point as v3)",
      "Each agent reviews the other K-1 designs and writes qualitative critique",
      "Each agent revises their own design based on peer feedback received",
      "Evaluator scores all K revised designs",
      "Convergence check: if score delta < ε across agents, stop. Else repeat steps 2-4.",
      "Final output = highest-scoring design at convergence (or best at round cap)",
      "Record: pre-convergence best individual score vs. post-convergence score"
    ],
    researchQ: "Can models self-organize toward quality without centralized authority? Does peer pressure drive convergence to excellence or to mediocrity (committee effect)? Is the converged output better than the best pre-convergence individual design?",
    prediction: "Lowest variance across runs (most reliable quality). But rarely the highest peak — the committee effect pushes toward safe, consensus-driven designs. Post-convergence score will often be lower than the best pre-convergence individual, suggesting the merge approach (v3) is superior.",
    costProfile: "Highest per-round cost. ~K × (generation + K-1 reviews + revision) per round × R rounds.",
    riskFlag: "Groupthink: models may converge on shared biases rather than complementary strengths. Must compare converged output vs. best individual."
  },
  v8: {
    id: "v8", name: "Structured Debate", color: "#EC4899",
    authority: "Decentralized", roles: "Specialized", dynamics: "Adversarial",
    agents: "2 debater agents + 1 judge agent + 1 evaluator",
    steps: [
      "Assign opposing architectural positions (e.g., event-sourcing vs. CRUD, microservices vs. monolith)",
      "Debater A produces full design defending Position A",
      "Debater B produces full design defending Position B",
      "Both debaters write a rebuttal: 'why the other approach is worse for THIS specific problem'",
      "Judge reads both designs + both rebuttals",
      "Judge produces final design: may pick a winner, synthesize best of both, or find a third way",
      "Evaluator scores the judge's final output"
    ],
    researchQ: "Does forced architectural disagreement explore the solution space more effectively than independent parallel work (v3)? Do rebuttals surface trade-offs that cooperative review misses? Is adversarial diversity more valuable than random diversity?",
    prediction: "Strongest on tasks with genuine architectural trade-offs (where two approaches are legitimately viable). Weakest when one position is clearly inferior — the debater assigned the weak position wastes tokens defending it. The rebuttal step is the key value-add: it forces each debater to articulate the OTHER approach's weaknesses, surfacing trade-offs the judge can act on.",
    costProfile: "High. ~(2 designs + 2 rebuttals + 1 synthesis) = ~5 × single-generation.",
    riskFlag: "Position assignment matters enormously. If both positions are reasonable, great results. If one is clearly wrong, you're wasting half your budget on a strawman."
  }
};

const matrixCells = {
  "C-H-Co": { variants: ["v2"], label: "Centralized\nHomogeneous\nCooperative" },
  "C-H-Ad": { variants: ["v4"], label: "Centralized\nHomogeneous\nAdversarial" },
  "C-S-Co": { variants: ["v5"], label: "Centralized\nSpecialized\nCooperative" },
  "C-S-Ad": { variants: [], label: "Centralized\nSpecialized\nAdversarial" },
  "D-H-Co": { variants: ["v3", "v7"], label: "Decentralized\nHomogeneous\nCooperative" },
  "D-H-Ad": { variants: ["v8"], label: "Decentralized\nHomogeneous\nAdversarial" },
  "D-S-Co": { variants: ["v6"], label: "Decentralized\nSpecialized\nCooperative" },
  "D-S-Ad": { variants: [], label: "Decentralized\nSpecialized\nAdversarial" },
};

function VariantBadge({ id, selected, onClick }) {
  const v = variants[id];
  return (
    <button
      onClick={() => onClick(id)}
      className="px-2.5 py-1 rounded-full text-xs font-bold text-white transition-all duration-200 border-2"
      style={{
        backgroundColor: selected ? v.color : v.color + "CC",
        borderColor: selected ? "#fff" : "transparent",
        transform: selected ? "scale(1.1)" : "scale(1)",
        boxShadow: selected ? `0 0 12px ${v.color}88` : "none"
      }}
    >
      {v.id.toUpperCase()}
    </button>
  );
}

function MatrixView({ selected, onSelect }) {
  const axes = [
    { label: "Authority", options: ["Centralized", "Decentralized"], keys: ["C", "D"] },
    { label: "Dynamics", options: ["Cooperative", "Adversarial"], keys: ["Co", "Ad"] },
  ];
  const roleOptions = [
    { label: "Homogeneous", key: "H" },
    { label: "Specialized", key: "S" },
  ];

  return (
    <div className="space-y-6">
      <div className="text-center mb-2">
        <div className="inline-flex items-center gap-3 px-4 py-2 bg-gray-800 rounded-lg border border-gray-700">
          <span className="text-xs text-gray-400 uppercase tracking-wider font-semibold">v1 Baseline sits outside matrix — the control to beat</span>
          <VariantBadge id="v1" selected={selected === "v1"} onClick={onSelect} />
        </div>
      </div>

      {roleOptions.map((role) => (
        <div key={role.key} className="bg-gray-800 rounded-xl p-4 border border-gray-700">
          <div className="text-center mb-3">
            <span className="text-sm font-bold text-indigo-400 uppercase tracking-wider">
              Roles: {role.label}
            </span>
          </div>
          <table className="w-full border-collapse">
            <thead>
              <tr>
                <th className="p-2 text-xs text-gray-500 w-28"></th>
                <th className="p-2 text-xs text-emerald-400 font-semibold text-center border-b border-gray-700">🤝 Cooperative</th>
                <th className="p-2 text-xs text-red-400 font-semibold text-center border-b border-gray-700">⚔️ Adversarial</th>
              </tr>
            </thead>
            <tbody>
              {axes[0].options.map((auth, ai) => {
                const authKey = axes[0].keys[ai];
                return (
                  <tr key={auth}>
                    <td className="p-2 text-xs text-amber-400 font-semibold text-right border-r border-gray-700 align-middle">
                      {authKey === "C" ? "🏛️" : "🌐"} {auth}
                    </td>
                    {axes[1].keys.map((dynKey) => {
                      const cellKey = `${authKey}-${role.key}-${dynKey}`;
                      const cell = matrixCells[cellKey];
                      const isEmpty = cell.variants.length === 0;
                      return (
                        <td key={cellKey} className="p-3 text-center align-middle border border-gray-700" style={{ minHeight: "80px" }}>
                          {isEmpty ? (
                            <div className="flex flex-col items-center gap-1">
                              <span className="text-gray-600 text-xs italic">Empty cell</span>
                              <span className="text-gray-600 text-xs">(future work)</span>
                            </div>
                          ) : (
                            <div className="flex flex-wrap gap-2 justify-center">
                              {cell.variants.map((vid) => (
                                <div key={vid} className="flex flex-col items-center gap-1">
                                  <VariantBadge id={vid} selected={selected === vid} onClick={onSelect} />
                                  <span className="text-xs text-gray-400 max-w-24 leading-tight">{variants[vid].name}</span>
                                </div>
                              ))}
                            </div>
                          )}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ))}

      <div className="flex flex-wrap gap-3 justify-center mt-4">
        <div className="flex items-center gap-2 text-xs text-gray-400">
          <span className="inline-block w-3 h-3 rounded bg-amber-400"></span> Authority axis (rows)
        </div>
        <div className="flex items-center gap-2 text-xs text-gray-400">
          <span className="inline-block w-3 h-3 rounded bg-emerald-400"></span><span className="inline-block w-3 h-3 rounded bg-red-400"></span> Dynamics axis (columns)
        </div>
        <div className="flex items-center gap-2 text-xs text-gray-400">
          <span className="inline-block w-3 h-3 rounded bg-indigo-400"></span> Roles axis (tables)
        </div>
      </div>
    </div>
  );
}

function DetailCard({ id }) {
  const v = variants[id];
  return (
    <div className="bg-gray-800 rounded-xl border border-gray-700 overflow-hidden">
      <div className="px-5 py-3 flex items-center gap-3" style={{ backgroundColor: v.color + "22", borderBottom: `2px solid ${v.color}` }}>
        <span className="text-lg font-black text-white" style={{ color: v.color }}>{v.id.toUpperCase()}</span>
        <span className="text-base font-bold text-white">{v.name}</span>
        {v.authority !== "N/A" && (
          <div className="ml-auto flex gap-2">
            <span className="px-2 py-0.5 text-xs rounded bg-gray-900 text-amber-400 border border-gray-700">{v.authority}</span>
            <span className="px-2 py-0.5 text-xs rounded bg-gray-900 text-indigo-400 border border-gray-700">{v.roles}</span>
            <span className="px-2 py-0.5 text-xs rounded bg-gray-900 border border-gray-700" style={{ color: v.dynamics === "Adversarial" ? "#EF4444" : "#10B981" }}>{v.dynamics}</span>
          </div>
        )}
      </div>
      <div className="p-5 space-y-4">
        <div>
          <h4 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-1">Agents Involved</h4>
          <p className="text-sm text-gray-200">{v.agents}</p>
        </div>
        <div>
          <h4 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-2">Step-by-Step Sequence</h4>
          <ol className="space-y-1.5">
            {v.steps.map((step, i) => (
              <li key={i} className="flex items-start gap-2 text-sm">
                <span className="flex-shrink-0 w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold mt-0.5" style={{ backgroundColor: v.color + "33", color: v.color }}>
                  {i + 1}
                </span>
                <span className="text-gray-300">{step}</span>
              </li>
            ))}
          </ol>
        </div>
        <div className="bg-gray-900 rounded-lg p-3 border border-gray-700">
          <h4 className="text-xs font-bold text-blue-400 uppercase tracking-wider mb-1">🔬 Research Question</h4>
          <p className="text-sm text-gray-300">{v.researchQ}</p>
        </div>
        <div className="bg-gray-900 rounded-lg p-3 border border-gray-700">
          <h4 className="text-xs font-bold text-purple-400 uppercase tracking-wider mb-1">🔮 Prediction</h4>
          <p className="text-sm text-gray-300">{v.prediction}</p>
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-gray-900 rounded-lg p-3 border border-gray-700">
            <h4 className="text-xs font-bold text-green-400 uppercase tracking-wider mb-1">💰 Cost Profile</h4>
            <p className="text-sm text-gray-300">{v.costProfile}</p>
          </div>
          <div className="bg-gray-900 rounded-lg p-3 border border-gray-700">
            <h4 className="text-xs font-bold text-red-400 uppercase tracking-wider mb-1">⚠️ Key Risk</h4>
            <p className="text-sm text-gray-300">{v.riskFlag}</p>
          </div>
        </div>
      </div>
    </div>
  );
}

function ComparisonTable({ selected, onSelect }) {
  const ids = ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"];
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs border-collapse">
        <thead>
          <tr className="bg-gray-800">
            <th className="p-2 text-left text-gray-400 border border-gray-700 sticky left-0 bg-gray-800 z-10">Variant</th>
            <th className="p-2 text-left text-gray-400 border border-gray-700">Authority</th>
            <th className="p-2 text-left text-gray-400 border border-gray-700">Roles</th>
            <th className="p-2 text-left text-gray-400 border border-gray-700">Dynamics</th>
            <th className="p-2 text-left text-gray-400 border border-gray-700">Agent Count</th>
            <th className="p-2 text-left text-gray-400 border border-gray-700">Key Question vs Baseline</th>
            <th className="p-2 text-left text-gray-400 border border-gray-700">Predicted Strength</th>
            <th className="p-2 text-left text-gray-400 border border-gray-700">Predicted Weakness</th>
          </tr>
        </thead>
        <tbody>
          {ids.map((id) => {
            const v = variants[id];
            const isSelected = selected === id;
            return (
              <tr
                key={id}
                className="cursor-pointer transition-colors duration-150"
                style={{
                  backgroundColor: isSelected ? v.color + "18" : "transparent",
                  borderLeft: isSelected ? `3px solid ${v.color}` : "3px solid transparent"
                }}
                onClick={() => onSelect(id)}
              >
                <td className="p-2 border border-gray-700 sticky left-0 z-10" style={{ backgroundColor: isSelected ? v.color + "18" : "#111827" }}>
                  <div className="flex items-center gap-2">
                    <span className="font-black" style={{ color: v.color }}>{v.id.toUpperCase()}</span>
                    <span className="text-gray-300">{v.name}</span>
                  </div>
                </td>
                <td className="p-2 border border-gray-700 text-gray-400">{v.authority}</td>
                <td className="p-2 border border-gray-700 text-gray-400">{v.roles}</td>
                <td className="p-2 border border-gray-700" style={{ color: v.dynamics === "Adversarial" ? "#EF4444" : v.dynamics === "Cooperative" ? "#10B981" : "#6B7280" }}>{v.dynamics}</td>
                <td className="p-2 border border-gray-700 text-gray-400">{v.agents.split("+").length}</td>
                <td className="p-2 border border-gray-700 text-gray-300" style={{ maxWidth: "200px" }}>
                  {id === "v1" ? "—" :
                    id === "v2" ? "Does external critique beat self-eval?" :
                    id === "v3" ? "Does parallel diversity escape mode collapse?" :
                    id === "v4" ? "Does adversarial pressure force robustness?" :
                    id === "v5" ? "Does role specialization beat general review?" :
                    id === "v6" ? "Does epistemic reset prevent anchoring?" :
                    id === "v7" ? "Can agents self-organize without authority?" :
                    "Does forced disagreement explore better?"}
                </td>
                <td className="p-2 border border-gray-700 text-emerald-400" style={{ maxWidth: "150px" }}>
                  {id === "v1" ? "Cost efficiency" :
                    id === "v2" ? "Blind spot detection" :
                    id === "v3" ? "Peak quality (complex)" :
                    id === "v4" ? "Defensive robustness" :
                    id === "v5" ? "Per-dimension depth" :
                    id === "v6" ? "Section-level quality" :
                    id === "v7" ? "Consistency (low variance)" :
                    "Trade-off exploration"}
                </td>
                <td className="p-2 border border-gray-700 text-red-400" style={{ maxWidth: "150px" }}>
                  {id === "v1" ? "Mode collapse" :
                    id === "v2" ? "Generic feedback" :
                    id === "v3" ? "Coherence risk" :
                    id === "v4" ? "Rejection loops" :
                    id === "v5" ? "Conflicting advice" :
                    id === "v6" ? "Coherence degradation" :
                    id === "v7" ? "Committee mediocrity" :
                    "Strawman waste"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default function App() {
  const [selected, setSelected] = useState("v1");
  const [tab, setTab] = useState("matrix");

  return (
    <div className="min-h-screen bg-gray-900 text-white p-4 md:p-6">
      <div className="max-w-6xl mx-auto space-y-6">
        <div className="text-center space-y-1">
          <h1 className="text-2xl md:text-3xl font-black tracking-tight">LLM Consortium — Variant Taxonomy</h1>
          <p className="text-gray-400 text-sm">2 × 2 × 2 Design Space: Authority × Roles × Dynamics</p>
        </div>

        <div className="flex justify-center gap-1 bg-gray-800 rounded-lg p-1 max-w-md mx-auto">
          {[
            { key: "matrix", label: "2×2×2 Matrix" },
            { key: "table", label: "Comparison Table" },
            { key: "detail", label: "Variant Deep Dive" }
          ].map((t) => (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              className={`flex-1 px-3 py-2 rounded-md text-sm font-semibold transition-all ${
                tab === t.key ? "bg-indigo-600 text-white" : "text-gray-400 hover:text-white"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>

        {tab === "matrix" && (
          <div className="space-y-4">
            <MatrixView selected={selected} onSelect={setSelected} />
            <div className="bg-gray-800 rounded-xl border border-gray-700 p-4 mt-4">
              <h3 className="text-sm font-bold text-gray-300 mb-2">📐 Axis Definitions</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs text-gray-400">
                <div>
                  <span className="text-amber-400 font-bold">Authority:</span>
                  <p><strong className="text-gray-300">Centralized</strong> — one agent owns the final design and arbitrates conflicts.</p>
                  <p><strong className="text-gray-300">Decentralized</strong> — no single agent has final say; authority is shared, rotated, or synthesized.</p>
                </div>
                <div>
                  <span className="text-indigo-400 font-bold">Roles:</span>
                  <p><strong className="text-gray-300">Homogeneous</strong> — all agents are general-purpose with the same capabilities.</p>
                  <p><strong className="text-gray-300">Specialized</strong> — agents have distinct expertise, personas, or forced positions.</p>
                </div>
                <div>
                  <span className="text-emerald-400 font-bold">Dynamics:</span>
                  <p><strong className="text-gray-300">Cooperative</strong> — agents help, improve, and build on each other's work.</p>
                  <p><strong className="text-gray-300">Adversarial</strong> — agents challenge, oppose, reject, or argue against each other.</p>
                </div>
              </div>
            </div>
            {selected && (
              <div className="mt-4">
                <DetailCard id={selected} />
              </div>
            )}
          </div>
        )}

        {tab === "table" && (
          <div className="space-y-4">
            <p className="text-xs text-gray-500 text-center">Click any row to see full details below</p>
            <ComparisonTable selected={selected} onSelect={setSelected} />
            {selected && (
              <div className="mt-4">
                <DetailCard id={selected} />
              </div>
            )}
          </div>
        )}

        {tab === "detail" && (
          <div className="space-y-4">
            <div className="flex flex-wrap gap-2 justify-center">
              {Object.keys(variants).map((id) => (
                <VariantBadge key={id} id={id} selected={selected === id} onClick={setSelected} />
              ))}
            </div>
            <DetailCard id={selected} />
          </div>
        )}

        <div className="bg-gray-800 rounded-xl border border-gray-700 p-4 text-xs text-gray-500">
          <p className="font-bold text-gray-400 mb-1">🔬 Experimental Note</p>
          <p>Two cells are empty (Centralized+Specialized+Adversarial, Decentralized+Specialized+Adversarial). These represent future work opportunities. v3 and v7 share a cell (Decentralized+Homogeneous+Cooperative) but differ in synthesis mechanism: v3 uses a centralized merger, v7 uses peer convergence with no central authority.</p>
        </div>
      </div>
    </div>
  );
}
