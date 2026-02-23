#!/usr/bin/env python3
"""Write Opus 4.6 evaluation results for v4b designs to opus_eval.db."""
import json, sqlite3, sys, uuid, glob as glob_mod

DB_PATH = "data/opus_eval.db"
MODEL = "claude-opus-4-6"

def lookup_run_id(design_id):
    for path in glob_mod.glob("data/v4b_eval_designs/*.json"):
        try:
            with open(path) as f:
                d = json.load(f)
            if d.get("design_id") == design_id:
                return d.get("run_id", "unknown")
        except Exception:
            continue
    return "unknown"

def normalize(data):
    design_id = data["design_id"]
    run_id = data.get("run_id") or lookup_run_id(design_id)
    if "evaluation" in data:
        evaluation = data["evaluation"]
        raw_scores = evaluation["dimension_scores"]
        overall = evaluation["overall_score"]
        summary = evaluation.get("qualitative_summary", "")
    else:
        raw_scores = data["dimension_scores"]
        overall = data["overall_score"]
        summary = data.get("qualitative_summary", "")
    if isinstance(raw_scores, dict):
        dim_scores = []
        for dim_name, info in raw_scores.items():
            entry = {"dimension": dim_name, "score": info["score"]}
            if "justification" in info:
                entry["justification"] = info["justification"]
            if "blockers" in info:
                entry["blockers"] = info["blockers"]
            dim_scores.append(entry)
    else:
        dim_scores = raw_scores
    raw_cc = data.get("coherence_checks", [])
    coherence_checks = []
    for cc in raw_cc:
        if "section_pair" in cc:
            pair = cc["section_pair"]
            contradicts = cc.get("contradicts", False)
        else:
            pair = f"{cc['section_a']} <-> {cc['section_b']}"
            contradicts = not cc.get("is_coherent", True)
        coherence_checks.append({"section_pair": pair, "contradicts": contradicts, "explanation": cc.get("explanation", "")})
    return design_id, run_id, dim_scores, overall, summary, coherence_checks

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/v4b_opus_write_eval.py <json_file>")
        sys.exit(1)
    with open(sys.argv[1]) as f:
        data = json.load(f)
    design_id, run_id, dim_scores, overall, summary, coherence_checks = normalize(data)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    eval_id = uuid.uuid4().hex
    conn.execute("""INSERT OR IGNORE INTO evaluations
        (evaluation_id, design_id, evaluator_run, evaluator_model, dimension_scores,
         overall_score, qualitative_summary, input_tokens, output_tokens, cost_usd)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (eval_id, design_id, 0, MODEL, json.dumps(dim_scores), overall, summary, 0, 0, 0.0))
    for cc in coherence_checks:
        check_id = uuid.uuid4().hex
        conn.execute("""INSERT OR IGNORE INTO coherence_checks
            (check_id, design_id, section_pair, contradicts, explanation)
            VALUES (?, ?, ?, ?, ?)""",
            (check_id, design_id, cc["section_pair"], 1 if cc["contradicts"] else 0, cc["explanation"]))
    dim_medians = {d["dimension"]: d["score"] for d in dim_scores}
    all_blockers = []
    for d in dim_scores:
        all_blockers.extend(d.get("blockers", []))
    unique_blockers = list(set(all_blockers))
    conn.execute("""INSERT OR REPLACE INTO scores_median
        (design_id, run_id, dimension_medians, overall_median,
         disagreement_flags, blocker_count, blockers_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (design_id, run_id, json.dumps(dim_medians), overall, json.dumps({}), len(unique_blockers), json.dumps(unique_blockers)))
    conn.commit()
    conn.close()
    print(f"OK: {design_id} -> eval + {len(coherence_checks)} coherence + median")

if __name__ == "__main__":
    main()
