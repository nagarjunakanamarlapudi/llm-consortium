#!/usr/bin/env python3
"""Write Opus 4.6 evaluation results to opus_eval.db.
Usage: python3 scripts/opus_write_eval.py <json_file_path>

Accepts two JSON formats:

Format A (original):
{
  "design_id": "...",
  "run_id": "...",
  "evaluation": {
    "dimension_scores": [{"dimension": "X", "score": 4.0, ...}, ...],
    "overall_score": 4.0,
    "qualitative_summary": "..."
  },
  "coherence_checks": [
    {"section_pair": "A <-> B", "contradicts": false, "explanation": "..."},
  ]
}

Format B (agent output):
{
  "design_id": "...",
  "overall_score": 4.0,
  "dimension_scores": {"Dim Name": {"score": 4.0, "justification": "..."}, ...},
  "coherence_checks": [
    {"section_a": "A", "section_b": "B", "is_coherent": true, "explanation": "..."},
  ]
}
"""

import json
import sqlite3
import sys
import uuid
import glob as glob_mod
import os

DB_PATH = "data/opus_eval.db"
MODEL = "claude-opus-4-6"


def lookup_run_id(design_id):
    """Look up run_id from the design files."""
    for path in glob_mod.glob("data/opus_eval_designs/*.json"):
        try:
            with open(path) as f:
                d = json.load(f)
            if d.get("design_id") == design_id:
                return d.get("run_id", "unknown")
        except Exception:
            continue
    return "unknown"


def normalize(data):
    """Normalize both formats into a common structure."""
    design_id = data["design_id"]

    # Get run_id (may be missing in agent output)
    run_id = data.get("run_id") or lookup_run_id(design_id)

    # Normalize dimension_scores
    if "evaluation" in data:
        evaluation = data["evaluation"]
        raw_scores = evaluation["dimension_scores"]
        overall = evaluation["overall_score"]
        summary = evaluation.get("qualitative_summary", "")
    else:
        raw_scores = data["dimension_scores"]
        overall = data["overall_score"]
        summary = data.get("qualitative_summary", "")

    # Convert dict format to list format
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

    # Normalize coherence checks
    raw_cc = data.get("coherence_checks", [])
    coherence_checks = []
    for cc in raw_cc:
        if "section_pair" in cc:
            pair = cc["section_pair"]
            contradicts = cc.get("contradicts", False)
        else:
            pair = f"{cc['section_a']} <-> {cc['section_b']}"
            # is_coherent=true means contradicts=false
            contradicts = not cc.get("is_coherent", True)
        coherence_checks.append({
            "section_pair": pair,
            "contradicts": contradicts,
            "explanation": cc.get("explanation", ""),
        })

    return design_id, run_id, dim_scores, overall, summary, coherence_checks


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/opus_write_eval.py <json_file>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        data = json.load(f)

    design_id, run_id, dim_scores, overall, summary, coherence_checks = normalize(data)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    # Write evaluation
    eval_id = uuid.uuid4().hex
    conn.execute("""
        INSERT OR IGNORE INTO evaluations
        (evaluation_id, design_id, evaluator_run, evaluator_model, dimension_scores,
         overall_score, qualitative_summary, input_tokens, output_tokens, cost_usd)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (eval_id, design_id, 0, MODEL,
          json.dumps(dim_scores),
          overall,
          summary,
          0, 0, 0.0))

    # Write coherence checks
    for cc in coherence_checks:
        check_id = uuid.uuid4().hex
        conn.execute("""
            INSERT OR IGNORE INTO coherence_checks
            (check_id, design_id, section_pair, contradicts, explanation)
            VALUES (?, ?, ?, ?, ?)
        """, (check_id, design_id, cc["section_pair"],
              1 if cc["contradicts"] else 0,
              cc["explanation"]))

    # Compute and write median (single eval so median = score)
    dim_medians = {d["dimension"]: d["score"] for d in dim_scores}
    all_blockers = []
    for d in dim_scores:
        all_blockers.extend(d.get("blockers", []))
    unique_blockers = list(set(all_blockers))

    conn.execute("""
        INSERT OR REPLACE INTO scores_median
        (design_id, run_id, dimension_medians, overall_median,
         disagreement_flags, blocker_count, blockers_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (design_id, run_id, json.dumps(dim_medians),
          overall,
          json.dumps({}), len(unique_blockers), json.dumps(unique_blockers)))

    conn.commit()
    conn.close()
    print(f"OK: {design_id} -> eval + {len(coherence_checks)} coherence + median")


if __name__ == "__main__":
    main()
