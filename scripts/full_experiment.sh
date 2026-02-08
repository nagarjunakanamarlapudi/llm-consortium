#!/bin/bash
set -euo pipefail

LOG_DIR="data/logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=== Full Experiment ==="
consortium validate --strict

# Tier 1: Simple tasks
echo "--- Tier 1: Simple Tasks ---"
consortium run experiment --tasks t1,t4 --resume 2>&1 | tee "$LOG_DIR/tier1_generation.log"
consortium run evaluate 2>&1 | tee "$LOG_DIR/tier1_evaluation.log"
consortium analyze doctor 2>&1 | tee "$LOG_DIR/tier1_doctor.log"

# Tier 2: Medium tasks
echo "--- Tier 2: Medium Tasks ---"
consortium run experiment --tasks t2,t3 --resume 2>&1 | tee "$LOG_DIR/tier2_generation.log"
consortium run evaluate 2>&1 | tee "$LOG_DIR/tier2_evaluation.log"
consortium analyze doctor 2>&1 | tee "$LOG_DIR/tier2_doctor.log"

# Tier 3: Complex tasks
echo "--- Tier 3: Complex Tasks ---"
consortium run experiment --tasks t5,t6,t7,t8 --resume 2>&1 | tee "$LOG_DIR/tier3_generation.log"
consortium run evaluate 2>&1 | tee "$LOG_DIR/tier3_evaluation.log"
consortium analyze doctor 2>&1 | tee "$LOG_DIR/tier3_doctor.log"

# Coherence
echo "--- Coherence Checks ---"
consortium run coherence --all 2>&1 | tee "$LOG_DIR/coherence.log"

# Retry failures
FAILED=$(consortium run status --failed --count 2>/dev/null || echo "0")
if [ "$FAILED" -gt 0 ]; then
    echo "--- Retrying $FAILED failed runs ---"
    consortium run experiment --resume --force 2>&1 | tee "$LOG_DIR/retry.log"
    consortium run evaluate 2>&1 | tee "$LOG_DIR/retry_eval.log"
fi

# Analysis
echo "--- Full Analysis ---"
consortium analyze all --output-dir data/exports/ 2>&1 | tee "$LOG_DIR/analysis.log"

echo "=== Experiment Complete ==="
consortium analyze doctor
