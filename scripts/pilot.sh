#!/bin/bash
set -euo pipefail

echo "=== Pilot Run (v1,v2 × t1,t5 × 2 reps) ==="
consortium validate --strict
consortium db init

consortium run experiment --variants v1,v2 --tasks t1,t5 --reps 2 --resume 2>&1 | tee data/pilot_run.log

echo "=== Evaluating ==="
consortium run evaluate --force 2>&1 | tee -a data/pilot_run.log

echo "=== Coherence Checks ==="
consortium run coherence --all 2>&1 | tee -a data/pilot_run.log

echo "=== Timeline ==="
consortium timeline summary --by cost

echo "=== Data Check ==="
consortium analyze doctor

echo "=== Pilot Complete ==="
