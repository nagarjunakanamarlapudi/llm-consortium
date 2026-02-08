#!/bin/bash
set -euo pipefail

echo "=== Variant Smoke Tests ==="
consortium validate

for v in v1 v2 v3 v4 v5 v6 v7 v8; do
    echo "--- Testing $v ---"
    consortium run single $v t1 --rep 0 --force 2>&1 | tail -5

    # Verify traces exist
    count=$(sqlite3 data/consortium.db "SELECT COUNT(*) FROM traces WHERE run_id IN (SELECT run_id FROM runs WHERE variant_id='$v' AND task_id='t1')")
    echo "$v: $count traces recorded"

    if [ "$count" -eq 0 ]; then
        echo "ERROR: $v produced no traces!"
        exit 1
    fi
done

echo "=== Evaluating all smoke test designs ==="
consortium run evaluate --force

echo "=== Smoke Tests Complete ==="
consortium db stats
