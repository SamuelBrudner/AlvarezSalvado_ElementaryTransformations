#!/bin/bash
set -euo pipefail

# Submit a batch script after expanding SBATCH variables with envsubst.
# Usage: ./run_test_batch.sh [script]
# Defaults to run_batch_job.sh if no script is given.

SCRIPT="${1:-run_batch_job.sh}"

if [[ ! -f "$SCRIPT" ]]; then
    echo "ERROR: Script $SCRIPT not found" >&2
    exit 1
fi

tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT

envsubst < "$SCRIPT" > "$tmp"

sbatch "$tmp"
