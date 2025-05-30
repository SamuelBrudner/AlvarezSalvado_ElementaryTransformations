#!/bin/bash
# run_full_batch.sh ── launch the 4 × 1000-agent production run
# ------------------------------------------------------------------
set -euo pipefail

### ── user-tunable knobs ───────────────────────────────────────────
PLUME_CONFIG="$(pwd)/configs/my_complex_plume_config.yaml"
PLUME_VIDEO="$(pwd)/data/smoke_1a_orig_backgroundsubtracted.avi"      # absolute path!
OUTPUT_BASE="$(pwd)/data/raw"

AGENTS_PER_CONDITION=1000
AGENTS_PER_JOB=10                 # 10 × 4 conds → 400 array tasks

# cluster resources
PARTITION=day
TIME_LIMIT=6:00:00
MEM_PER_TASK=64G                  # enough for full AVI + model
MAX_CONCURRENT=100                # keep node load civil
### ─────────────────────────────────────────────────────────────────

EXP_NAME="plume_full_$(date +%Y%m%d_%H%M%S)"
TOTAL_JOBS=$(( (AGENTS_PER_CONDITION + AGENTS_PER_JOB - 1) / AGENTS_PER_JOB * 4 ))

mkdir -p slurm_out slurm_err

export EXPERIMENT_NAME="$EXP_NAME"
export PLUME_CONFIG PLUME_VIDEO
export AGENTS_PER_CONDITION AGENTS_PER_JOB
export OUTPUT_BASE
export SLURM_MEM="$MEM_PER_TASK"
export SLURM_TIME="$TIME_LIMIT"
export SLURM_ARRAY_CONCURRENT="$MAX_CONCURRENT"

echo "Submitting $TOTAL_JOBS array tasks (4 conditions × $AGENTS_PER_CONDITION agents)…"

sbatch --job-name="${EXP_NAME}_sim" \
       --partition="$PARTITION" \
       --time="$TIME_LIMIT" \
       --mem="$MEM_PER_TASK" \
       --cpus-per-task=1 \
       --array=0-$((TOTAL_JOBS-1))%"$MAX_CONCURRENT" \
       --export=ALL \
       run_batch_job_4000.sh
