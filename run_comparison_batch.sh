#!/bin/bash
# run_comparison_batch.sh - Run 1000 agents each for crimaldi and custom plumes (unilateral)
set -euo pipefail

# Configuration
EXPERIMENT_NAME="plume_comparison_$(date +%Y%m%d_%H%M%S)"
AGENTS_PER_CONDITION=1000
AGENTS_PER_JOB=10  # 10 agents per job = 100 jobs per condition

# Plume settings
PLUME_TYPES="crimaldi custom"
SENSING_MODES="unilateral"  # Changed to unilateral
PLUME_CONFIG="$(pwd)/configs/comparison_config.yaml"
PLUME_METADATA="/home/snb6/palmer_scratch/plume/smoke_1a_orig_backgroundsubtracted_meta.yaml"
OUTPUT_BASE="$(pwd)/data/raw"

# Cluster settings
PARTITION="day"
TIME_LIMIT="6:00:00"
MEM_PER_TASK="64G"
MAX_CONCURRENT=50

# First, create a config file that sets triallength=3600 for fair comparison
cat > "$PLUME_CONFIG" << EOF
# Configuration for fair plume comparison
environment: video
triallength: 3600  # Match Crimaldi duration
plotting: 0
ntrials: 1
bilateral: false  # Unilateral sensing
# Note: plume_video or plume_metadata will be set by the batch script
EOF

# Calculate total jobs
NUM_PLUMES=2  # crimaldi and custom
NUM_SENSING=1  # just unilateral
NUM_CONDITIONS=$((NUM_PLUMES * NUM_SENSING))
JOBS_PER_CONDITION=$((AGENTS_PER_CONDITION / AGENTS_PER_JOB))
TOTAL_JOBS=$((NUM_CONDITIONS * JOBS_PER_CONDITION))

echo "🔬 Plume Comparison Batch (Unilateral)"
echo "======================================"
echo "Experiment: $EXPERIMENT_NAME"
echo "Plumes: crimaldi, custom"
echo "Sensing: unilateral"
echo "Agents per plume: $AGENTS_PER_CONDITION"
echo "Total jobs: $TOTAL_JOBS (2 conditions × $JOBS_PER_CONDITION jobs/condition)"
echo "Output: $OUTPUT_BASE/$EXPERIMENT_NAME"

# Create directories
mkdir -p slurm_out slurm_err

# Export all necessary variables
export EXPERIMENT_NAME
export PLUME_TYPES
export SENSING_MODES
export AGENTS_PER_CONDITION
export AGENTS_PER_JOB
export PLUME_CONFIG
export PLUME_METADATA
export OUTPUT_BASE

# Submit the job array
JOB_ID=$(sbatch \
  --job-name="${EXPERIMENT_NAME}" \
  --partition="$PARTITION" \
  --time="$TIME_LIMIT" \
  --mem="$MEM_PER_TASK" \
  --cpus-per-task=1 \
  --array=0-$((TOTAL_JOBS-1))%"$MAX_CONCURRENT" \
  --output="slurm_out/%A_%a.out" \
  --error="slurm_err/%A_%a.err" \
  --export=ALL \
  run_batch_job_4000.sh | awk '{print $4}')

echo -e "\n✅ Submitted array job: $JOB_ID"
echo "Monitor with: squeue -j $JOB_ID"
echo "Watch output: tail -f slurm_out/${JOB_ID}_*.out"