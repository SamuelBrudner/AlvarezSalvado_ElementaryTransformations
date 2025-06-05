#!/bin/bash
# run_my_pipeline.sh - Combined pipeline for SLURM simulations and analysis

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

echo "Pipeline started at $(date)"
echo "Project Root: $PROJECT_ROOT"
echo "--------------------------------------------------"


# Show pipeline configuration
PIPELINE_CFG="$PROJECT_ROOT/configs/pipeline/pipeline_plumes.json"
if [ -f "$PIPELINE_CFG" ]; then
    echo "STEP 0: Plumes configured for pipeline"
    python3 - <<EOF
import json
with open('$PIPELINE_CFG') as f:
    cfg = json.load(f)
for plume in cfg.get('plumes', []):
    print(f' - {plume}')
EOF
    echo ""
else
    echo "No pipeline config found at $PIPELINE_CFG"
fi

# Prepare directories
mkdir -p "$PROJECT_ROOT/results" \
         "$PROJECT_ROOT/logs" \
         "$PROJECT_ROOT/validation_sessions" \
         "$PROJECT_ROOT/slurm_logs/nav_crim" \
         "$PROJECT_ROOT/slurm_logs/nav_smoke"

# Step 1: Generate or update configs
echo "STEP 1: Generating clean configs..."
cd "$PROJECT_ROOT"
matlab -batch "try, addpath(genpath('Code')), generate_clean_configs, catch ME, fprintf('Error: %s\n', ME.message), exit(1), end, exit(0);"
echo "Configurations generated."
echo ""

# Step 2: Submit jobs
echo "STEP 2: Submitting SLURM test jobs..."
CRIM_LOG_DIR="$PROJECT_ROOT/slurm_logs/nav_crim"
SMOKE_LOG_DIR="$PROJECT_ROOT/slurm_logs/nav_smoke"

CRIM_JOB_ID=$(sbatch --parsable \
    --output=${CRIM_LOG_DIR}/nav_crim_logs_%A_%a.out \
    --error=${CRIM_LOG_DIR}/nav_crim_logs_%A_%a.err \
    nav_job_crimaldi.slurm --array=0-0%1)
SMOKE_JOB_ID=$(sbatch --parsable \
    --output=${SMOKE_LOG_DIR}/nav_smoke_logs_%A_%a.out \
    --error=${SMOKE_LOG_DIR}/nav_smoke_logs_%A_%a.err \
    --array=1000-1000%1 nav_job_smoke.slurm)

if [[ ! "$CRIM_JOB_ID" =~ ^[0-9]+$ ]] || [[ ! "$SMOKE_JOB_ID" =~ ^[0-9]+$ ]]; then
    echo "Failed to submit jobs: CRIM_JOB_ID=$CRIM_JOB_ID SMOKE_JOB_ID=$SMOKE_JOB_ID"
    exit 1
fi

echo "Crimaldi job submitted with ID $CRIM_JOB_ID"
echo "Smoke job submitted with ID $SMOKE_JOB_ID"
echo ""

# Step 3: Wait for completion
echo "STEP 3: Waiting for SLURM jobs to finish..."
while true; do
    crim_status=$(squeue -j "$CRIM_JOB_ID" -h -o %T 2>/dev/null)
    smoke_status=$(squeue -j "$SMOKE_JOB_ID" -h -o %T 2>/dev/null)

    if [[ -z "$crim_status" && -z "$smoke_status" ]]; then
        echo "Jobs appear to have finished." 
        break
    fi
    echo "$(date): Crimaldi: ${crim_status:-done}; Smoke: ${smoke_status:-done}"
    sleep 60
done

echo ""

# Step 4: Generate summary and plots
echo "STEP 4: Generating reports and plots..."
REPORT_FILE="$PROJECT_ROOT/pipeline_results_summary_${TIMESTAMP}.txt"
./create_results_report.sh > "$REPORT_FILE"
echo "Summary written to $REPORT_FILE"

CRIM_RESULT="$PROJECT_ROOT/results/nav_results_0000.mat"
SMOKE_RESULT="$PROJECT_ROOT/results/smoke_nav_results_1000.mat"

if [ -f "$CRIM_RESULT" ]; then
    ./run_plot_results.sh "$CRIM_RESULT"
else
    echo "Warning: Missing $CRIM_RESULT"
fi

if [ -f "$SMOKE_RESULT" ]; then
    ./run_plot_results.sh "$SMOKE_RESULT"
else
    echo "Warning: Missing $SMOKE_RESULT"
fi

echo "--------------------------------------------------"
echo "Pipeline finished at $(date)"
