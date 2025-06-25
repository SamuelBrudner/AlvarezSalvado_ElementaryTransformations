#!/bin/bash
# run_my_pipeline.sh - Combined pipeline for SLURM simulations and analysis
# 
# Usage: ./scripts/run_my_pipeline.sh [-v|--verbose]
#
# Options:
#   -v, --verbose    Enable detailed trace output and logging

set -euo pipefail

# Parse command line arguments for verbose logging
VERBOSE=0
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [-v|--verbose]"
            echo "Options:"
            echo "  -v, --verbose    Enable detailed trace output and logging"
            echo "  -h, --help       Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE=""

# Initialize logging if verbose mode is enabled
if [[ $VERBOSE -eq 1 ]]; then
    LOG_FILE="$PROJECT_ROOT/logs/run_my_pipeline_${TIMESTAMP}.log"
    mkdir -p "$PROJECT_ROOT/logs"
    echo "[$(date)] Starting verbose logging for run_my_pipeline.sh" | tee -a "$LOG_FILE"
    echo "[$(date)] Log file: $LOG_FILE" | tee -a "$LOG_FILE"
fi

# Verbose logging function
log_verbose() {
    if [[ $VERBOSE -eq 1 ]]; then
        echo "[$(date)] $1" | tee -a "$LOG_FILE"
    fi
}

# Standard output function that respects verbose mode
log_info() {
    echo "$1"
    if [[ $VERBOSE -eq 1 ]]; then
        echo "[$(date)] INFO: $1" >> "$LOG_FILE"
    fi
}

log_verbose "Pipeline script started with verbose logging enabled"
log_verbose "Project Root: $PROJECT_ROOT"
log_verbose "Timestamp: $TIMESTAMP"

log_info "Pipeline started at $(date)"
log_info "Project Root: $PROJECT_ROOT"
log_info "--------------------------------------------------"

log_verbose "Checking pipeline configuration"

# Show pipeline configuration
PIPELINE_CFG="$PROJECT_ROOT/configs/pipeline/pipeline_plumes.json"
if [ -f "$PIPELINE_CFG" ]; then
    log_info "STEP 0: Plumes configured for pipeline"
    log_verbose "Reading pipeline configuration from $PIPELINE_CFG"
    python3 - <<EOF
import json
with open('$PIPELINE_CFG') as f:
    cfg = json.load(f)
for plume in cfg.get('plumes', []):
    print(f' - {plume}')
EOF
    log_info ""
else
    log_info "No pipeline config found at $PIPELINE_CFG"
    log_verbose "Pipeline configuration file not found, continuing without plaume config"
fi

log_verbose "Creating required directories"

# Prepare directories
mkdir -p "$PROJECT_ROOT/results" \
         "$PROJECT_ROOT/logs" \
         "$PROJECT_ROOT/validation_sessions" \
         "$PROJECT_ROOT/slurm_logs/nav_crim" \
         "$PROJECT_ROOT/slurm_logs/nav_smoke"

log_verbose "Directory structure prepared successfully"

# Step 1: Generate or update configs
log_info "STEP 1: Generating clean configs..."
log_verbose "Changing to project root directory: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

log_verbose "Starting MATLAB configuration generation"
matlab -batch "try, addpath(genpath('Code')), generate_clean_configs, catch ME, fprintf('Error: %s\n', ME.message), exit(1), end, exit(0);"
log_info "Configurations generated."
log_verbose "MATLAB configuration generation completed successfully"
log_info ""

# Step 2: Submit jobs
log_info "STEP 2: Submitting SLURM test jobs..."
CRIM_LOG_DIR="$PROJECT_ROOT/slurm_logs/nav_crim"
SMOKE_LOG_DIR="$PROJECT_ROOT/slurm_logs/nav_smoke"

log_verbose "SLURM log directories configured:"
log_verbose "  Crimaldi logs: $CRIM_LOG_DIR"
log_verbose "  Smoke logs: $SMOKE_LOG_DIR"

log_verbose "Submitting Crimaldi SLURM job"
CRIM_JOB_ID=$(sbatch --parsable \
    --output=${CRIM_LOG_DIR}/nav_crim_logs_%A_%a.out \
    --error=${CRIM_LOG_DIR}/nav_crim_logs_%A_%a.err \
    slurm/nav_job_crimaldi.slurm --array=0-0%1)

log_verbose "Submitting Smoke SLURM job"    
SMOKE_JOB_ID=$(sbatch --parsable \
    --output=${SMOKE_LOG_DIR}/nav_smoke_logs_%A_%a.out \
    --error=${SMOKE_LOG_DIR}/nav_smoke_logs_%A_%a.err \
    --array=1000-1000%1 slurm/nav_job_smoke.slurm)

if [[ ! "$CRIM_JOB_ID" =~ ^[0-9]+$ ]] || [[ ! "$SMOKE_JOB_ID" =~ ^[0-9]+$ ]]; then
    log_info "Failed to submit jobs: CRIM_JOB_ID=$CRIM_JOB_ID SMOKE_JOB_ID=$SMOKE_JOB_ID"
    log_verbose "Job submission failed with invalid job IDs"
    exit 1
fi

log_info "Crimaldi job submitted with ID $CRIM_JOB_ID"
log_info "Smoke job submitted with ID $SMOKE_JOB_ID"
log_verbose "Both SLURM jobs submitted successfully"
log_info ""

# Step 3: Wait for completion
log_info "STEP 3: Waiting for SLURM jobs to finish..."
log_verbose "Starting job monitoring loop"

while true; do
    crim_status=$(squeue -j "$CRIM_JOB_ID" -h -o %T 2>/dev/null)
    smoke_status=$(squeue -j "$SMOKE_JOB_ID" -h -o %T 2>/dev/null)
    
    log_verbose "Job status check - Crimaldi: ${crim_status:-completed}, Smoke: ${smoke_status:-completed}"

    if [[ -z "$crim_status" && -z "$smoke_status" ]]; then
        log_info "Jobs appear to have finished." 
        log_verbose "Both jobs completed, exiting monitoring loop"
        break
    fi
    log_info "$(date): Crimaldi: ${crim_status:-done}; Smoke: ${smoke_status:-done}"
    log_verbose "Jobs still running, sleeping for 60 seconds"
    sleep 60
done

log_info ""

# Step 4: Generate summary and plots
log_info "STEP 4: Generating reports and plots..."
REPORT_FILE="$PROJECT_ROOT/pipeline_results_summary_${TIMESTAMP}.txt"

log_verbose "Generating results report to $REPORT_FILE"
log_verbose "Calling scripts/create_results_report.sh"
./scripts/create_results_report.sh > "$REPORT_FILE"
log_info "Summary written to $REPORT_FILE"

CRIM_RESULT="$PROJECT_ROOT/results/nav_results_0000.mat"
SMOKE_RESULT="$PROJECT_ROOT/results/smoke_nav_results_1000.mat"

log_verbose "Checking for Crimaldi results file: $CRIM_RESULT"
if [ -f "$CRIM_RESULT" ]; then
    log_verbose "Generating plots for Crimaldi results"
    ./scripts/run_plot_results.sh "$CRIM_RESULT"
else
    log_info "Warning: Missing $CRIM_RESULT"
    log_verbose "Crimaldi results file not found, skipping plot generation"
fi

log_verbose "Checking for Smoke results file: $SMOKE_RESULT"
if [ -f "$SMOKE_RESULT" ]; then
    log_verbose "Generating plots for Smoke results"
    ./scripts/run_plot_results.sh "$SMOKE_RESULT"
else
    log_info "Warning: Missing $SMOKE_RESULT"
    log_verbose "Smoke results file not found, skipping plot generation"
fi

log_info "--------------------------------------------------"
log_info "Pipeline finished at $(date)"
log_verbose "Pipeline execution completed successfully"

if [[ $VERBOSE -eq 1 ]]; then
    log_verbose "Verbose logging completed. Log file saved to: $LOG_FILE"
fi