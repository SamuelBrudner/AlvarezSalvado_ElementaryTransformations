#!/bin/bash
# hpc_batch_submit.sh - Quick HPC batch submission without interactive prompts
# Relocated to scripts/ directory with enhanced verbose logging support
#
# Usage: ./scripts/hpc_batch_submit.sh PLUME NUM_AGENTS [OPTIONS] [-v|--verbose]
#
# Examples:
#   ./scripts/hpc_batch_submit.sh smoke 1000              # 1000 agents on smoke
#   ./scripts/hpc_batch_submit.sh crimaldi 2000 -v       # 2000 agents on crimaldi with verbose output
#   ./scripts/hpc_batch_submit.sh both 500 --verbose     # 500 agents on EACH plume with detailed logging
#   ./scripts/hpc_batch_submit.sh both 1000 --partition gpu -v  # Use GPU partition with verbose mode
#
# Logging:
#   Use -v or --verbose flag to enable detailed trace output to stdout
#   Redirect output to logs/ directory: ./scripts/hpc_batch_submit.sh args -v > logs/batch_$(date +%Y%m%d_%H%M%S).log 2>&1

# Initialize verbose logging flag
VERBOSE=0

# Function to handle verbose logging
log_verbose() {
    [[ $VERBOSE -eq 1 ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] HPC_BATCH: $*"
}

# Parse command line arguments for verbose flag and extract other parameters
ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=1
            log_verbose "Verbose logging enabled"
            shift
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

# Restore positional parameters
set -- "${ARGS[@]}"

log_verbose "Script started with arguments: $*"
log_verbose "Current working directory: $(pwd)"
log_verbose "User: $USER, SLURM cluster: ${SLURM_CLUSTER_NAME:-not_detected}"

PLUME_TYPE="$1"
NUM_AGENTS="${2:-1000}"

log_verbose "Parsed plume type: '$PLUME_TYPE'"
log_verbose "Parsed number of agents: $NUM_AGENTS"

# Validate inputs
if [ -z "$PLUME_TYPE" ]; then
    echo "Usage: $0 {smoke|crimaldi|both} NUM_AGENTS [SLURM_OPTIONS] [-v|--verbose]"
    log_verbose "ERROR: Missing required plume type argument"
    exit 1
fi

log_verbose "Input validation passed"

# Calculate tasks
TASKS_PER_PLUME=$((NUM_AGENTS / 10))
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

log_verbose "Calculated tasks per plume: $TASKS_PER_PLUME"
log_verbose "Generated timestamp: $TIMESTAMP"

# Extract additional SLURM options (skip verbose flags that were already processed)
shift 2
EXTRA_SLURM_OPTS=""
for arg in "$@"; do
    if [[ "$arg" != "-v" && "$arg" != "--verbose" ]]; then
        EXTRA_SLURM_OPTS="$EXTRA_SLURM_OPTS $arg"
    fi
done

log_verbose "Extra SLURM options: '$EXTRA_SLURM_OPTS'"

echo "=== HPC Batch Submission ==="
echo "Plume: $PLUME_TYPE"
echo "Agents: $NUM_AGENTS (${TASKS_PER_PLUME} tasks)"
echo "Timestamp: $TIMESTAMP"
[ -n "$EXTRA_SLURM_OPTS" ] && echo "Extra options:$EXTRA_SLURM_OPTS"
[[ $VERBOSE -eq 1 ]] && echo "Verbose logging: ENABLED"
echo ""

log_verbose "=== Starting job submission workflow ==="

# Function to submit jobs
submit_job() {
    local plume=$1
    local start_idx=$2
    local end_idx=$3
    local job_name=$4
    
    log_verbose "Entering submit_job function with parameters:"
    log_verbose "  Plume: $plume"
    log_verbose "  Start index: $start_idx"
    log_verbose "  End index: $end_idx"
    log_verbose "  Job name: $job_name"
    
    # Select appropriate script (updated paths to slurm/ directory)
    if [ "$plume" = "smoke" ]; then
        SCRIPT="slurm/nav_job_smoke.slurm"
        TIME="2:00:00"
        MEM="16G"
        log_verbose "Selected smoke plume configuration: $SCRIPT, time=$TIME, mem=$MEM"
    else
        SCRIPT="slurm/nav_job_crimaldi.slurm"
        TIME="4:00:00"
        MEM="32G"
        log_verbose "Selected crimaldi plume configuration: $SCRIPT, time=$TIME, mem=$MEM"
    fi
    
    # Verify SLURM template exists
    if [ ! -f "$SCRIPT" ]; then
        echo "ERROR: SLURM template not found: $SCRIPT"
        log_verbose "CRITICAL: SLURM template missing at path: $SCRIPT"
        log_verbose "Current directory contents: $(ls -la slurm/ 2>/dev/null || echo 'slurm/ directory not found')"
        return 1
    fi
    
    log_verbose "SLURM template verified: $SCRIPT exists"
    
    # Build sbatch command
    local sbatch_cmd="sbatch --array=${start_idx}-${end_idx}%100 --job-name=${job_name}_${TIMESTAMP} --time=$TIME --mem=$MEM $EXTRA_SLURM_OPTS $SCRIPT"
    
    log_verbose "Executing sbatch command: $sbatch_cmd"
    
    # Submit job and capture output
    local sbatch_output
    sbatch_output=$(sbatch --array=${start_idx}-${end_idx}%100 \
           --job-name="${job_name}_${TIMESTAMP}" \
           --time=$TIME \
           --mem=$MEM \
           $EXTRA_SLURM_OPTS \
           $SCRIPT 2>&1)
    
    local sbatch_exit_code=$?
    
    if [ $sbatch_exit_code -eq 0 ]; then
        echo "$sbatch_output"
        log_verbose "Job submission successful: $sbatch_output"
        log_verbose "Exit code: $sbatch_exit_code"
    else
        echo "ERROR: Job submission failed with exit code $sbatch_exit_code"
        echo "$sbatch_output"
        log_verbose "Job submission FAILED with exit code: $sbatch_exit_code"
        log_verbose "Error output: $sbatch_output"
        return $sbatch_exit_code
    fi
}

log_verbose "Job submission function defined, proceeding with plume type evaluation"

# Submit based on type
case $PLUME_TYPE in
    smoke)
        echo "Submitting smoke plume jobs..."
        log_verbose "Processing smoke plume submission case"
        log_verbose "Task range: 0 to $((TASKS_PER_PLUME - 1))"
        submit_job smoke 0 $((TASKS_PER_PLUME - 1)) "smoke"
        submission_result=$?
        log_verbose "Smoke plume submission completed with exit code: $submission_result"
        ;;
        
    crimaldi)
        echo "Submitting Crimaldi plume jobs..."
        log_verbose "Processing crimaldi plume submission case"
        log_verbose "Task range: 0 to $((TASKS_PER_PLUME - 1))"
        submit_job crimaldi 0 $((TASKS_PER_PLUME - 1)) "crim"
        submission_result=$?
        log_verbose "Crimaldi plume submission completed with exit code: $submission_result"
        ;;
        
    both)
        echo "Submitting comparative study..."
        log_verbose "Processing comparative study (both plumes) submission case"
        
        echo "  Crimaldi: tasks 0-$((TASKS_PER_PLUME - 1))"
        log_verbose "Submitting crimaldi component of comparative study"
        submit_job crimaldi 0 $((TASKS_PER_PLUME - 1)) "crim_comp"
        crimaldi_result=$?
        log_verbose "Crimaldi comparative submission result: $crimaldi_result"
        
        echo "  Smoke: tasks 1000-$((1000 + TASKS_PER_PLUME - 1))"
        log_verbose "Submitting smoke component of comparative study"
        submit_job smoke 1000 $((1000 + TASKS_PER_PLUME - 1)) "smoke_comp"
        smoke_result=$?
        log_verbose "Smoke comparative submission result: $smoke_result"
        
        # Create comparison info
        COMP_DIR="comparative_studies/batch_${TIMESTAMP}"
        log_verbose "Creating comparison directory: $COMP_DIR"
        mkdir -p "$COMP_DIR"
        
        if [ -d "$COMP_DIR" ]; then
            log_verbose "Comparison directory created successfully"
        else
            log_verbose "WARNING: Failed to create comparison directory"
        fi
        
        log_verbose "Generating comparison info file"
        cat > "$COMP_DIR/info.txt" << EOF
Comparative Batch ${TIMESTAMP}
==============================
Submitted: $(date)
Agents per plume: $NUM_AGENTS
Tasks per plume: $TASKS_PER_PLUME

Crimaldi results: results/nav_results_0*
Smoke results: results/smoke_nav_results_10*

Quick comparison:
  matlab -batch "
    c = load('results/nav_results_0000.mat');
    s = load('results/smoke_nav_results_1000.mat');
    fprintf('Crimaldi: %.1f%%\n', c.out.successrate*100);
    fprintf('Smoke: %.1f%%\n', s.out.successrate*100);
  "
EOF
        
        if [ -f "$COMP_DIR/info.txt" ]; then
            log_verbose "Comparison info file created successfully: $COMP_DIR/info.txt"
            echo ""
            echo "Comparison info saved: $COMP_DIR/info.txt"
        else
            log_verbose "WARNING: Failed to create comparison info file"
        fi
        
        # Set overall result based on both submissions
        if [ $crimaldi_result -eq 0 ] && [ $smoke_result -eq 0 ]; then
            submission_result=0
            log_verbose "Both comparative submissions successful"
        else
            submission_result=1
            log_verbose "One or both comparative submissions failed (crimaldi: $crimaldi_result, smoke: $smoke_result)"
        fi
        ;;
        
    *)
        echo "Error: Invalid plume type '$PLUME_TYPE'"
        echo "Use: smoke, crimaldi, or both"
        log_verbose "ERROR: Invalid plume type specified: '$PLUME_TYPE'"
        log_verbose "Valid options are: smoke, crimaldi, both"
        exit 1
        ;;
esac

echo ""
echo "Monitor with: squeue -u $USER | grep ${TIMESTAMP}"

log_verbose "=== Job submission workflow completed ==="
log_verbose "Final submission result: ${submission_result:-0}"
log_verbose "Monitoring command provided to user"
log_verbose "Script execution finished successfully"

# Create logs directory if it doesn't exist and verbose mode is enabled
if [[ $VERBOSE -eq 1 ]] && [[ ! -d "logs" ]]; then
    mkdir -p logs
    log_verbose "Created logs directory for future verbose output redirection"
fi

log_verbose "Tip: To save this verbose output, run: $0 $* > logs/hpc_batch_$(date +%Y%m%d_%H%M%S).log 2>&1"

exit ${submission_result:-0}