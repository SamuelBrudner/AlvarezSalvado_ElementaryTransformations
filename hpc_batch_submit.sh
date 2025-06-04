#!/bin/bash
# hpc_batch_submit.sh - Quick HPC batch submission without interactive prompts
#
# Usage: ./hpc_batch_submit.sh PLUME NUM_AGENTS [OPTIONS]
#
# Examples:
#   ./hpc_batch_submit.sh smoke 1000              # 1000 agents on smoke
#   ./hpc_batch_submit.sh crimaldi 2000           # 2000 agents on crimaldi  
#   ./hpc_batch_submit.sh both 500                # 500 agents on EACH plume
#   ./hpc_batch_submit.sh both 1000 --partition gpu  # Use GPU partition

PLUME_TYPE="$1"
NUM_AGENTS="${2:-1000}"

# Validate inputs
if [ -z "$PLUME_TYPE" ]; then
    echo "Usage: $0 {smoke|crimaldi|both} NUM_AGENTS [SLURM_OPTIONS]"
    exit 1
fi

# Calculate tasks
TASKS_PER_PLUME=$((NUM_AGENTS / 10))
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Extract additional SLURM options
shift 2
EXTRA_SLURM_OPTS="$@"

echo "=== HPC Batch Submission ==="
echo "Plume: $PLUME_TYPE"
echo "Agents: $NUM_AGENTS (${TASKS_PER_PLUME} tasks)"
echo "Timestamp: $TIMESTAMP"
[ -n "$EXTRA_SLURM_OPTS" ] && echo "Extra options: $EXTRA_SLURM_OPTS"
echo ""

# Function to submit jobs
submit_job() {
    local plume=$1
    local start_idx=$2
    local end_idx=$3
    local job_name=$4
    
    # Select appropriate script
    if [ "$plume" = "smoke" ]; then
        SCRIPT="nav_job_smoke.slurm"
        TIME="2:00:00"
        MEM="16G"
    else
        SCRIPT="nav_job_crimaldi.slurm"
        TIME="4:00:00"
        MEM="32G"
    fi
    
    # Submit
    sbatch --array=${start_idx}-${end_idx}%100 \
           --job-name="${job_name}_${TIMESTAMP}" \
           --time=$TIME \
           --mem=$MEM \
           $EXTRA_SLURM_OPTS \
           $SCRIPT
}

# Submit based on type
case $PLUME_TYPE in
    smoke)
        echo "Submitting smoke plume jobs..."
        submit_job smoke 0 $((TASKS_PER_PLUME - 1)) "smoke"
        ;;
        
    crimaldi)
        echo "Submitting Crimaldi plume jobs..."
        submit_job crimaldi 0 $((TASKS_PER_PLUME - 1)) "crim"
        ;;
        
    both)
        echo "Submitting comparative study..."
        echo "  Crimaldi: tasks 0-$((TASKS_PER_PLUME - 1))"
        submit_job crimaldi 0 $((TASKS_PER_PLUME - 1)) "crim_comp"
        
        echo "  Smoke: tasks 1000-$((1000 + TASKS_PER_PLUME - 1))"
        submit_job smoke 1000 $((1000 + TASKS_PER_PLUME - 1)) "smoke_comp"
        
        # Create comparison info
        COMP_DIR="comparative_studies/batch_${TIMESTAMP}"
        mkdir -p "$COMP_DIR"
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
        echo ""
        echo "Comparison info saved: $COMP_DIR/info.txt"
        ;;
        
    *)
        echo "Error: Invalid plume type '$PLUME_TYPE'"
        echo "Use: smoke, crimaldi, or both"
        exit 1
        ;;
esac

echo ""
echo "Monitor with: squeue -u $USER | grep ${TIMESTAMP}"