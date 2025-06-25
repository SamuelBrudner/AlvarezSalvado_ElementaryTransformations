#!/bin/bash
# run_nav_model.sh - Navigation model execution script for running individual simulation instances
#
# Usage: ./scripts/run_nav_model.sh [OPTIONS] [MODE]
#        MODE: test|full
#              test - Run a single test job (array task 0 only)
#              full - Run full 400-task array job
#        OPTIONS:
#              -v, --verbose - Enable detailed trace output
#
# Submits slurm/nav_job_final.slurm to SLURM with appropriate parameters
# 
# Relocated to scripts/ directory as part of repository reorganization
# Enhanced with verbose logging support for improved debugging and monitoring

# Initialize verbose mode flag
VERBOSE=0

# Parse command line arguments
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        -*)
            echo "Unknown option $1"
            echo "Usage: ./scripts/run_nav_model.sh [OPTIONS] [MODE]"
            echo "  OPTIONS:"
            echo "    -v, --verbose  Enable detailed trace output"
            echo "  MODE:"
            echo "    test          Run single test job"
            echo "    full          Run full 400-task array"
            exit 1
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Restore positional parameters
set -- "${POSITIONAL_ARGS[@]}"

# Verbose logging function
log_verbose() {
    if [[ $VERBOSE -eq 1 ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [run_nav_model] $1"
    fi
}

# Initialize logging
log_verbose "Navigation model execution script started"
log_verbose "Script location: scripts/run_nav_model.sh"
log_verbose "Working directory: $(pwd)"
log_verbose "Verbose mode: enabled"

# Check if SLURM template exists
SLURM_TEMPLATE="slurm/nav_job_final.slurm"
if [[ ! -f "$SLURM_TEMPLATE" ]]; then
    echo "Error: SLURM template not found at $SLURM_TEMPLATE"
    log_verbose "SLURM template check failed: $SLURM_TEMPLATE not found"
    exit 1
fi
log_verbose "SLURM template validated: $SLURM_TEMPLATE"

# Parse execution mode and submit appropriate job
if [ "$1" == "test" ]; then
    echo "Submitting single test job..."
    log_verbose "Execution mode: test"
    log_verbose "Job type: single test job (array task 0 only)"
    log_verbose "SLURM command: sbatch --array=0-0 $SLURM_TEMPLATE"
    
    # Submit test job
    if sbatch --array=0-0 "$SLURM_TEMPLATE"; then
        log_verbose "Test job submission successful"
        echo "Test job submitted successfully"
    else
        log_verbose "Test job submission failed with exit code $?"
        echo "Error: Test job submission failed"
        exit 1
    fi
    
elif [ "$1" == "full" ]; then
    echo "Submitting full array job (400 tasks)..."
    log_verbose "Execution mode: full"
    log_verbose "Job type: full array job (400 tasks)"
    log_verbose "SLURM command: sbatch $SLURM_TEMPLATE"
    
    # Submit full array job
    if sbatch "$SLURM_TEMPLATE"; then
        log_verbose "Full array job submission successful"
        echo "Full array job submitted successfully"
    else
        log_verbose "Full array job submission failed with exit code $?"
        echo "Error: Full array job submission failed"
        exit 1
    fi
    
else
    echo "Usage: ./scripts/run_nav_model.sh [OPTIONS] [MODE]"
    echo "  OPTIONS:"
    echo "    -v, --verbose  Enable detailed trace output"
    echo "  MODE:"
    echo "    test          Run single test job"
    echo "    full          Run full 400-task array"
    
    if [[ $VERBOSE -eq 1 ]]; then
        log_verbose "Invalid or missing execution mode: '$1'"
        log_verbose "Valid modes: test, full"
        log_verbose "Script execution terminated with usage information"
    fi
    
    exit 1
fi

log_verbose "Navigation model execution script completed successfully"

# Provide verbose logging instructions
if [[ $VERBOSE -eq 1 ]]; then
    echo ""
    echo "Verbose logging enabled. To capture this output to logs directory:"
    echo "  ./scripts/run_nav_model.sh -v [MODE] > logs/nav_model_\$(date +%Y%m%d_%H%M%S).log 2>&1"
fi