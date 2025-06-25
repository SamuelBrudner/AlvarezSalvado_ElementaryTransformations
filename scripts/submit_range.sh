#!/bin/bash
# submit_range.sh - Submit specific task ranges with smart defaults
#
# Usage: ./scripts/submit_range.sh              # Interactive mode
#        ./scripts/submit_range.sh 0 49         # Tasks 0-49
#        ./scripts/submit_range.sh 50 99        # Tasks 50-99
#        ./scripts/submit_range.sh failed       # Rerun failed tasks
#        ./scripts/submit_range.sh --verbose 0 49  # With verbose logging
#        ./scripts/submit_range.sh -v failed    # Verbose missing tasks mode

# Initialize verbose logging flag
VERBOSE=0

# Function for verbose logging
log_verbose() {
    if [[ $VERBOSE -eq 1 ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] submit_range.sh: $1"
    fi
}

# Parse verbose flag first, then shift arguments
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

PROJECT_DIR=$(pwd)
log_verbose "Project directory: $PROJECT_DIR"

# Function to find failed or missing tasks
find_missing_tasks() {
    log_verbose "Starting search for missing tasks..."
    echo "Checking for missing tasks..."
    
    # Create a list of expected vs actual
    MISSING=""
    for i in {0..99}; do
        FILE=$(printf "results/nav_results_%04d.mat" $i)
        if [ ! -f "$FILE" ]; then
            MISSING="$MISSING $i"
            log_verbose "Task $i missing (file: $FILE)"
        fi
    done
    
    if [ -z "$MISSING" ]; then
        echo "✓ All tasks completed!"
        log_verbose "All 100 tasks found to be completed"
        return 1
    else
        echo "Missing tasks:$MISSING"
        log_verbose "Found $(echo $MISSING | wc -w) missing tasks"
        return 0
    fi
}

# Parse arguments
log_verbose "Parsing command line arguments: $*"

if [ "$1" = "failed" ] || [ "$1" = "missing" ]; then
    log_verbose "Mode: Find and resubmit missing/failed tasks"
    # Find failed/missing tasks
    if find_missing_tasks; then
        # Convert missing list to ranges
        # This is simplified - just uses the full list
        echo ""
        echo "Found missing tasks:$MISSING"
        log_verbose "Missing task list: $MISSING"
        echo ""
        read -p "Submit all missing tasks? (yes/no): " CONFIRM
        log_verbose "User response to missing tasks submission: $CONFIRM"
        if [ "$CONFIRM" = "yes" ]; then
            # Create array specification
            ARRAY_SPEC=$(echo $MISSING | tr ' ' ',')
            log_verbose "SLURM array specification: $ARRAY_SPEC"
            log_verbose "Submitting job array with template: slurm/nav_job_paths.slurm"
            sbatch --array=$ARRAY_SPEC%20 slurm/nav_job_paths.slurm
        else
            log_verbose "User cancelled missing tasks submission"
        fi
    fi
    exit 0
elif [ $# -eq 2 ]; then
    # Specific range provided
    START=$1
    END=$2
    log_verbose "Mode: Specific range provided - START=$START, END=$END"
elif [ $# -eq 0 ]; then
    # Interactive mode
    log_verbose "Mode: Interactive mode - no arguments provided"
    echo "=== Submit Task Range ==="
    echo ""
    
    # Show current status
    COMPLETED=$(ls -1 results/nav_results_*.mat 2>/dev/null | wc -l)
    echo "Current status: $COMPLETED/100 tasks completed"
    log_verbose "Current completion status: $COMPLETED/100 tasks"
    
    if [ $COMPLETED -gt 0 ] && [ $COMPLETED -lt 100 ]; then
        echo ""
        echo "Quick options:"
        echo "  1) Submit remaining tasks ($COMPLETED-99)"
        echo "  2) Submit first half (0-49)"
        echo "  3) Submit second half (50-99)"
        echo "  4) Submit custom range"
        echo "  5) Find and submit missing tasks"
        echo ""
        read -p "Choose option (1-5): " OPTION
        log_verbose "User selected interactive option: $OPTION"
        
        case $OPTION in
            1)
                START=$COMPLETED
                END=99
                log_verbose "Option 1 selected: remaining tasks $START-$END"
                ;;
            2)
                START=0
                END=49
                log_verbose "Option 2 selected: first half 0-49"
                ;;
            3)
                START=50
                END=99
                log_verbose "Option 3 selected: second half 50-99"
                ;;
            4)
                read -p "Start index (0-99): " START
                read -p "End index (0-99): " END
                log_verbose "Option 4 selected: custom range $START-$END"
                ;;
            5)
                log_verbose "Option 5 selected: recursive call to find missing tasks"
                exec $0 $([ $VERBOSE -eq 1 ] && echo "--verbose") missing
                ;;
            *)
                echo "Invalid option"
                log_verbose "Invalid option selected: $OPTION"
                exit 1
                ;;
        esac
    else
        read -p "Start index (0-99): " START
        read -p "End index (0-99): " END
        log_verbose "Custom range input: $START-$END"
    fi
else
    echo "Usage: $0 [start end | failed | missing] [-v|--verbose]"
    log_verbose "Invalid usage - displaying help and exiting"
    exit 1
fi

# Validate range
log_verbose "Validating range: START=$START, END=$END"
if [ $START -lt 0 ] || [ $END -gt 99 ] || [ $START -gt $END ]; then
    echo "✗ Invalid range: $START-$END"
    echo "Must be between 0-99"
    log_verbose "Range validation failed: START=$START, END=$END"
    exit 1
fi

NUM_TASKS=$((END - START + 1))
log_verbose "Range validation passed. Number of tasks: $NUM_TASKS"

echo ""
echo "=== Submitting Task Range ==="
echo "Range: $START to $END"
echo "Tasks: $NUM_TASKS"
echo "Total agents: $((NUM_TASKS * 10))"
echo ""

# Quick validation - just check config
echo "Validating configuration..."
log_verbose "Starting MATLAB configuration validation"

matlab -batch "
addpath(genpath('Code'));
try
    [pf, pc] = get_plume_file();
    fprintf('✓ Plume file: %s\n', pf);
    if isfield(pc, 'simulation') && isfield(pc.simulation, 'duration_seconds')
        fprintf('✓ Duration: %.0f seconds\n', pc.simulation.duration_seconds);
    end
    fprintf('✓ Configuration valid\n');
catch ME
    fprintf('✗ Configuration error: %s\n', ME.message);
    exit(1);
end
" || {
    log_verbose "MATLAB configuration validation failed"
    exit 1
}

log_verbose "MATLAB configuration validation completed successfully"

echo ""
read -p "Submit $NUM_TASKS tasks ($START-$END)? (yes/no): " CONFIRM
log_verbose "Final confirmation from user: $CONFIRM"

if [ "$CONFIRM" = "yes" ]; then
    echo ""
    echo "Submitting jobs..."
    log_verbose "Proceeding with job submission using template: slurm/nav_job_paths.slurm"
    log_verbose "SLURM array specification: ${START}-${END}%20"
    
    JOB_OUTPUT=$(sbatch --array=${START}-${END}%20 slurm/nav_job_paths.slurm 2>&1)
    log_verbose "SLURM sbatch output: $JOB_OUTPUT"
    
    if [[ $JOB_OUTPUT =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
        JOB_ID="${BASH_REMATCH[1]}"
        echo "✓ Submitted job $JOB_ID"
        log_verbose "Job successfully submitted with ID: $JOB_ID"
        echo ""
        echo "Monitor with:"
        echo "  squeue -u $USER -j $JOB_ID"
        echo "  tail -f logs/nav-${JOB_ID}_${START}.out"
        
        # Save quick record
        log_verbose "Creating validation session record"
        mkdir -p validation_sessions
        RECORD_FILE="validation_sessions/quick_${JOB_ID}.txt"
        cat > "$RECORD_FILE" << EOF
Quick submission
Date: $(date)
Job ID: $JOB_ID
Range: $START-$END
Tasks: $NUM_TASKS
Verbose Mode: $([ $VERBOSE -eq 1 ] && echo "Enabled" || echo "Disabled")
EOF
        log_verbose "Session record saved to: $RECORD_FILE"
        
    else
        echo "✗ Submission failed: $JOB_OUTPUT"
        log_verbose "Job submission failed with output: $JOB_OUTPUT"
    fi
else
    echo "Cancelled."
    log_verbose "Job submission cancelled by user"
fi

log_verbose "Script execution completed"