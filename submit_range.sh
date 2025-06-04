#!/bin/bash
# submit_range.sh - Submit specific task ranges with smart defaults
#
# Usage: ./submit_range.sh              # Interactive mode
#        ./submit_range.sh 0 49         # Tasks 0-49
#        ./submit_range.sh 50 99        # Tasks 50-99
#        ./submit_range.sh failed       # Rerun failed tasks

PROJECT_DIR=$(pwd)

# Function to find failed or missing tasks
find_missing_tasks() {
    echo "Checking for missing tasks..."
    
    # Create a list of expected vs actual
    MISSING=""
    for i in {0..99}; do
        FILE=$(printf "results/nav_results_%04d.mat" $i)
        if [ ! -f "$FILE" ]; then
            MISSING="$MISSING $i"
        fi
    done
    
    if [ -z "$MISSING" ]; then
        echo "✓ All tasks completed!"
        return 1
    else
        echo "Missing tasks:$MISSING"
        return 0
    fi
}

# Parse arguments
if [ "$1" = "failed" ] || [ "$1" = "missing" ]; then
    # Find failed/missing tasks
    if find_missing_tasks; then
        # Convert missing list to ranges
        # This is simplified - just uses the full list
        echo ""
        echo "Found missing tasks:$MISSING"
        echo ""
        read -p "Submit all missing tasks? (yes/no): " CONFIRM
        if [ "$CONFIRM" = "yes" ]; then
            # Create array specification
            ARRAY_SPEC=$(echo $MISSING | tr ' ' ',')
            sbatch --array=$ARRAY_SPEC%20 nav_job_paths.slurm
        fi
    fi
    exit 0
elif [ $# -eq 2 ]; then
    # Specific range provided
    START=$1
    END=$2
elif [ $# -eq 0 ]; then
    # Interactive mode
    echo "=== Submit Task Range ==="
    echo ""
    
    # Show current status
    COMPLETED=$(ls -1 results/nav_results_*.mat 2>/dev/null | wc -l)
    echo "Current status: $COMPLETED/100 tasks completed"
    
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
        
        case $OPTION in
            1)
                START=$COMPLETED
                END=99
                ;;
            2)
                START=0
                END=49
                ;;
            3)
                START=50
                END=99
                ;;
            4)
                read -p "Start index (0-99): " START
                read -p "End index (0-99): " END
                ;;
            5)
                exec $0 missing
                ;;
            *)
                echo "Invalid option"
                exit 1
                ;;
        esac
    else
        read -p "Start index (0-99): " START
        read -p "End index (0-99): " END
    fi
else
    echo "Usage: $0 [start end | failed | missing]"
    exit 1
fi

# Validate range
if [ $START -lt 0 ] || [ $END -gt 99 ] || [ $START -gt $END ]; then
    echo "✗ Invalid range: $START-$END"
    echo "Must be between 0-99"
    exit 1
fi

NUM_TASKS=$((END - START + 1))

echo ""
echo "=== Submitting Task Range ==="
echo "Range: $START to $END"
echo "Tasks: $NUM_TASKS"
echo "Total agents: $((NUM_TASKS * 10))"
echo ""

# Quick validation - just check config
echo "Validating configuration..."

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
" || exit 1

echo ""
read -p "Submit $NUM_TASKS tasks ($START-$END)? (yes/no): " CONFIRM

if [ "$CONFIRM" = "yes" ]; then
    echo ""
    echo "Submitting jobs..."
    JOB_OUTPUT=$(sbatch --array=${START}-${END}%20 nav_job_paths.slurm 2>&1)
    
    if [[ $JOB_OUTPUT =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
        JOB_ID="${BASH_REMATCH[1]}"
        echo "✓ Submitted job $JOB_ID"
        echo ""
        echo "Monitor with:"
        echo "  squeue -u $USER -j $JOB_ID"
        echo "  tail -f logs/nav-${JOB_ID}_${START}.out"
        
        # Save quick record
        mkdir -p validation_sessions
        cat > "validation_sessions/quick_${JOB_ID}.txt" << EOF
Quick submission
Date: $(date)
Job ID: $JOB_ID
Range: $START-$END
Tasks: $NUM_TASKS
EOF
        
    else
        echo "✗ Submission failed: $JOB_OUTPUT"
    fi
else
    echo "Cancelled."
fi