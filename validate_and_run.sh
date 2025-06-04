#!/bin/bash
# validate_and_run.sh - Simple validation workflow with proper here-doc

NUM_TASKS="${1:-100}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
VALIDATION_FILE="validation_${TIMESTAMP}.png"
PROJECT_DIR=$(pwd)

echo "=== Creating Validation Figure ==="

# Create temporary MATLAB script
TEMP_MATLAB=$(mktemp /tmp/validate_XXXXXX.m)

cat > "$TEMP_MATLAB" << EOF
% Validation script
fprintf('\\nCreating validation figure...\\n');

% Ensure we're in the project directory
cd('$PROJECT_DIR');
addpath(genpath('Code'));

try
    validate_plume_setup_simple('$VALIDATION_FILE');
    fprintf('\\n✓ Validation figure saved to: $VALIDATION_FILE\\n');
catch ME
    fprintf('\\n✗ Error: %s\\n', ME.message);
    fprintf('Stack trace:\\n');
    for i = 1:length(ME.stack)
        fprintf('  In %s (line %d)\\n', ME.stack(i).name, ME.stack(i).line);
    end
end
exit;
EOF

# Run MATLAB with the temporary script
matlab -nodisplay -nosplash < "$TEMP_MATLAB"

# Clean up
rm -f "$TEMP_MATLAB"

# Check if validation figure was created
if [ -f "$VALIDATION_FILE" ]; then
    echo ""
    echo "=== Validation Figure Created ==="
    echo "File: $VALIDATION_FILE"
    echo "Size: $(ls -lh $VALIDATION_FILE | awk '{print $5}')"
    echo ""
    echo "To view the figure:"
    echo "  1. Download: scp $USER@$(hostname):$(pwd)/$VALIDATION_FILE ."
    echo "  2. Or if X11 enabled: display $VALIDATION_FILE"
    echo ""
    echo "Review the figure and check:"
    echo "  - Plume data looks correct"
    echo "  - Agent initialization zone (yellow box)"
    echo "  - Success zone (green circle)"
    echo "  - Scale bar shows 10 cm"
    echo ""
    
    read -p "Do you approve? Submit $NUM_TASKS simulation jobs? (yes/no): " ANSWER
    
    if [ "$ANSWER" = "yes" ]; then
        echo ""
        echo "Submitting array job..."
        
        # Submit the job and capture the output
        JOB_OUTPUT=$(sbatch --array=0-$((NUM_TASKS-1))%50 nav_job_paths.slurm 2>&1)
        
        if [[ $JOB_OUTPUT =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
            JOB_ID="${BASH_REMATCH[1]}"
            echo "✓ Submitted job array: $JOB_ID"
            echo ""
            echo "Monitor with:"
            echo "  squeue -u $USER"
            echo "  tail -f logs/nav-${JOB_ID}_*.out"
            
            # Save session info
            cat > "validation_session_${TIMESTAMP}.txt" << EOFINFO
Validation Session
==================
Date: $(date)
Validation figure: $VALIDATION_FILE
Job ID: $JOB_ID
Array tasks: $NUM_TASKS
Total agents: $((NUM_TASKS * 10))
Approved by: $USER
EOFINFO
            
            echo ""
            echo "Session info saved to: validation_session_${TIMESTAMP}.txt"
        else
            echo "✗ Job submission failed"
            echo "Output: $JOB_OUTPUT"
        fi
    else
        echo "Simulation cancelled."
    fi
else
    echo ""
    echo "✗ Validation figure creation failed!"
    echo "Check for error messages above."
fi