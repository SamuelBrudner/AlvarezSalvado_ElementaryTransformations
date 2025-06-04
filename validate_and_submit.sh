#!/bin/bash
# validate_and_submit.sh - Validate simulation setup and submit with confirmation
#
# Usage: ./validate_and_submit.sh [NUM_TASKS] [START_IDX]
#        NUM_TASKS: Number of tasks to run (default: 100)
#        START_IDX: Starting task index (default: 0)
#
# Example: ./validate_and_submit.sh 50 0    # Run tasks 0-49
#          ./validate_and_submit.sh 50 50   # Run tasks 50-99

# Parse arguments
NUM_TASKS="${1:-100}"
START_IDX="${2:-0}"
END_IDX=$((START_IDX + NUM_TASKS - 1))
MAX_CONCURRENT="${3:-20}"

# Set up paths and timestamp
PROJECT_DIR=$(pwd)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
VALIDATION_DIR="validation_sessions"
VALIDATION_FILE="${VALIDATION_DIR}/validation_${TIMESTAMP}.png"
SESSION_FILE="${VALIDATION_DIR}/session_${TIMESTAMP}.txt"
PARAMS_FILE="${VALIDATION_DIR}/params_${TIMESTAMP}.json"

# Create validation directory
mkdir -p "$VALIDATION_DIR"

echo "=== Navigation Model Validation & Submission ==="
echo ""
echo "Simulation Parameters:"
echo "  Tasks: $START_IDX to $END_IDX ($NUM_TASKS total)"
echo "  Agents per task: 10"
echo "  Total agents: $((NUM_TASKS * 10))"
echo "  Max concurrent: $MAX_CONCURRENT"
echo "  Project dir: $PROJECT_DIR"
echo ""

# Step 1: Create validation figure
echo "Step 1: Creating validation figure..."

matlab -batch "
% Validation script
fprintf('\nGenerating validation figure...\n');
cd('$PROJECT_DIR');
addpath(genpath('Code'));

try
    % Create the validation figure
    validate_plume_setup_simple('$VALIDATION_FILE');
    fprintf('✓ Validation figure saved to: $VALIDATION_FILE\n');
    
    % Also save current parameters
    [plume_file, plume_config] = get_plume_file();
    
    % Extract key parameters
    params = struct();
    params.timestamp = '$TIMESTAMP';
    params.num_tasks = $NUM_TASKS;
    params.start_idx = $START_IDX;
    params.end_idx = $END_IDX;
    params.agents_per_task = 10;
    params.total_agents = $NUM_TASKS * 10;
    
    if isfield(plume_config, 'simulation')
        params.duration_seconds = plume_config.simulation.duration_seconds;
        params.success_radius_cm = plume_config.simulation.success_radius_cm;
        if isfield(plume_config.simulation, 'agent_initialization')
            params.init_x_range = plume_config.simulation.agent_initialization.x_range_cm;
            params.init_y_range = plume_config.simulation.agent_initialization.y_range_cm;
        end
    end
    
    params.plume_file = plume_file;
    params.frame_rate = plume_config.frame_rate;
    
    % Save parameters
    json_str = jsonencode(params);
    fid = fopen('$PARAMS_FILE', 'w');
    fprintf(fid, '%s', json_str);
    fclose(fid);
    
    fprintf('✓ Parameters saved to: $PARAMS_FILE\n');
    
catch ME
    fprintf('✗ Error: %s\n', ME.message);
    exit(1);
end
"

# Check if validation figure was created
if [ ! -f "$VALIDATION_FILE" ]; then
    echo ""
    echo "✗ Validation figure creation failed!"
    echo "Check for MATLAB errors above."
    exit 1
fi

# Step 2: Display validation information
echo ""
echo "=== Validation Figure Created ==="
echo "File: $VALIDATION_FILE"
echo "Size: $(ls -lh $VALIDATION_FILE | awk '{print $5}')"
echo ""

# Try to extract parameters from JSON if it exists
if [ -f "$PARAMS_FILE" ]; then
    echo "Simulation Configuration:"
    # Use python or jq if available, otherwise show raw
    if command -v python3 >/dev/null 2>&1; then
        python3 -c "
import json
with open('$PARAMS_FILE') as f:
    params = json.load(f)
    print(f'  Duration: {params.get(\"duration_seconds\", \"unknown\")} seconds')
    print(f'  Success radius: {params.get(\"success_radius_cm\", \"unknown\")} cm')
    print(f'  Frame rate: {params.get(\"frame_rate\", \"unknown\")} Hz')
    if 'init_x_range' in params:
        print(f'  Init X range: {params[\"init_x_range\"]}')
    if 'init_y_range' in params:
        print(f'  Init Y range: {params[\"init_y_range\"]}')
"
    else
        cat "$PARAMS_FILE"
    fi
    echo ""
fi

echo "To view the validation figure:"
echo "  1. Download: scp $USER@$(hostname):$(pwd)/$VALIDATION_FILE ."
echo "  2. If X11 forwarding enabled: display $VALIDATION_FILE"
echo ""
echo "Please review the figure and verify:"
echo "  ✓ Plume data visualization looks correct"
echo "  ✓ Agent initialization zone (yellow box) is properly placed"
echo "  ✓ Success zone (green circle) is at the odor source"
echo "  ✓ Scale bar confirms 10 cm reference"
echo "  ✓ Simulation duration and parameters are correct"
echo ""

# Step 3: Get user confirmation
read -p "Do you approve this configuration? Submit $NUM_TASKS jobs? (yes/no): " ANSWER

if [ "$ANSWER" != "yes" ]; then
    echo ""
    echo "Simulation cancelled by user."
    echo "Validation figure saved for reference: $VALIDATION_FILE"
    exit 0
fi

# Step 4: Submit the job
echo ""
echo "Submitting SLURM array job..."

# Check if nav_job_paths.slurm exists
if [ ! -f "nav_job_paths.slurm" ]; then
    echo "✗ Error: nav_job_paths.slurm not found!"
    echo "Run ./apply_final_fix.sh first"
    exit 1
fi

# Submit the job
JOB_OUTPUT=$(sbatch --array=${START_IDX}-${END_IDX}%${MAX_CONCURRENT} nav_job_paths.slurm 2>&1)

if [[ $JOB_OUTPUT =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
    JOB_ID="${BASH_REMATCH[1]}"
    
    echo "✓ Successfully submitted job array: $JOB_ID"
    echo ""
    
    # Step 5: Save session information
    cat > "$SESSION_FILE" << EOF
Navigation Model Validation Session
===================================
Date: $(date)
User: $USER@$(hostname)
Working Directory: $PROJECT_DIR

Job Information:
  Job ID: $JOB_ID
  Array indices: ${START_IDX}-${END_IDX}
  Number of tasks: $NUM_TASKS
  Agents per task: 10
  Total agents: $((NUM_TASKS * 10))
  Max concurrent: $MAX_CONCURRENT

Files:
  Validation figure: $VALIDATION_FILE
  Parameters JSON: $PARAMS_FILE
  SLURM script: nav_job_paths.slurm
  
Commands:
  Monitor queue: squeue -u $USER -j $JOB_ID
  Watch progress: watch -n 10 'ls -1 results/*.mat | wc -l'
  Check logs: tail -f logs/nav-${JOB_ID}_*.out
  
Approved by: $USER
Approval time: $(date +"%Y-%m-%d %H:%M:%S")
EOF

    echo "Session information saved to: $SESSION_FILE"
    echo ""
    echo "=== Job Submitted Successfully ==="
    echo ""
    echo "Monitor your job:"
    echo "  squeue -u $USER -j $JOB_ID"
    echo "  sacct -j $JOB_ID --format=JobID,State,ExitCode,Elapsed"
    echo ""
    echo "Watch results appear:"
    echo "  watch -n 10 'ls -1 results/nav_results_*.mat 2>/dev/null | wc -l'"
    echo ""
    echo "Check logs:"
    echo "  tail -f logs/nav-${JOB_ID}_${START_IDX}.out"
    echo ""
    echo "All validation files saved in: $VALIDATION_DIR/"
    
else
    echo "✗ Job submission failed!"
    echo "SLURM output: $JOB_OUTPUT"
    echo ""
    echo "Common issues:"
    echo "  - Check if you're logged into a submission node"
    echo "  - Verify your account has allocation"
    echo "  - Check SLURM partition availability: sinfo"
    exit 1
fi