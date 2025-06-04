#!/bin/bash
# validate_and_submit_plume.sh - HPC-optimized validation and submission for plume simulations
#
# Usage: ./validate_and_submit_plume.sh [NUM_TASKS] [START_IDX] [PLUME_TYPE]
#        NUM_TASKS: Number of tasks to run (default: 100)
#        START_IDX: Starting task index (default: 0)
#        PLUME_TYPE: 'smoke', 'crimaldi', 'both', or 'active' (default: shows menu)

# Parse arguments
NUM_TASKS="${1:-100}"
START_IDX="${2:-0}"
PLUME_ARG="${3:-}"
END_IDX=$((START_IDX + NUM_TASKS - 1))
MAX_CONCURRENT="${4:-50}"

# Get project directory from current location or environment
if [ -f "configs/paths.json" ]; then
    PROJECT_DIR=$(pwd)
else
    # Try to find project directory
    PROJECT_DIR="${MATLAB_PROJECT_ROOT:-/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations}"
fi

cd "$PROJECT_DIR" || { echo "Error: Cannot find project directory"; exit 1; }

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
VALIDATION_DIR="validation_sessions"

# Create validation directory
mkdir -p "$VALIDATION_DIR"

echo "=== HPC Navigation Model Submission ==="
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "Working directory: $PROJECT_DIR"
echo ""

# Step 1: Determine plume selection
if [ -z "$PLUME_ARG" ]; then
    # Get current plume info using here-doc
    TEMP_MATLAB=$(mktemp /tmp/get_plume_$$.m)
    cat > "$TEMP_MATLAB" << 'EOF'
% Get current plume
try
    % Change to project directory
    cd(getenv('PROJECT_DIR'));
    addpath(genpath('Code'));
    
    % Load configuration
    [~, pc] = get_plume_file();
    fprintf('%s\n', pc.plume_id);
catch ME
    fprintf('unknown\n');
end
exit;
EOF
    
    CURRENT_PLUME=$(matlab -batch "PROJECT_DIR='$PROJECT_DIR'; run('$TEMP_MATLAB')" 2>&1 | tail -1)
    rm -f "$TEMP_MATLAB"
    
    echo "Currently active plume: $CURRENT_PLUME"
    echo ""
    echo "Select simulation mode:"
    echo "  1) Crimaldi only (15 Hz, 0.74 mm/px)"
    echo "  2) Smoke only (60 Hz, 0.153 mm/px)"
    echo "  3) BOTH plumes (comparative study)"
    echo "  4) Use current active plume"
    echo ""
    read -p "Select mode (1-4): " PLUME_CHOICE
    
    case $PLUME_CHOICE in
        1) PLUME_TYPE="crimaldi" ;;
        2) PLUME_TYPE="smoke" ;;
        3) PLUME_TYPE="both" ;;
        4) PLUME_TYPE="active" ;;
        *) echo "Invalid choice"; exit 1 ;;
    esac
else
    PLUME_TYPE="$PLUME_ARG"
fi

# Step 2: Configure based on plume type
if [ "$PLUME_TYPE" = "both" ]; then
    echo ""
    echo "=== COMPARATIVE STUDY MODE ==="
    echo "Will run $NUM_TASKS tasks on EACH plume"
    echo "Total tasks: $((NUM_TASKS * 2))"
    echo ""
    echo "Task allocation:"
    echo "  Crimaldi: tasks ${START_IDX}-${END_IDX}"
    echo "  Smoke: tasks $((START_IDX + 1000))-$((END_IDX + 1000))"
    echo ""
    
    VALIDATION_FILE_CRIM="${VALIDATION_DIR}/validation_crimaldi_${TIMESTAMP}.png"
    VALIDATION_FILE_SMOKE="${VALIDATION_DIR}/validation_smoke_${TIMESTAMP}.png"
    SESSION_FILE="${VALIDATION_DIR}/session_both_${TIMESTAMP}.txt"
    PARAMS_FILE="${VALIDATION_DIR}/params_both_${TIMESTAMP}.json"
else
    PLUME_SUFFIX="_${PLUME_TYPE}"
    VALIDATION_FILE="${VALIDATION_DIR}/validation${PLUME_SUFFIX}_${TIMESTAMP}.png"
    SESSION_FILE="${VALIDATION_DIR}/session${PLUME_SUFFIX}_${TIMESTAMP}.txt"
    PARAMS_FILE="${VALIDATION_DIR}/params${PLUME_SUFFIX}_${TIMESTAMP}.json"
fi

# Step 3: Generate validation figures
echo "Generating validation figure(s)..."

if [ "$PLUME_TYPE" = "both" ]; then
    # Generate for both plumes using temp MATLAB script
    TEMP_VALIDATION=$(mktemp /tmp/validate_both_$$.m)
    cat > "$TEMP_VALIDATION" << 'EOF'
% Validation script for both plumes
project_dir = getenv('PROJECT_DIR');
cd(project_dir);
addpath(genpath('Code'));

try
    % Get paths configuration
    paths = load_paths();
    
    % Generate Crimaldi validation
    fprintf('Generating Crimaldi validation...\n');
    paths_json = jsondecode(fileread('configs/paths.json'));
    paths_json.plume_config = fullfile(project_dir, 'configs/plumes/crimaldi_10cms_bounded.json');
    
    fid = fopen('configs/paths.json', 'w');
    fprintf(fid, '%s', jsonencode(paths_json));
    fclose(fid);
    
    validate_plume_setup_simple(getenv('VALIDATION_FILE_CRIM'));
    
    % Generate Smoke validation
    fprintf('Generating Smoke validation...\n');
    paths_json.plume_config = fullfile(project_dir, 'configs/plumes/smoke_1a_backgroundsubtracted.json');
    
    fid = fopen('configs/paths.json', 'w');
    fprintf(fid, '%s', jsonencode(paths_json));
    fclose(fid);
    
    validate_plume_setup_simple(getenv('VALIDATION_FILE_SMOKE'));
    
    % Save combined parameters
    params = struct();
    params.mode = 'comparative';
    params.timestamp = getenv('TIMESTAMP');
    params.num_tasks_per_plume = str2double(getenv('NUM_TASKS'));
    params.total_tasks = params.num_tasks_per_plume * 2;
    
    fid = fopen(getenv('PARAMS_FILE'), 'w');
    fprintf(fid, '%s', jsonencode(params));
    fclose(fid);
    
    fprintf('✓ Validation figures generated\n');
catch ME
    fprintf('Error: %s\n', ME.message);
    exit(1);
end
exit(0);
EOF
    
    # Run validation
    export PROJECT_DIR VALIDATION_FILE_CRIM VALIDATION_FILE_SMOKE PARAMS_FILE TIMESTAMP NUM_TASKS
    matlab -batch "run('$TEMP_VALIDATION')" > /dev/null 2>&1
    rm -f "$TEMP_VALIDATION"
    
    echo "✓ Generated validation figures for both plumes"
else
    # Single plume validation
    TEMP_SINGLE=$(mktemp /tmp/validate_single_$$.m)
    cat > "$TEMP_SINGLE" << 'EOF'
% Single plume validation
project_dir = getenv('PROJECT_DIR');
plume_type = getenv('PLUME_TYPE');
cd(project_dir);
addpath(genpath('Code'));

try
    % Set plume if needed
    if ~strcmp(plume_type, 'active')
        paths_json = jsondecode(fileread('configs/paths.json'));
        
        if strcmp(plume_type, 'smoke')
            config_file = 'configs/plumes/smoke_1a_backgroundsubtracted.json';
        else
            config_file = 'configs/plumes/crimaldi_10cms_bounded.json';
        end
        
        paths_json.plume_config = fullfile(project_dir, config_file);
        fid = fopen('configs/paths.json', 'w');
        fprintf(fid, '%s', jsonencode(paths_json));
        fclose(fid);
    end
    
    % Generate validation
    validate_plume_setup_simple(getenv('VALIDATION_FILE'));
    
    % Save parameters
    [pf, pc] = get_plume_file();
    params = struct();
    params.mode = 'single';
    params.plume_type = plume_type;
    params.plume_id = pc.plume_id;
    params.num_tasks = str2double(getenv('NUM_TASKS'));
    params.frame_rate = pc.temporal.frame_rate;
    params.duration_seconds = pc.simulation.duration_seconds;
    
    fid = fopen(getenv('PARAMS_FILE'), 'w');
    fprintf(fid, '%s', jsonencode(params));
    fclose(fid);
    
    fprintf('✓ Validation complete\n');
catch ME
    fprintf('Error: %s\n', ME.message);
    exit(1);
end
exit(0);
EOF
    
    export PROJECT_DIR PLUME_TYPE VALIDATION_FILE PARAMS_FILE NUM_TASKS
    matlab -batch "run('$TEMP_SINGLE')" > /dev/null 2>&1
    rm -f "$TEMP_SINGLE"
    
    echo "✓ Generated validation figure"
fi

# Step 4: Display configuration summary
echo ""
echo "=== Configuration Summary ==="
if [ "$PLUME_TYPE" = "both" ]; then
    echo "Mode: COMPARATIVE STUDY"
    echo "Tasks per plume: $NUM_TASKS"
    echo "Total tasks: $((NUM_TASKS * 2))"
    echo "Total agents: $((NUM_TASKS * 20))"
    echo ""
    echo "Crimaldi tasks: ${START_IDX}-${END_IDX}"
    echo "Smoke tasks: $((START_IDX + 1000))-$((END_IDX + 1000))"
    echo ""
    echo "Validation figures:"
    echo "  Crimaldi: $VALIDATION_FILE_CRIM"
    echo "  Smoke: $VALIDATION_FILE_SMOKE"
else
    echo "Mode: SINGLE PLUME"
    echo "Plume: $PLUME_TYPE"
    echo "Tasks: ${START_IDX}-${END_IDX} ($NUM_TASKS total)"
    echo "Total agents: $((NUM_TASKS * 10))"
    echo "Validation: $VALIDATION_FILE"
fi
echo ""

# Step 5: Confirmation
read -p "Submit to HPC queue? (yes/no): " ANSWER
if [ "$ANSWER" != "yes" ]; then
    echo "Cancelled."
    exit 0
fi

# Step 6: Submit jobs
echo ""
echo "Submitting to SLURM..."

if [ "$PLUME_TYPE" = "both" ]; then
    # Submit Crimaldi jobs
    echo "Submitting Crimaldi jobs..."
    CRIM_OUTPUT=$(sbatch --array=${START_IDX}-${END_IDX}%${MAX_CONCURRENT} \
                        --job-name=nav_crim_comp \
                        nav_job_crimaldi.slurm 2>&1)
    
    if [[ $CRIM_OUTPUT =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
        CRIM_JOB_ID="${BASH_REMATCH[1]}"
        echo "✓ Crimaldi job: $CRIM_JOB_ID"
    else
        echo "✗ Crimaldi submission failed: $CRIM_OUTPUT"
        exit 1
    fi
    
    # Submit Smoke jobs with offset
    echo "Submitting Smoke jobs..."
    SMOKE_OUTPUT=$(sbatch --array=$((START_IDX + 1000))-$((END_IDX + 1000))%${MAX_CONCURRENT} \
                         --job-name=nav_smoke_comp \
                         nav_job_smoke.slurm 2>&1)
    
    if [[ $SMOKE_OUTPUT =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
        SMOKE_JOB_ID="${BASH_REMATCH[1]}"
        echo "✓ Smoke job: $SMOKE_JOB_ID"
    else
        echo "✗ Smoke submission failed: $SMOKE_OUTPUT"
        exit 1
    fi
    
    # Save session info
    cat > "$SESSION_FILE" << EOF
Comparative Navigation Model Study
==================================
Date: $(date)
User: $USER@$(hostname)
Working Directory: $PROJECT_DIR

Study Configuration:
  Mode: Comparative (both plumes)
  Tasks per plume: $NUM_TASKS
  Total tasks: $((NUM_TASKS * 2))
  Total agents: $((NUM_TASKS * 20))

Job Information:
  Crimaldi Job ID: $CRIM_JOB_ID
    Array: ${START_IDX}-${END_IDX}
    Results: results/nav_results_XXXX.mat
    
  Smoke Job ID: $SMOKE_JOB_ID
    Array: $((START_IDX + 1000))-$((END_IDX + 1000))
    Results: results/smoke_nav_results_1XXX.mat

Commands:
  Monitor both: squeue -u $USER -j ${CRIM_JOB_ID},${SMOKE_JOB_ID}
  
  Crimaldi progress: watch 'ls -1 results/nav_results_*.mat 2>/dev/null | wc -l'
  Smoke progress: watch 'ls -1 results/smoke_nav_results_1*.mat 2>/dev/null | wc -l'

Approved: $(date +"%Y-%m-%d %H:%M:%S")
EOF

    echo ""
    echo "=== Comparative Study Submitted ==="
    echo "Monitor: squeue -u $USER -j ${CRIM_JOB_ID},${SMOKE_JOB_ID}"
else
    # Single plume submission
    if [ "$PLUME_TYPE" = "smoke" ] || [[ "$PLUME_TYPE" == "active" && "$CURRENT_PLUME" == *"smoke"* ]]; then
        JOB_SCRIPT="nav_job_smoke.slurm"
        JOB_NAME="nav_smoke"
    else
        JOB_SCRIPT="nav_job_crimaldi.slurm"
        JOB_NAME="nav_crim"
    fi
    
    # Submit job
    JOB_OUTPUT=$(sbatch --array=${START_IDX}-${END_IDX}%${MAX_CONCURRENT} \
                       --job-name=$JOB_NAME \
                       $JOB_SCRIPT 2>&1)
    
    if [[ $JOB_OUTPUT =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
        JOB_ID="${BASH_REMATCH[1]}"
        echo "✓ Submitted job: $JOB_ID"
        
        # Save session
        cat > "$SESSION_FILE" << EOF
Navigation Model Session
========================
Date: $(date)
User: $USER@$(hostname)
Working Directory: $PROJECT_DIR

Configuration:
  Plume: $PLUME_TYPE
  Tasks: ${START_IDX}-${END_IDX}
  Job ID: $JOB_ID
  Script: $JOB_SCRIPT

Commands:
  Monitor: squeue -u $USER -j $JOB_ID
  Progress: watch 'ls -1 results/*nav_results_*.mat 2>/dev/null | wc -l'

Approved: $(date +"%Y-%m-%d %H:%M:%S")
EOF
        
        echo ""
        echo "Monitor with: squeue -u $USER -j $JOB_ID"
    else
        echo "✗ Submission failed: $JOB_OUTPUT"
        exit 1
    fi
fi

echo ""
echo "Session saved: $SESSION_FILE"