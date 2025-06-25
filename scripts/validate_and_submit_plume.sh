#!/bin/bash
# validate_and_submit_plume.sh - HPC-optimized validation and submission for plume simulations
#
# Usage: ./validate_and_submit_plume.sh [OPTIONS] [NUM_TASKS] [START_IDX] [PLUME_TYPE]
#        OPTIONS: -v, --verbose (enable detailed trace output)
#        NUM_TASKS: Number of tasks to run (default: 100)
#        START_IDX: Starting task index (default: 0)
#        PLUME_TYPE: 'smoke', 'crimaldi', 'both', or 'active' (default: shows menu)

# Initialize verbose logging
VERBOSE=0
SCRIPT_NAME="$(basename "$0")"
LOG_DIR="logs"

# Function to create log directory
create_log_dir() {
    if [ ! -d "$LOG_DIR" ]; then
        mkdir -p "$LOG_DIR"
        [[ $VERBOSE -eq 1 ]] && echo "[$(date)] $SCRIPT_NAME: Created logs directory: $LOG_DIR"
    fi
}

# Function for verbose logging
log_verbose() {
    [[ $VERBOSE -eq 1 ]] && echo "[$(date)] $SCRIPT_NAME: $*"
}

# Parse arguments for verbose flag
ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=1
            log_verbose "Verbose logging enabled"
            shift
            ;;
        -*)
            echo "Unknown option: $1"
            echo "Usage: $0 [-v|--verbose] [NUM_TASKS] [START_IDX] [PLUME_TYPE]"
            exit 1
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

# Restore positional arguments
set -- "${ARGS[@]}"

# Parse remaining arguments
NUM_TASKS="${1:-100}"
START_IDX="${2:-0}"
PLUME_ARG="${3:-}"
END_IDX=$((START_IDX + NUM_TASKS - 1))
MAX_CONCURRENT="${4:-50}"

log_verbose "Parsed arguments - NUM_TASKS: $NUM_TASKS, START_IDX: $START_IDX, PLUME_ARG: $PLUME_ARG"

# Create log directory
create_log_dir

# Get project directory from current location or environment
if [ -f "configs/paths.json" ]; then
    PROJECT_DIR=$(pwd)
    log_verbose "Found project directory from current location: $PROJECT_DIR"
else
    # Try to find project directory
    PROJECT_DIR="${MATLAB_PROJECT_ROOT:-/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations}"
    log_verbose "Using project directory from environment: $PROJECT_DIR"
fi

cd "$PROJECT_DIR" || { 
    echo "Error: Cannot find project directory"
    log_verbose "Failed to change to project directory: $PROJECT_DIR"
    exit 1
}

log_verbose "Successfully changed to project directory: $PROJECT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
VALIDATION_DIR="validation_sessions"

log_verbose "Generated timestamp: $TIMESTAMP"
log_verbose "Validation directory: $VALIDATION_DIR"

# Create validation directory
mkdir -p "$VALIDATION_DIR"
log_verbose "Created validation directory: $VALIDATION_DIR"

echo "=== HPC Navigation Model Submission ==="
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "Working directory: $PROJECT_DIR"
[[ $VERBOSE -eq 1 ]] && echo "Verbose logging: ENABLED"
echo ""

# Step 1: Determine plume selection
log_verbose "Starting plume selection process"

if [ -z "$PLUME_ARG" ]; then
    log_verbose "No plume argument provided, starting interactive selection"
    
    # Get current plume info using here-doc
    TEMP_MATLAB=$(mktemp /tmp/get_plume_$$.m)
    log_verbose "Created temporary MATLAB script: $TEMP_MATLAB"
    
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
    
    log_verbose "Executing MATLAB script to get current plume configuration"
    CURRENT_PLUME=$(matlab -batch "PROJECT_DIR='$PROJECT_DIR'; run('$TEMP_MATLAB')" 2>&1 | tail -1)
    rm -f "$TEMP_MATLAB"
    log_verbose "Current plume detected: $CURRENT_PLUME"
    
    echo "Currently active plume: $CURRENT_PLUME"
    echo ""
    echo "Select simulation mode:"
    echo "  1) Crimaldi only (15 Hz, 0.74 mm/px)"
    echo "  2) Smoke only (60 Hz, 0.153 mm/px)"
    echo "  3) BOTH plumes (comparative study)"
    echo "  4) Use current active plume"
    echo ""
    read -p "Select mode (1-4): " PLUME_CHOICE
    
    log_verbose "User selected plume choice: $PLUME_CHOICE"
    
    case $PLUME_CHOICE in
        1) PLUME_TYPE="crimaldi" ;;
        2) PLUME_TYPE="smoke" ;;
        3) PLUME_TYPE="both" ;;
        4) PLUME_TYPE="active" ;;
        *) 
            echo "Invalid choice"
            log_verbose "Invalid plume choice: $PLUME_CHOICE"
            exit 1 
            ;;
    esac
else
    PLUME_TYPE="$PLUME_ARG"
    log_verbose "Using provided plume argument: $PLUME_TYPE"
fi

log_verbose "Final plume type selection: $PLUME_TYPE"

# Step 2: Configure based on plume type
log_verbose "Configuring validation files based on plume type"

if [ "$PLUME_TYPE" = "both" ]; then
    log_verbose "Configuring for comparative study mode"
    
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
    
    log_verbose "Crimaldi validation file: $VALIDATION_FILE_CRIM"
    log_verbose "Smoke validation file: $VALIDATION_FILE_SMOKE"
    log_verbose "Session file: $SESSION_FILE"
    log_verbose "Parameters file: $PARAMS_FILE"
else
    log_verbose "Configuring for single plume mode"
    
    PLUME_SUFFIX="_${PLUME_TYPE}"
    VALIDATION_FILE="${VALIDATION_DIR}/validation${PLUME_SUFFIX}_${TIMESTAMP}.png"
    SESSION_FILE="${VALIDATION_DIR}/session${PLUME_SUFFIX}_${TIMESTAMP}.txt"
    PARAMS_FILE="${VALIDATION_DIR}/params${PLUME_SUFFIX}_${TIMESTAMP}.json"
    
    log_verbose "Validation file: $VALIDATION_FILE"
    log_verbose "Session file: $SESSION_FILE"
    log_verbose "Parameters file: $PARAMS_FILE"
fi

# Step 3: Generate validation figures
echo "Generating validation figure(s)..."
log_verbose "Starting validation figure generation"

if [ "$PLUME_TYPE" = "both" ]; then
    log_verbose "Generating validation figures for both plumes"
    
    # Generate for both plumes using temp MATLAB script
    TEMP_VALIDATION=$(mktemp /tmp/validate_both_$$.m)
    log_verbose "Created temporary validation script: $TEMP_VALIDATION"
    
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
    log_verbose "Exporting environment variables for MATLAB validation"
    export PROJECT_DIR VALIDATION_FILE_CRIM VALIDATION_FILE_SMOKE PARAMS_FILE TIMESTAMP NUM_TASKS
    
    log_verbose "Executing MATLAB validation script for both plumes"
    matlab -batch "run('$TEMP_VALIDATION')" > /dev/null 2>&1
    MATLAB_EXIT_CODE=$?
    
    rm -f "$TEMP_VALIDATION"
    log_verbose "Removed temporary validation script"
    
    if [ $MATLAB_EXIT_CODE -eq 0 ]; then
        echo "✓ Generated validation figures for both plumes"
        log_verbose "Successfully generated validation figures for both plumes"
    else
        echo "✗ Failed to generate validation figures"
        log_verbose "MATLAB validation script failed with exit code: $MATLAB_EXIT_CODE"
        exit 1
    fi
else
    log_verbose "Generating validation figure for single plume: $PLUME_TYPE"
    
    # Single plume validation
    TEMP_SINGLE=$(mktemp /tmp/validate_single_$$.m)
    log_verbose "Created temporary single plume validation script: $TEMP_SINGLE"
    
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
    
    log_verbose "Exporting environment variables for single plume validation"
    export PROJECT_DIR PLUME_TYPE VALIDATION_FILE PARAMS_FILE NUM_TASKS
    
    log_verbose "Executing MATLAB validation script for single plume"
    matlab -batch "run('$TEMP_SINGLE')" > /dev/null 2>&1
    MATLAB_EXIT_CODE=$?
    
    rm -f "$TEMP_SINGLE"
    log_verbose "Removed temporary single plume validation script"
    
    if [ $MATLAB_EXIT_CODE -eq 0 ]; then
        echo "✓ Generated validation figure"
        log_verbose "Successfully generated validation figure for $PLUME_TYPE"
    else
        echo "✗ Failed to generate validation figure"
        log_verbose "MATLAB validation script failed with exit code: $MATLAB_EXIT_CODE"
        exit 1
    fi
fi

# Step 4: Display configuration summary
log_verbose "Displaying configuration summary"

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
    
    log_verbose "Configuration summary for comparative study displayed"
else
    echo "Mode: SINGLE PLUME"
    echo "Plume: $PLUME_TYPE"
    echo "Tasks: ${START_IDX}-${END_IDX} ($NUM_TASKS total)"
    echo "Total agents: $((NUM_TASKS * 10))"
    echo "Validation: $VALIDATION_FILE"
    
    log_verbose "Configuration summary for single plume displayed"
fi
echo ""

# Step 5: Confirmation
log_verbose "Requesting user confirmation for job submission"
read -p "Submit to HPC queue? (yes/no): " ANSWER
log_verbose "User response: $ANSWER"

if [ "$ANSWER" != "yes" ]; then
    echo "Cancelled."
    log_verbose "Job submission cancelled by user"
    exit 0
fi

# Step 6: Submit jobs
echo ""
echo "Submitting to SLURM..."
log_verbose "Starting SLURM job submission process"

if [ "$PLUME_TYPE" = "both" ]; then
    log_verbose "Submitting jobs for comparative study"
    
    # Submit Crimaldi jobs
    echo "Submitting Crimaldi jobs..."
    log_verbose "Submitting Crimaldi jobs with array ${START_IDX}-${END_IDX}"
    
    CRIM_OUTPUT=$(sbatch --array=${START_IDX}-${END_IDX}%${MAX_CONCURRENT} \
                        --job-name=nav_crim_comp \
                        slurm/nav_job_crimaldi.slurm 2>&1)
    
    log_verbose "Crimaldi sbatch output: $CRIM_OUTPUT"
    
    if [[ $CRIM_OUTPUT =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
        CRIM_JOB_ID="${BASH_REMATCH[1]}"
        echo "✓ Crimaldi job: $CRIM_JOB_ID"
        log_verbose "Successfully submitted Crimaldi job with ID: $CRIM_JOB_ID"
    else
        echo "✗ Crimaldi submission failed: $CRIM_OUTPUT"
        log_verbose "Crimaldi job submission failed: $CRIM_OUTPUT"
        exit 1
    fi
    
    # Submit Smoke jobs with offset
    echo "Submitting Smoke jobs..."
    log_verbose "Submitting Smoke jobs with array $((START_IDX + 1000))-$((END_IDX + 1000))"
    
    SMOKE_OUTPUT=$(sbatch --array=$((START_IDX + 1000))-$((END_IDX + 1000))%${MAX_CONCURRENT} \
                         --job-name=nav_smoke_comp \
                         slurm/nav_job_smoke.slurm 2>&1)
    
    log_verbose "Smoke sbatch output: $SMOKE_OUTPUT"
    
    if [[ $SMOKE_OUTPUT =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
        SMOKE_JOB_ID="${BASH_REMATCH[1]}"
        echo "✓ Smoke job: $SMOKE_JOB_ID"
        log_verbose "Successfully submitted Smoke job with ID: $SMOKE_JOB_ID"
    else
        echo "✗ Smoke submission failed: $SMOKE_OUTPUT"
        log_verbose "Smoke job submission failed: $SMOKE_OUTPUT"
        exit 1
    fi
    
    # Save session info
    log_verbose "Saving session information to: $SESSION_FILE"
    
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
    log_verbose "Comparative study submitted successfully"
else
    log_verbose "Submitting jobs for single plume: $PLUME_TYPE"
    
    # Single plume submission
    if [ "$PLUME_TYPE" = "smoke" ] || [[ "$PLUME_TYPE" == "active" && "$CURRENT_PLUME" == *"smoke"* ]]; then
        JOB_SCRIPT="slurm/nav_job_smoke.slurm"
        JOB_NAME="nav_smoke"
        log_verbose "Selected smoke job template: $JOB_SCRIPT"
    else
        JOB_SCRIPT="slurm/nav_job_crimaldi.slurm"
        JOB_NAME="nav_crim"
        log_verbose "Selected crimaldi job template: $JOB_SCRIPT"
    fi
    
    # Submit job
    log_verbose "Submitting single plume job with array ${START_IDX}-${END_IDX}"
    
    JOB_OUTPUT=$(sbatch --array=${START_IDX}-${END_IDX}%${MAX_CONCURRENT} \
                       --job-name=$JOB_NAME \
                       $JOB_SCRIPT 2>&1)
    
    log_verbose "Job sbatch output: $JOB_OUTPUT"
    
    if [[ $JOB_OUTPUT =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
        JOB_ID="${BASH_REMATCH[1]}"
        echo "✓ Submitted job: $JOB_ID"
        log_verbose "Successfully submitted job with ID: $JOB_ID"
        
        # Save session
        log_verbose "Saving session information to: $SESSION_FILE"
        
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
        log_verbose "Single plume job submitted successfully"
    else
        echo "✗ Submission failed: $JOB_OUTPUT"
        log_verbose "Job submission failed: $JOB_OUTPUT"
        exit 1
    fi
fi

echo ""
echo "Session saved: $SESSION_FILE"
log_verbose "Script execution completed successfully"

# Log completion to logs directory if verbose
if [ $VERBOSE -eq 1 ]; then
    LOG_FILE="$LOG_DIR/validate_and_submit_plume_${TIMESTAMP}.log"
    echo "[$(date)] $SCRIPT_NAME: Script execution completed successfully" >> "$LOG_FILE"
    echo "Verbose log also saved to: $LOG_FILE"
fi