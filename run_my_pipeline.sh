#!/bin/bash
# run_my_pipeline.sh - Combined pipeline for SLURM simulations and analysis

set -euo pipefail

# Global constants and configuration
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MAX_WAIT_TIME=7200  # Maximum wait time for jobs in seconds (2 hours)
MAX_CONCURRENT=100   # Maximum number of array tasks running concurrently

# Prepare log directories early to avoid brittle failures if they don't exist
LOG_ROOT="$PROJECT_ROOT/logs"
LOG_PIPE="$LOG_ROOT/pipeline"
mkdir -p "$LOG_PIPE"

# Define log files
MAIN_LOG="$LOG_PIPE/pipeline_main_${TIMESTAMP}.log"
ERROR_LOG="$LOG_PIPE/pipeline_errors_${TIMESTAMP}.log"

# Function to log messages with timestamps
log_message() {
    local level="$1"
    local message="$2"
    local timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$timestamp] [$level] $message" | tee -a "$MAIN_LOG"
    
    if [[ "$level" == "ERROR" ]]; then
        echo "[$timestamp] [$level] $message" >> "$ERROR_LOG"
    fi
}

# Function to check for required commands
check_requirements() {
    local missing_cmds=0
    
    for cmd in matlab sbatch squeue python3; do
        if ! command -v "$cmd" &> /dev/null; then
            log_message "ERROR" "Required command '$cmd' not found in PATH"
            missing_cmds=$((missing_cmds + 1))
        fi
    done
    
    if [[ $missing_cmds -gt 0 ]]; then
        log_message "ERROR" "Missing $missing_cmds required command(s). Pipeline cannot proceed."
        exit 1
    fi
}

# Function to validate project structure
check_project_structure() {
    local missing_dirs=0
    
    for dir in "Code" "configs"; do
        if [[ ! -d "$PROJECT_ROOT/$dir" ]]; then
            log_message "ERROR" "Required directory '$dir' not found"
            missing_dirs=$((missing_dirs + 1))
        fi
    done
    
    if [[ $missing_dirs -gt 0 ]]; then
        log_message "ERROR" "Missing $missing_dirs required directories. Pipeline cannot proceed."
        exit 1
    fi
}

# Initialize pipeline
check_requirements
log_message "INFO" "Pipeline started at $(date)"
log_message "INFO" "Project Root: $PROJECT_ROOT"
log_message "INFO" "Logs: $MAIN_LOG (main), $ERROR_LOG (errors only)"
log_message "INFO" "--------------------------------------------------"

# Check project structure
check_project_structure

# Show pipeline configuration
PIPELINE_CFG="$PROJECT_ROOT/configs/pipeline/pipeline_plumes.json"
if [ -f "$PIPELINE_CFG" ]; then
    log_message "INFO" "STEP 0: Plumes configured for pipeline"
    python3 - <<EOF
try:
    import json
    import sys
    
    try:
        with open('$PIPELINE_CFG') as f:
            cfg = json.load(f)
        for plume in cfg.get('plumes', []):
            print(f' - {plume}')
        if not cfg.get('plumes'):
            print('Warning: No plumes configured in pipeline config')
    except json.JSONDecodeError as e:
        print(f'Error: Invalid JSON in pipeline config: {e}', file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f'Error: Failed to load pipeline config: {e}', file=sys.stderr)
        sys.exit(1)
except Exception as e:
    print(f'Unexpected error: {e}', file=sys.stderr)
    sys.exit(1)
EOF
    PYTHON_STATUS=$?
    if [[ $PYTHON_STATUS -ne 0 ]]; then
        log_message "ERROR" "Failed to parse pipeline configuration"
    fi
    echo ""
else
    log_message "WARNING" "No pipeline config found at $PIPELINE_CFG"
fi

# Prepare directories with error handling
log_message "INFO" "Creating required directories"
dirs_to_create=(
    "$PROJECT_ROOT/results"
    "$PROJECT_ROOT/logs"
    "$PROJECT_ROOT/logs/pipeline"
    "$PROJECT_ROOT/logs/crimaldi"
    "$PROJECT_ROOT/logs/smoke"
    "$PROJECT_ROOT/validation_sessions"
)

for dir in "${dirs_to_create[@]}"; do
    if [[ ! -d "$dir" ]]; then
        mkdir -p "$dir" || { log_message "ERROR" "Failed to create directory: $dir"; exit 1; }
        log_message "INFO" "Created directory: $dir"
    fi
done

# Step 1: Generate or update configs
log_message "INFO" "STEP 1: Generating clean configs..."
cd "$PROJECT_ROOT" || { log_message "ERROR" "Failed to change directory to $PROJECT_ROOT"; exit 1; }

# Create temporary MATLAB script file with more robustness
TEMP_MATLAB_SCRIPT="$PROJECT_ROOT/logs/temp_generate_configs_${TIMESTAMP}.m"
cat > "$TEMP_MATLAB_SCRIPT" << 'EOF'
try
    % Add all necessary paths
    addpath(genpath('Code'));
    
    % Check working directory
    fprintf('Working directory: %s\n', pwd);
    
    % Check if expected code files exist
    if ~exist('Code/generate_clean_configs.m', 'file')
        error('Required file Code/generate_clean_configs.m not found');
    end
    
    % Load environment script if it exists
    if exist('setup_env.m', 'file')
        fprintf('Loading environment setup script...\n');
        setup_env;
    end
    
    % Run the config generation
    fprintf('Starting config generation...\n');
    generate_clean_configs;
    fprintf('Config generation completed successfully.\n');
    
    % Verify configs were created
    config_files = dir('configs/*.json');
    fprintf('Found %d config files in configs/ directory\n', numel(config_files));
    
    % List the generated config files
    fprintf('Generated configs:\n');
    for i = 1:numel(config_files)
        fprintf('  - %s\n', config_files(i).name);
    end
catch ME
    fprintf('ERROR in config generation: %s\n', getReport(ME, 'extended'));
    exit(1);
end
EOF

# Run MATLAB with timeout and capture output
log_message "INFO" "Running MATLAB to generate configs (timeout: 5 minutes)"
# Extract script name without extension for MATLAB execution
TEMP_MATLAB_SCRIPT_NAME=$(basename "$TEMP_MATLAB_SCRIPT" .m)
log_message "DEBUG" "Command: matlab -batch 'run(\'$TEMP_MATLAB_SCRIPT\')'"

# Run with timeout
timeout 300s matlab -batch "run('$TEMP_MATLAB_SCRIPT')" 2>&1 | tee -a "$MAIN_LOG"
MATLAB_EXIT=${PIPESTATUS[0]}

# Check result of MATLAB execution
if [[ $MATLAB_EXIT -eq 0 ]]; then
    log_message "INFO" "MATLAB config generation completed successfully"
else
    if [[ $MATLAB_EXIT -eq 124 ]]; then
        log_message "ERROR" "MATLAB config generation timed out after 5 minutes"
    else
        log_message "ERROR" "MATLAB config generation failed with exit code $MATLAB_EXIT"
    fi
    
    # Check if any config files were generated despite the error
    CONFIG_COUNT=$(find "$PROJECT_ROOT/configs" -name "*.json" -type f | wc -l)
    if [[ $CONFIG_COUNT -eq 0 ]]; then
        log_message "ERROR" "No config files were generated. Pipeline cannot proceed."
        exit 1
    else
        log_message "WARNING" "Found $CONFIG_COUNT config files despite MATLAB error. Proceeding with caution."
    fi
fi

# Clean up temporary script
if [[ -f "$TEMP_MATLAB_SCRIPT" ]]; then
    rm -f "$TEMP_MATLAB_SCRIPT"
fi

log_message "INFO" "Configurations generated."

# Step 2: Submit jobs with improved error handling
log_message "INFO" "STEP 2: Submitting SLURM test jobs..."
CRIM_LOG_DIR="$PROJECT_ROOT/logs/crimaldi"
SMOKE_LOG_DIR="$PROJECT_ROOT/logs/smoke"

# Check if SLURM script files exist
for script in "jobs/nav_job_template.slurm"; do
    if [[ ! -f "$PROJECT_ROOT/$script" ]]; then
        log_message "ERROR" "SLURM job script not found: $script"
        exit 1
    fi
    
    if [[ ! -x "$PROJECT_ROOT/$script" ]]; then
        log_message "WARNING" "SLURM job script not executable: $script, fixing permissions"
        chmod +x "$PROJECT_ROOT/$script" || { log_message "ERROR" "Failed to make $script executable"; exit 1; }
    fi
done

# Submit Crimaldi job with error handling
log_message "INFO" "Submitting Crimaldi job"
CRIM_JOB_ID=$(sbatch --parsable \
    --output=${CRIM_LOG_DIR}/nav_crim_%A_%a.out \
    --error=${CRIM_LOG_DIR}/nav_crim_%A_%a.err \
    --array=0-399%${MAX_CONCURRENT} \
    "$PROJECT_ROOT/jobs/nav_job_template.slurm" crimaldi 2>&1)
CRIM_STATUS=$?

if [[ $CRIM_STATUS -ne 0 ]] || [[ ! "$CRIM_JOB_ID" =~ ^[0-9]+$ ]]; then
    log_message "ERROR" "Failed to submit Crimaldi job: $CRIM_JOB_ID (status=$CRIM_STATUS)"
    exit 1
fi

# Submit Smoke job with error handling
log_message "INFO" "Submitting Smoke job"
SMOKE_JOB_ID=$(sbatch --parsable \
    --output=${SMOKE_LOG_DIR}/nav_smoke_%A_%a.out \
    --error=${SMOKE_LOG_DIR}/nav_smoke_%A_%a.err \
    --array=0-399%${MAX_CONCURRENT} \
    "$PROJECT_ROOT/jobs/nav_job_template.slurm" smoke 2>&1)
SMOKE_STATUS=$?

if [[ $SMOKE_STATUS -ne 0 ]] || [[ ! "$SMOKE_JOB_ID" =~ ^[0-9]+$ ]]; then
    log_message "ERROR" "Failed to submit Smoke job: $SMOKE_JOB_ID (status=$SMOKE_STATUS)"
    scancel $CRIM_JOB_ID 2>/dev/null  # Cancel Crimaldi job if Smoke job failed
    log_message "WARNING" "Cancelled Crimaldi job $CRIM_JOB_ID due to Smoke job submission failure"
    exit 1
fi

# Log success and store job IDs in a file for potential recovery
log_message "INFO" "Crimaldi job submitted with ID $CRIM_JOB_ID"
log_message "INFO" "Smoke job submitted with ID $SMOKE_JOB_ID"

# Write job IDs to file for potential recovery
JOB_IDS_FILE="$PROJECT_ROOT/logs/job_ids_${TIMESTAMP}.txt"
echo "crimaldi_job_id=$CRIM_JOB_ID" > "$JOB_IDS_FILE"
echo "smoke_job_id=$SMOKE_JOB_ID" >> "$JOB_IDS_FILE"
echo "timestamp=$TIMESTAMP" >> "$JOB_IDS_FILE"
log_message "DEBUG" "Job IDs stored in $JOB_IDS_FILE"

# Step 3: Wait for completion with improved monitoring
log_message "INFO" "STEP 3: Waiting for SLURM jobs to finish..."

# Initialize monitoring variables
START_WAIT_TIME=$(date +%s)
PROGRESS_INTERVAL=60  # Update progress every minute
MAX_WAIT_SECONDS=$MAX_WAIT_TIME  # 2 hours by default (from global vars)
CRIM_COMPLETE=false
SMOKE_COMPLETE=false
CRIM_EXIT_CODE="unknown"
SMOKE_EXIT_CODE="unknown"

# Function to check job completion and exit codes
check_job_completion() {
    local job_id=$1
    local job_name=$2
    local status_var=$3
    local exit_code_var=$4
    local job_status
    
    # Check if any tasks with this job ID (including array tasks) are still in the queue
    if squeue -h -u "$USER" | grep -q "^${job_id}[_ ]"; then
        return 1  # Still running
    else
        # Nothing in queue anymore, fetch final exit code

        # Try to get the exit code from sacct
        local exit_code
        exit_code=$(sacct -j "$job_id" --format=exitcode --noheader | head -1 | tr -d ' ')
        
        # Check if exit code is retrievable and is valid
        if [[ -n "$exit_code" && "$exit_code" != "0:0" ]]; then
            # Extract the first number (job exit code, not step exit code)
            exit_code=${exit_code%%:*}
            log_message "WARNING" "$job_name job completed with non-zero exit code: $exit_code"
        elif [[ -n "$exit_code" ]]; then
            log_message "INFO" "$job_name job completed successfully"
            exit_code=0
        else
            log_message "WARNING" "Could not determine exit code for $job_name job"
            exit_code="unknown"
        fi
        
        # Set the reference variables through nameref
        eval $status_var=true
        eval $exit_code_var=\"$exit_code\"
        return 0  # Job is complete
    fi
    

}

# Progress indicator variables
progress_chars=("-" "\\" "|" "/")
progress_idx=0
last_update_time=$(date +%s)

# Monitoring loop with timeout
while true; do
    current_time=$(date +%s)
    elapsed_seconds=$((current_time - START_WAIT_TIME))
    
    # Check for timeout
    if [[ $elapsed_seconds -ge $MAX_WAIT_SECONDS ]]; then
        log_message "ERROR" "Timeout waiting for jobs after ${MAX_WAIT_SECONDS} seconds"
        log_message "WARNING" "Canceling jobs that might still be running"
        scancel $CRIM_JOB_ID 2>/dev/null
        scancel $SMOKE_JOB_ID 2>/dev/null
        exit 1
    fi
    
    # Check Crimaldi job status if not already complete
    if [[ "$CRIM_COMPLETE" != "true" ]]; then
        check_job_completion "$CRIM_JOB_ID" "Crimaldi" CRIM_COMPLETE CRIM_EXIT_CODE
    fi
    
    # Check Smoke job status if not already complete
    if [[ "$SMOKE_COMPLETE" != "true" ]]; then
        check_job_completion "$SMOKE_JOB_ID" "Smoke" SMOKE_COMPLETE SMOKE_EXIT_CODE
    fi
    
    # Both jobs are complete, break the loop
    if [[ "$CRIM_COMPLETE" == "true" && "$SMOKE_COMPLETE" == "true" ]]; then
        log_message "INFO" "All jobs have finished"
        break
    fi
    
    # Update progress indicator at defined interval
    if [[ $((current_time - last_update_time)) -ge $PROGRESS_INTERVAL ]]; then
        # Get current statuses
        crim_status=$(squeue -j "$CRIM_JOB_ID" -h -o %T 2>/dev/null)
        smoke_status=$(squeue -j "$SMOKE_JOB_ID" -h -o %T 2>/dev/null)
        
        # Calculate elapsed time in a readable format
        elapsed_min=$((elapsed_seconds / 60))
        elapsed_sec=$((elapsed_seconds % 60))
        
        # Update progress indicator
        progress_char=${progress_chars[$progress_idx]}
        progress_idx=$(( (progress_idx + 1) % 4 ))
        
        log_message "INFO" "[$progress_char] Elapsed: ${elapsed_min}m ${elapsed_sec}s - Crimaldi: ${crim_status:-done}; Smoke: ${smoke_status:-done}"
        last_update_time=$current_time
    fi
    
    # Brief sleep to prevent excessive CPU usage
    sleep 5
done

# Verify job success and check for output files
if [[ "$CRIM_EXIT_CODE" != "0" || "$SMOKE_EXIT_CODE" != "0" ]]; then
    log_message "WARNING" "One or more jobs completed with non-zero exit codes:"
    log_message "WARNING" "  Crimaldi: $CRIM_EXIT_CODE"
    log_message "WARNING" "  Smoke: $SMOKE_EXIT_CODE"
    log_message "WARNING" "Checking for result files anyway before proceeding"
fi

# Step 4: Generate summary and plots with improved error handling
log_message "INFO" "STEP 4: Generating reports and plots..."

# Create a summary of result file status
log_message "INFO" "Checking for result files"
RESULT_FILES_DIR="$PROJECT_ROOT/results"

# Define expected result files
CRIM_RESULT="$RESULT_FILES_DIR/nav_results_0000.mat"
SMOKE_RESULT="$RESULT_FILES_DIR/smoke_nav_results_0000.mat"

# Check results directory content
if [[ ! -d "$RESULT_FILES_DIR" ]]; then
    log_message "ERROR" "Results directory does not exist: $RESULT_FILES_DIR"
    mkdir -p "$RESULT_FILES_DIR"
    log_message "INFO" "Created missing results directory"
fi

# Count and list all result files
RESULT_COUNT=$(find "$RESULT_FILES_DIR" -name "*nav_results_*.mat" -type f | wc -l)
log_message "INFO" "Found $RESULT_COUNT result files in $RESULT_FILES_DIR"

# Check permissions on script files
for script in "create_results_report.sh" "run_plot_results.sh"; do
    if [[ ! -f "$PROJECT_ROOT/$script" ]]; then
        log_message "ERROR" "Required script not found: $script"
        continue
    fi
    
    if [[ ! -x "$PROJECT_ROOT/$script" ]]; then
        log_message "WARNING" "Script not executable: $script, fixing permissions"
        chmod +x "$PROJECT_ROOT/$script" || log_message "ERROR" "Failed to make $script executable"
    fi
done

# Generate report with proper error handling
REPORT_FILE="$PROJECT_ROOT/logs/pipeline/pipeline_results_summary_${TIMESTAMP}.txt"
log_message "INFO" "Generating summary report to $REPORT_FILE"

cd "$PROJECT_ROOT" || { log_message "ERROR" "Failed to change directory to $PROJECT_ROOT"; exit 1; }
"$PROJECT_ROOT/create_results_report.sh" "$REPORT_FILE" 2>&1 | tee -a "$MAIN_LOG"
REPORT_STATUS=${PIPESTATUS[0]}

if [[ $REPORT_STATUS -eq 0 && -f "$REPORT_FILE" && -s "$REPORT_FILE" ]]; then
    log_message "INFO" "Summary report generated successfully: $REPORT_FILE"
else
    log_message "WARNING" "Problem generating summary report (status=$REPORT_STATUS)"
fi

# Run MATLAB success-rate aggregation
log_message "INFO" "Aggregating success rates across result files"
matlab -nodisplay -nosplash -r "addpath(genpath('Code')); summarize_success_rates; exit" 2>&1 | tee -a "$MAIN_LOG"
AGG_STATUS=${PIPESTATUS[0]}
if [[ $AGG_STATUS -ne 0 ]]; then
    log_message "WARNING" "Success-rate aggregation script failed (status=$AGG_STATUS)"
else
    log_message "INFO" "Success-rate summary generated successfully"
fi

# Generate plots for Crimaldi results
if [[ -f "$CRIM_RESULT" ]]; then
    log_message "INFO" "Generating plots for Crimaldi results"
    "$PROJECT_ROOT/run_plot_results.sh" "$CRIM_RESULT" 2>&1 | tee -a "$MAIN_LOG"
    CRIM_PLOT_STATUS=${PIPESTATUS[0]}
    
    if [[ $CRIM_PLOT_STATUS -ne 0 ]]; then
        log_message "WARNING" "Problem generating Crimaldi plots (status=$CRIM_PLOT_STATUS)"
    fi
    
    # Check if plots were actually created
    CRIM_PLOT_COUNT=$(find "$RESULT_FILES_DIR" -name "nav_results_0000_*.pdf" -type f | wc -l)
    log_message "INFO" "Generated $CRIM_PLOT_COUNT Crimaldi plots"
else
    log_message "WARNING" "Missing Crimaldi result file: $CRIM_RESULT"
fi

# Generate plots for Smoke results
if [[ -f "$SMOKE_RESULT" ]]; then
    log_message "INFO" "Generating plots for Smoke results"
    "$PROJECT_ROOT/run_plot_results.sh" "$SMOKE_RESULT" 2>&1 | tee -a "$MAIN_LOG"
    SMOKE_PLOT_STATUS=${PIPESTATUS[0]}
    
    if [[ $SMOKE_PLOT_STATUS -ne 0 ]]; then
        log_message "WARNING" "Problem generating Smoke plots (status=$SMOKE_PLOT_STATUS)"
    fi
    
    # Check if plots were actually created
    SMOKE_PLOT_COUNT=$(find "$RESULT_FILES_DIR" -name "smoke_nav_results_*_*.pdf" -type f | wc -l)
    log_message "INFO" "Generated $SMOKE_PLOT_COUNT Smoke plots"
else
    log_message "WARNING" "Missing Smoke result file: $SMOKE_RESULT"
fi

# Create a final pipeline status report
PIPELINE_STATUS="SUCCESS"
if [[ "$CRIM_EXIT_CODE" != "0" || "$SMOKE_EXIT_CODE" != "0" || ! -f "$CRIM_RESULT" || ! -f "$SMOKE_RESULT" ]]; then
    PIPELINE_STATUS="WARNING"
fi

STATUS_FILE="$PROJECT_ROOT/logs/pipeline/pipeline_status_${TIMESTAMP}.json"
cat > "$STATUS_FILE" << EOF
{
    "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "status": "$PIPELINE_STATUS",
    "jobs": {
        "crimaldi": {
            "job_id": "$CRIM_JOB_ID",
            "exit_code": "$CRIM_EXIT_CODE",
            "result_file": "$(basename "$CRIM_RESULT")",
            "result_exists": $([ -f "$CRIM_RESULT" ] && echo "true" || echo "false")
        },
        "smoke": {
            "job_id": "$SMOKE_JOB_ID",
            "exit_code": "$SMOKE_EXIT_CODE",
            "result_file": "$(basename "$SMOKE_RESULT")",
            "result_exists": $([ -f "$SMOKE_RESULT" ] && echo "true" || echo "false")
        }
    },
    "results": {
        "total_files": $RESULT_COUNT,
        "report_file": "$(basename "$REPORT_FILE")",
        "report_status": $REPORT_STATUS
    }
}
EOF

log_message "INFO" "--------------------------------------------------"
log_message "INFO" "Pipeline finished at $(date) with status: $PIPELINE_STATUS"
log_message "INFO" "Logs available at: $MAIN_LOG"
log_message "INFO" "Status report: $STATUS_FILE"

if [[ "$PIPELINE_STATUS" == "WARNING" ]]; then
    log_message "WARNING" "Pipeline completed with warnings, check logs for details"
    exit 0  # Still exit with 0 to indicate pipeline completion, despite warnings
else
    log_message "INFO" "Pipeline completed successfully"
    exit 0
fi
