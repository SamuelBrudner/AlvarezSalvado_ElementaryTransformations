#!/bin/bash
set -euo pipefail  # Exit on error, undefined variable, and pipe failure

#SBATCH --begin=now
#SBATCH --job-name=${EXPERIMENT_NAME}_sim
#SBATCH --mem-per-cpu=${SLURM_MEM}
#SBATCH --cpus-per-task=${SLURM_CPUS_PER_TASK}
#SBATCH --partition=${SLURM_PARTITION}
# ───────────────────────────────────────────────────────────
# Submit with:  sbatch --array=0-$((TOTAL_JOBS-1))%${SLURM_ARRAY_CONCURRENT} $0
# ⇧ the "%N" part limits to N concurrent tasks (cores) to prevent overloading
# ───────────────────────────────────────────────────────────
#SBATCH --open-mode=append
#SBATCH --output=slurm_out/%A_%a.out
#SBATCH --error=slurm_err/%A_%a.err
#SBATCH --time=${SLURM_TIME}
# Email notifications (uncomment and customize as needed)
#SBATCH --mail-user=${USER}@yale.edu
#SBATCH --mail-type=ALL

# Cleanup function to run on exit or error
cleanup() {
    local exit_code=$?
    # Clean up temporary MATLAB scripts if they exist
    [[ -n "${MATLAB_SCRIPT:-}" && -f "$MATLAB_SCRIPT" ]] && rm -f "$MATLAB_SCRIPT"
    [[ -n "${EXPORT_SCRIPT:-}" && -f "$EXPORT_SCRIPT" ]] && rm -f "$EXPORT_SCRIPT"
    exit $exit_code
}

trap cleanup EXIT SIGTERM SIGINT

# ───────────────────────────────────────────────────────────
# 1.  Directory plumbing and environment setup
# ───────────────────────────────────────────────────────────
# Create necessary directories with error checking
for dir in slurm_out slurm_err data/processed; do
    if ! mkdir -p "$dir"; then
        echo "ERROR: Failed to create directory: $dir" >&2
        exit 1
    fi
done

# Create raw data directory and check available space
RAW_DIR="data/raw"
mkdir -p "$RAW_DIR"

# Get available space in KB
get_available_space_kb() {
    if [[ "$(uname -s)" == "Linux" ]]; then
        df -k --output=avail "$1" | tail -n 1
    else  # For macOS
        df -k "$1" | tail -n 1 | awk '{print $4}'
    fi
}

# Check disk space
AVAILABLE_SPACE_KB=$(get_available_space_kb "$OUTPUT_BASE")

if [[ $AVAILABLE_SPACE_KB -lt $REQUIRED_SPACE_KB ]]; then
    echo "ERROR: Not enough disk space in $OUTPUT_BASE" >&2
    echo "  Required: $((REQUIRED_SPACE_KB / 1024)) MB" >&2
    echo "  Available: $((AVAILABLE_SPACE_KB / 1024)) MB" >&2
    echo "  Please free up space or change the output directory" >&2
    exit 1
fi

echo "Disk space check passed: $((AVAILABLE_SPACE_KB / 1024)) MB available in $OUTPUT_BASE"

# Disable GUI / X11 during batch jobs
export DISPLAY=
unset X11

# Check if MATLAB module exists
if ! module is-avail "$MATLAB_MODULE" 2>/dev/null; then
    echo "ERROR: MATLAB module '$MATLAB_MODULE' is not available" >&2
    echo "Available MATLAB versions:"
    module -t avail 2>&1 | grep -i matlab || echo "  (none found)"
    exit 1
fi

# Load MATLAB module
if ! module load "$MATLAB_MODULE"; then
    echo "ERROR: Failed to load MATLAB module: $MATLAB_MODULE" >&2
    exit 1
fi

# ───────────────────────────────────────────────────────────
# 2.  Configuration - All parameters can be overridden with environment variables
# ───────────────────────────────────────────────────────────

# Experiment configuration
: ${EXPERIMENT_NAME:="default_experiment"}  # Name for this experiment
: ${PLUME_TYPES:="crimaldi custom"}         # Space-separated list of plume types
: ${SENSING_MODES:="bilateral unilateral"}   # Space-separated list of sensing modes
: ${AGENTS_PER_CONDITION:=1000}              # Number of agents per condition
: ${AGENTS_PER_JOB:=100}                     # Agents to simulate per SLURM task
: ${PLUME_CONFIG:="configs/my_complex_plume_config.yaml"}  # Path to config file
: ${OUTPUT_BASE:="data/raw"}                 # Base directory for output files
: ${MATLAB_VERSION:="R2021a"}                # MATLAB version to use
: ${MATLAB_MODULE:="matlab/${MATLAB_VERSION}"} # MATLAB module to load
: ${SLURM_PARTITION:="day"}                  # SLURM partition to use
: ${SLURM_TIME:="6:00:00"}                   # Maximum runtime per job
: ${SLURM_MEM:="16G"}                        # Memory per CPU
: ${SLURM_CPUS_PER_TASK:=1}                  # CPUs per task
: ${SLURM_ARRAY_CONCURRENT:=100}             # Maximum concurrent array jobs
: ${MATLAB_OPTIONS:="-nodisplay -nosplash"}  # Additional MATLAB options

# Derived parameters - don't modify these
IFS=' ' read -r -a PLUMES <<< "$PLUME_TYPES"
IFS=' ' read -r -a SENSING <<< "$SENSING_MODES"
NUM_PLUMES=${#PLUMES[@]}
NUM_SENSING=${#SENSING[@]}
NUM_CONDITIONS=$((NUM_PLUMES * NUM_SENSING))
JOBS_PER_CONDITION=$(( (AGENTS_PER_CONDITION + AGENTS_PER_JOB - 1) / AGENTS_PER_JOB ))
TOTAL_JOBS=$((NUM_CONDITIONS * JOBS_PER_CONDITION))

# Calculate required disk space (400KB per agent with 20% buffer)
BYTES_PER_AGENT=400000  # 400KB per agent
REQUIRED_SPACE_KB=$(( (AGENTS_PER_CONDITION * NUM_CONDITIONS * BYTES_PER_AGENT * 12 / 10) / 1024 ))

# ───────────────────────────────────────────────────────────
# 3.  Print configuration
# ───────────────────────────────────────────────────────────
echo "=== Experiment Configuration ==="
echo "Experiment Name:    $EXPERIMENT_NAME"
echo "Plume Types:       ${PLUMES[*]}"
echo "Sensing Modes:     ${SENSING[*]}"
echo "Agents/Condition:  $AGENTS_PER_CONDITION"
echo "Agents/Job:        $AGENTS_PER_JOB"
echo "Total Jobs:        $TOTAL_JOBS"
echo "SLURM Concurrency: $SLURM_ARRAY_CONCURRENT"
echo "MATLAB Version:    $MATLAB_VERSION"
echo "Output Directory:  $OUTPUT_BASE"
echo "Required Disk:     $((REQUIRED_SPACE_KB / 1024)) MB"
echo "================================"


# (SLURM sets $SLURM_ARRAY_TASK_ID from 0 .. TOTAL_JOBS-1)
if [[ -n "$SLURM_ARRAY_TASK_ID" && $SLURM_ARRAY_TASK_ID -ge $TOTAL_JOBS ]]; then
  echo "Task $SLURM_ARRAY_TASK_ID exceeds TOTAL_JOBS=$TOTAL_JOBS — exiting."
  exit 0
fi

# ───────────────────────────────────────────────────────────
# 3.  Map array index → (plume,sensing) + agent-slice
# ───────────────────────────────────────────────────────────
CONDITION=$(( SLURM_ARRAY_TASK_ID % NUM_CONDITIONS ))
JOB_INDEX_IN_CONDITION=$(( SLURM_ARRAY_TASK_ID / NUM_CONDITIONS ))

PLUME_INDEX=$(( CONDITION / NUM_SENSING ))            # 0 or 1
SENSING_INDEX=$(( CONDITION % NUM_SENSING ))          # 0 or 1

PLUME_NAME=${PLUMES[$PLUME_INDEX]}
SENSING_NAME=${SENSING[$SENSING_INDEX]}

# Slice of agents handled by this job
START_AGENT=$(( JOB_INDEX_IN_CONDITION * AGENTS_PER_JOB + 1 ))
END_AGENT=$(( (JOB_INDEX_IN_CONDITION + 1) * AGENTS_PER_JOB ))
(( END_AGENT > AGENTS_PER_CONDITION )) && END_AGENT=$AGENTS_PER_CONDITION

# ───────────────────────────────────────────────────────────
# 4.  Echo job parameters
# ───────────────────────────────────────────────────────────
echo "──────── Job Context ────────"
echo " SLURM_ARRAY_TASK_ID : ${SLURM_ARRAY_TASK_ID:-N/A (not running in SLURM)}"
echo " Plume Type          : $PLUME_NAME"
echo " Sensing Mode        : $SENSING_NAME"
echo " Agent Range         : $START_AGENT – $END_AGENT of $AGENTS_PER_CONDITION"
echo " Agents / Job        : $AGENTS_PER_JOB"
echo " Total Jobs          : $TOTAL_JOBS"
echo " Max Concurrent      : $SLURM_ARRAY_CONCURRENT"
echo " Output Directory    : $OUTPUT_BASE"
echo "──────────────────────────────"

# ───────────────────────────────────────────────────────────
# 5.  Load configuration from YAML file
# ───────────────────────────────────────────────────────────
PLUME_CONFIG="configs/my_complex_plume_config.yaml"
if [[ ! -f "$PLUME_CONFIG" ]]; then
    echo "ERROR: Plume configuration file not found: $PLUME_CONFIG" >&2
    echo "Please create a configuration file or update the path in the script." >&2
    exit 1
fi

# Ensure the config file is readable
if [[ ! -r "$PLUME_CONFIG" ]]; then
    echo "ERROR: Cannot read configuration file: $PLUME_CONFIG" >&2
    exit 1
fi

echo "Using configuration from: $PLUME_CONFIG"

# ───────────────────────────────────────────────────────────
# 6.  Build a temporary MATLAB script that runs every agent slice serially
#     (1 CPU per MATLAB, so we keep the bash loop outside MATLAB)
# ───────────────────────────────────────────────────────────
# Create temporary file in a more robust way
TMP_DIR="${TMPDIR:-/tmp}"
MATLAB_SCRIPT=$(mktemp -p "$TMP_DIR" batch_job_XXXXXX.m)

if [[ ! -f "$MATLAB_SCRIPT" ]]; then
    echo "ERROR: Failed to create temporary MATLAB script" >&2
    exit 1
fi

# Ensure the script is deleted on exit
trap 'rm -f "$MATLAB_SCRIPT"' EXIT

# ───────────────────────────────────────────────────────────
# 7.  Create MATLAB script with initialization
# ─────────────────────────────────────────────────────────--
# Write the MATLAB script header and initialization
cat > "$MATLAB_SCRIPT" << 'EOL'
% Add Code directory to path if not already there
if isempty(which('run_navigation_cfg'))
    addpath(fullfile(pwd, 'Code'));
end

% Ensure we have the required functions
if ~exist('load_config', 'file') || ~exist('run_navigation_cfg', 'file')
    error('Required functions not found. Make sure the Code directory is on your path.');
end

% Set up error handling
original_warning_state = warning('off', 'all');
cleanupObj = onCleanup(@() warning(original_warning_state));

EOL

# Add agent-specific code for each agent
for AGENT_ID in $(seq $START_AGENT $END_AGENT); do
  SEED=$AGENT_ID                       # reproducible 1-to-1 seed
  OUT_DIR="${OUTPUT_BASE}/${PLUME_NAME}_${SENSING_NAME}/${AGENT_ID}_${SEED}"
  mkdir -p "$OUT_DIR"

  # Add agent simulation code to MATLAB script
  cat >> "$MATLAB_SCRIPT" << EOF
% Agent $AGENT_ID (Seed: $SEED)
try
    % Load the base configuration
    cfg = load_config('$PLUME_CONFIG');
    
    % Override with simulation-specific parameters
    cfg.bilateral = $([[ $SENSING_NAME == "bilateral" ]] && echo 'true' || echo 'false');
    cfg.randomSeed = $SEED;
    cfg.ntrials = 1;
    cfg.plotting = 0;
    cfg.outputDir = '$OUT_DIR';
    
    % Run the simulation
    fprintf('Starting simulation for seed %d...\\n', $SEED);
    result = run_navigation_cfg(cfg);
    
    % Save the results
    save(fullfile(cfg.outputDir, 'result.mat'), '-struct', 'result');
    fprintf('Successfully completed simulation for seed %d\\n', cfg.randomSeed);
    
    % Clear large variables to save memory
    clear result cfg;
    
catch ME
    fprintf('Error in simulation (seed %d): %s\\n', $SEED, getReport(ME));
    % Don't exit on error - continue with next agent
    continue;
end

% Add a small pause to prevent file system overload
pause(0.1);

EOF
done

echo "exit(0);" >> "$MATLAB_SCRIPT"


# ───────────────────────────────────────────────────────────
# 8.  Launch MATLAB to run the generated file
# ───────────────────────────────────────────────────────────
echo "Running MATLAB with script: $MATLAB_SCRIPT"
if ! matlab ${MATLAB_OPTIONS} -r "run('$MATLAB_SCRIPT')"; then
    echo "ERROR: MATLAB execution failed" >&2
    exit 1
fi

# ───────────────────────────────────────────────────────────
# 7.  Export results to processed format
# ───────────────────────────────────────────────────────────
echo "Exporting results to processed format..."

# Create a temporary MATLAB script for exporting results
EXPORT_SCRIPT=$(mktemp -p "$TMP_DIR" export_job_XXXXXX.m)
trap 'rm -f "$EXPORT_SCRIPT"' EXIT

# Find all result.mat files and create export commands
find "$RAW_DIR" -name 'result.mat' | while read -r result_file; do
    # Get the directory containing the result file
    result_dir=$(dirname "$result_file")
    
    # Create corresponding processed directory structure
    processed_dir=${result_dir/$RAW_DIR/}
    processed_dir="data/processed${processed_dir}"
    mkdir -p "$processed_dir"
    
    # Add export command to the script
    cat >> "$EXPORT_SCRIPT" <<EOF
try
    export_results('$result_file', '$processed_dir', 'Format', 'both');
    fprintf('Exported results to %s\\n', '$processed_dir');
catch ME
    fprintf('Error exporting %s: %s\\n', '$result_file', getReport(ME));
end

EOF
done

# Add exit command to the script
echo "exit;" >> "$EXPORT_SCRIPT"

# Run the export script if we found any result files
if [ -s "$EXPORT_SCRIPT" ]; then
    echo "Running export script..."
    if ! matlab -nodisplay -nosplash -r "run('$EXPORT_SCRIPT')"; then
        echo "WARNING: Some exports may have failed" >&2
    fi
else
    echo "No result files found to export"
fi

echo "Job completed successfully"
exit 0
