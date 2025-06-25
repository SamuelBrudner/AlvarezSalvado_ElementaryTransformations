#!/bin/bash

set -euo pipefail

# Initialize verbose logging flag
VERBOSE=0

usage() {
    cat <<'USAGE'
Usage: slurm_submit.sh [OPTIONS] [OUTPUT_FILE]

Options:
  -v, --verbose      Enable verbose logging with detailed trace output
  -h, --help         Show this help message

Environment variables:
  TRIAL_LENGTH       Length of each trial
  ENVIRONMENT        Environment name
  OUTPUT_DIR         Directory for output
  AGENTS_PER_CONDITION
  AGENTS_PER_JOB
  PARTITION
  TIME_LIMIT
  MEM_PER_TASK
  MAX_CONCURRENT
  EXP_NAME
USAGE
}

# Function for verbose logging
log_verbose() {
    [[ $VERBOSE -eq 1 ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] slurm_submit.sh: $*" || true
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=1
            log_verbose "Verbose logging enabled"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "Error: Unknown option $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            # First non-option argument is the output file
            if [[ -z ${OUTPUT:-} ]]; then
                OUTPUT="$1"
            else
                echo "Error: Unexpected argument $1" >&2
                usage >&2
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate required output path
if [[ -z ${OUTPUT:-} ]]; then
    echo "Error: Output path required" >&2
    usage >&2
    exit 1
fi

log_verbose "Output file: $OUTPUT"

# Set default values for environment variables
AGENTS_PER_CONDITION=${AGENTS_PER_CONDITION:-100}
AGENTS_PER_JOB=${AGENTS_PER_JOB:-10}
EXP_NAME=${EXP_NAME:-nav}
PARTITION=${PARTITION:-day}
TIME_LIMIT=${TIME_LIMIT:-1:00:00}
MEM_PER_TASK=${MEM_PER_TASK:-4G}
MAX_CONCURRENT=${MAX_CONCURRENT:-100}

log_verbose "Configuration: AGENTS_PER_CONDITION=$AGENTS_PER_CONDITION, AGENTS_PER_JOB=$AGENTS_PER_JOB"
log_verbose "SLURM settings: PARTITION=$PARTITION, TIME_LIMIT=$TIME_LIMIT, MEM_PER_TASK=$MEM_PER_TASK"

# Calculate job parameters
CONDITIONS=4
JOBS_PER_COND=$(( (AGENTS_PER_CONDITION + AGENTS_PER_JOB -1)/AGENTS_PER_JOB ))
TOTAL_JOBS=$(( JOBS_PER_COND * CONDITIONS ))
ARRAY_UPPER=$(( TOTAL_JOBS - 1 ))

log_verbose "Job calculation: CONDITIONS=$CONDITIONS, JOBS_PER_COND=$JOBS_PER_COND"
log_verbose "Array parameters: TOTAL_JOBS=$TOTAL_JOBS, ARRAY_UPPER=$ARRAY_UPPER"

# Create log directory using the new slurm_logs structure
LOG_DIR="slurm_logs/${EXP_NAME}"

log_verbose "Creating directories: $(dirname "$OUTPUT") and $LOG_DIR"
mkdir -p "$(dirname "$OUTPUT")" "$LOG_DIR"

# Ensure logs directory exists for verbose output
if [[ $VERBOSE -eq 1 ]]; then
    mkdir -p logs
    log_verbose "Created logs directory for verbose output"
fi

log_verbose "Generating SLURM job script at $OUTPUT"

# Generate the SLURM job script with updated template reference
cat > "$OUTPUT" <<EOF2
#!/bin/bash
#SBATCH --job-name=${EXP_NAME}_sim
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --mem=${MEM_PER_TASK}
#SBATCH --output=${LOG_DIR}/${EXP_NAME}_logs_%A_%a.out
#SBATCH --error=${LOG_DIR}/${EXP_NAME}_logs_%A_%a.err
#SBATCH --array=0-${ARRAY_UPPER}%${MAX_CONCURRENT}

$(cat slurm/slurm_job_template.slurm)
EOF2

log_verbose "SLURM job script generated successfully"

# Output summary information
{
    echo "total_jobs=$TOTAL_JOBS"
    echo "array_upper=$ARRAY_UPPER"
    echo "output_file=$OUTPUT"
} >&2

log_verbose "Job summary written to stderr"
log_verbose "SLURM submission script creation completed"
