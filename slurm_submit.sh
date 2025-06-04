#!/bin/bash

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: slurm_submit.sh [OUTPUT_FILE]

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

if [[ ${1:-} == '-h' || ${1:-} == '--help' ]]; then
    usage
    exit 0
fi

OUTPUT=${1:?"Output path required"}

AGENTS_PER_CONDITION=${AGENTS_PER_CONDITION:-100}
AGENTS_PER_JOB=${AGENTS_PER_JOB:-10}
EXP_NAME=${EXP_NAME:-nav}
PARTITION=${PARTITION:-day}
TIME_LIMIT=${TIME_LIMIT:-1:00:00}
MEM_PER_TASK=${MEM_PER_TASK:-4G}
MAX_CONCURRENT=${MAX_CONCURRENT:-100}

CONDITIONS=4
JOBS_PER_COND=$(( (AGENTS_PER_CONDITION + AGENTS_PER_JOB -1)/AGENTS_PER_JOB ))
TOTAL_JOBS=$(( JOBS_PER_COND * CONDITIONS ))
ARRAY_UPPER=$(( TOTAL_JOBS - 1 ))

mkdir -p "$(dirname "$OUTPUT")"

cat > "$OUTPUT" <<EOF2
#!/bin/bash
#SBATCH --job-name=${EXP_NAME}_sim
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --mem=${MEM_PER_TASK}
#SBATCH --array=0-${ARRAY_UPPER}%${MAX_CONCURRENT}

$(cat slurm_job_template.slurm)
EOF2

{
    echo "total_jobs=$TOTAL_JOBS"
    echo "array_upper=$ARRAY_UPPER"
    echo "output_file=$OUTPUT"
} >&2


