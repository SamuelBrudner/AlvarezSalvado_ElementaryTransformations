#!/bin/bash

# ---------------------------------------------------------------------------
# Submit jobs to SLURM for running olfactory navigation simulations.
#
# Usage:
#   ./slurm_submit.sh [output_file]
#   ./slurm_submit.sh -h|--help
#
# Environment variables:
#   TRIAL_LENGTH        Duration of each trial in ms (default: 5000)
#   ENVIRONMENT         Environment name (default: Crimaldi)
#   OUTPUT_DIR          Directory for output files (default: current dir)
#   AGENTS_PER_CONDITION Number of agents per condition (default: 1000)
#   AGENTS_PER_JOB      Number of agents simulated per SLURM task (default: 10)
#   PARTITION           Cluster partition to use (default: day)
#   TIME_LIMIT          Wall time limit for each task (default: 6:00:00)
#   MEM_PER_TASK        Memory per task (default: 64G)
#   MAX_CONCURRENT      Maximum concurrent tasks in array (default: 100)
#   EXP_NAME            Name prefix for the job (default: crimaldi)
#
# Example:
#   AGENTS_PER_CONDITION=20 AGENTS_PER_JOB=5 ./slurm_submit.sh job.slurm
# ---------------------------------------------------------------------------

set -euo pipefail

usage() {
    sed -n '3,24p' "$0"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

script_dir="$(cd "$(dirname "$0")" && pwd)"
template="$script_dir/slurm_job_template.slurm"
output_file="${1:-$script_dir/generated_job.slurm}"

trial_length="${TRIAL_LENGTH:-3600}"
environment="${ENVIRONMENT:-Crimaldi}"
output_dir="${OUTPUT_DIR:-$PWD}"
project_dir="$script_dir"  # The project root is where this script lives

agents_per_condition="${AGENTS_PER_CONDITION:-1000}"
agents_per_job="${AGENTS_PER_JOB:-10}"
partition="${PARTITION:-day}"
time_limit="${TIME_LIMIT:-6:00:00}"
mem_per_task="${MEM_PER_TASK:-82G}"
max_concurrent="${MAX_CONCURRENT:-100}"
exp_name="${EXP_NAME:-crimaldi}"

total_jobs=$(( (agents_per_condition + agents_per_job - 1) / agents_per_job * 4 ))
array_upper=$(( total_jobs - 1 ))

# Create logs directory if it doesn't exist
logs_dir="${output_dir}/${exp_name}_logs"
mkdir -p "$logs_dir"

printf 'Using template: %s\n' "$template" >&2
printf 'Output file will be: %s\n' "$output_file" >&2
printf 'Project directory: %s\n' "$project_dir" >&2
printf 'Logs directory: %s\n' "$logs_dir" >&2
printf 'Calculated total_jobs=%s array_upper=%s\n' "$total_jobs" "$array_upper" >&2

export TRIAL_LENGTH="$trial_length" ENVIRONMENT="$environment" OUTPUT_DIR="$output_dir" \
       PROJECT_DIR="$project_dir" AGENTS_PER_JOB="$agents_per_job" PARTITION="$partition" \
       TIME_LIMIT="$time_limit" MEM_PER_TASK="$mem_per_task" MAX_CONCURRENT="$max_concurrent" \
       EXP_NAME="$exp_name" ARRAY_UPPER="$array_upper"

>&2 echo "Writing SLURM script to $output_file with $total_jobs array task(s)"

envsubst < "$template" > "$output_file"
cat "$output_file"