#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
template="$script_dir/slurm_job_template.slurm"
output_file="${1:-$script_dir/generated_job.slurm}"

trial_length="${TRIAL_LENGTH:-5000}"
environment="${ENVIRONMENT:-Crimaldi}"
output_dir="${OUTPUT_DIR:-$PWD}"

agents_per_condition="${AGENTS_PER_CONDITION:-1000}"
agents_per_job="${AGENTS_PER_JOB:-10}"
partition="${PARTITION:-day}"
time_limit="${TIME_LIMIT:-6:00:00}"
mem_per_task="${MEM_PER_TASK:-64G}"
max_concurrent="${MAX_CONCURRENT:-100}"
exp_name="${EXP_NAME:-crimaldi}"

total_jobs=$(( (agents_per_condition + agents_per_job - 1) / agents_per_job * 4 ))
array_upper=$(( total_jobs - 1 ))

export TRIAL_LENGTH="$trial_length" ENVIRONMENT="$environment" OUTPUT_DIR="$output_dir" \
       AGENTS_PER_JOB="$agents_per_job" PARTITION="$partition" TIME_LIMIT="$time_limit" \
       MEM_PER_TASK="$mem_per_task" MAX_CONCURRENT="$max_concurrent" EXP_NAME="$exp_name" \
       ARRAY_UPPER="$array_upper"

>&2 echo "Writing SLURM script to $output_file with $total_jobs array task(s)"

envsubst < "$template" > "$output_file"
cat "$output_file"
