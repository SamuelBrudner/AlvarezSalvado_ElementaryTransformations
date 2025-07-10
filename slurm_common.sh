#!/usr/bin/env bash
# slurm_common.sh
# Shared helper functions for navigation-model SLURM jobs.
# Source this script from any *.slurm file:  source slurm_common.sh
#
# The goal is to keep all boilerplate (module loading, directory setup,
# logging helpers) in one place so individual SLURM job scripts focus only
# on configuration (memory, time, plume JSON, etc.).

set -euo pipefail

###########################################################################
# Module / environment helpers
###########################################################################

load_matlab() {
  # Unset DISPLAY to prevent MATLAB from trying to open X-windows
  unset DISPLAY
  # Adjust MATLAB_JAVA path if the cluster defines JAVA_HOME
  if [[ -n "${JAVA_HOME:-}" ]]; then
    export MATLAB_JAVA="$JAVA_HOME/jre"
  fi

  # Load the requested MATLAB module (version is cluster-specific)
  module load MATLAB/2023b

  echo "MATLAB module loaded ($(which matlab))"
}

###########################################################################
# Directory helpers
###########################################################################

prepare_logs() {
  local log_dir=${1:-logs}
  local results_dir=${2:-results}
  mkdir -p "$log_dir" "$results_dir"
}

###########################################################################
# MATLAB execution helper
###########################################################################

run_matlab_batch() {
  # Run an arbitrary MATLAB command in batch mode and propagate exit code.
  # Usage: run_matlab_batch "my_matlab_command"
  local cmd="$1"

  matlab -nodisplay -nosplash -nodesktop -nojvm -batch "$cmd"
  return $?
}

###########################################################################
# Utility
###########################################################################

log_header() {
  local title="$1"
  echo "======================================"
  echo "$title"
  echo "======================================"
  echo "Job started at: $(date)"
  echo "Running on: $(hostname)"
  echo "Array task: ${SLURM_ARRAY_TASK_ID:-N/A}"
  echo "Working directory: $(pwd)"
  echo "======================================"
}
