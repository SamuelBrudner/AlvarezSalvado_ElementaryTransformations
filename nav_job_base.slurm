#!/bin/bash
# nav_job_base.slurm — parameterised SLURM job for the navigation model
#
# Usage examples (submit 400-task array; Crimaldi plume):
#   sbatch --export=ALL,PLUME_JSON=configs/plumes/crimaldi_10cms_bounded.json,\
#          FRAME_RATE=15,RESULTS_PREFIX=crim_nav nav_job_base.slurm
#
# Override resources with standard sbatch flags if needed, e.g.:
#   sbatch -J nav_crim --mem=32G --time=6:00:00 ... nav_job_base.slurm
#
# Required exported variables               Default        Description
# ─────────────────────────────────────────────────────────────────────
# PLUME_JSON         (none)   Path to plume config JSON.
# FRAME_RATE         (auto)   Target Hz to simulate; defaults to config value.
# RESULTS_PREFIX     nav      Prefix for saved .mat result files.
#
# The script relies on `slurm_common.sh` for shared helpers and on
# `nav_driver.m` as the MATLAB entry point.

#SBATCH --job-name=nav
#SBATCH --partition=day
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-399%100
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Abort on unset vars and errors
set -euo pipefail

# Locate repo root (directory containing this script)
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# Ensure mandatory variables are set
: "${PLUME_JSON:?Need to export PLUME_JSON}" || exit 1
export FRAME_RATE="${FRAME_RATE:-}"   # may be empty → MATLAB autodetect
export RESULTS_PREFIX="${RESULTS_PREFIX:-nav_results}"

# Shared helpers
source slurm_common.sh

load_matlab
prepare_logs
log_header "Navigation model simulation"

# Export for MATLAB consumption
export PROJECT_DIR PLUME_JSON FRAME_RATE RESULTS_PREFIX

run_matlab_batch "nav_driver"
EXIT_CODE=$?
# (run_matlab_batch already uses -batch; just confirm and document)

echo ""
echo "Job completed at: $(date)"
exit $EXIT_CODE
