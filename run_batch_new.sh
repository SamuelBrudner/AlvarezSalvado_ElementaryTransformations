#!/bin/bash
set -euo pipefail

# Default configuration file
CONFIG_FILE="${1:-configs/batch_job_config.yaml}"

# ───────────────────────────────────────────────────────────
# SLURM Configuration
# ───────────────────────────────────────────────────────────
#SBATCH --begin=now
#SBATCH --job-name=plume_sim
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --partition=day
#SBATCH --time=6:00:00
#SBATCH --output=slurm_out/%A_%a.out
#SBATCH --error=slurm_err/%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=${USER}@yale.edu

# ───────────────────────────────────────────────────────────
# 1.  Setup and Configuration
# ───────────────────────────────────────────────────────────

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create necessary directories
mkdir -p slurm_out slurm_err data/raw

# Check if running under SLURM
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "WARNING: Not running under SLURM, using default values"
    SLURM_ARRAY_TASK_ID=0
fi

# ───────────────────────────────────────────────────────────
# 2.  Run MATLAB Simulation
# ───────────────────────────────────────────────────────────

echo "Starting job $SLURM_ARRAY_TASK_ID"

# Call MATLAB to run the simulation
matlab -nodisplay -nosplash -r \
    "addpath('Code'); "\
    "try, "\
    "  run_batch_job($SLURM_ARRAY_TASK_ID, '$CONFIG_FILE'); "\
    "catch ME, "\
    "  disp(getReport(ME)); "\
    "  exit(1); "\
    "end, "\
    "exit;"

# Check MATLAB exit status
if [[ $? -ne 0 ]]; then
    echo "ERROR: MATLAB simulation failed" >&2
    exit 1
fi

echo "Job $SLURM_ARRAY_TASK_ID completed successfully"
exit 0
