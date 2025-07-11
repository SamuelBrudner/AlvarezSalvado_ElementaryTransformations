#!/bin/bash
# nav_job_template.slurm — generic SLURM job for navigation_model_vec
#
# Usage:
#   sbatch --array=0-399%100 jobs/nav_job_template.slurm crimaldi
#   sbatch jobs/nav_job_template.slurm smoke
#
# The first positional argument (PLUME_ID) selects the plume configuration.
# Supported values: crimaldi | smoke
# -----------------------------------------------------------------------------

#SBATCH --job-name=nav_plume
#SBATCH --partition=day
#SBATCH --time=6:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

module load MATLAB/2023b

# ----- Repo root (update if different on cluster) ---------------------------
PROJECT_DIR="/home/${USER}/Documents/AlvarezSalvado_ElementaryTransformations"
cd "$PROJECT_DIR" || { echo "Could not cd to $PROJECT_DIR"; exit 1; }
mkdir -p logs results

# ----- Parse arguments ------------------------------------------------------
if [[ $# -lt 1 ]]; then
  echo "PLUME_ID argument required (crimaldi | smoke)" >&2; exit 2;
fi
PLUME_ID="$1"

case "$PLUME_ID" in
  crimaldi)
    CONFIG_PATH="$PROJECT_DIR/configs/plumes/crimaldi_10cms_bounded.json"
    ;;
  smoke)
    CONFIG_PATH="$PROJECT_DIR/configs/plumes/smoke_1a_backgroundsubtracted.json"
    ;;
  *)
    echo "Unknown PLUME_ID: $PLUME_ID" >&2; exit 3;
    ;;
esac

# ----- Diagnostics ----------------------------------------------------------
echo "==============================================="
echo "Navigation Simulation – $PLUME_ID plume"
echo "==============================================="

echo "[INFO] SLURM Job ID     : $SLURM_JOB_ID"
echo "[INFO] Array Task ID    : ${SLURM_ARRAY_TASK_ID:-0}"
echo "[INFO] Node             : $(hostname)"
echo "[INFO] Config JSON      : $CONFIG_PATH"
head -n 10 "$CONFIG_PATH" || echo "[WARN] Could not read config JSON"

# ----- Export for MATLAB ----------------------------------------------------
export PLUME_CONFIG="$CONFIG_PATH"

# ----- Run MATLAB -----------------------------------------------------------
echo "[INFO] Starting MATLAB…"
matlab -nodisplay -nosplash -nodesktop -batch "run_navigation_job('$CONFIG_PATH')"
MATLAB_EXIT=$?

echo "[INFO] MATLAB finished with exit code $MATLAB_EXIT"
exit $MATLAB_EXIT
