#!/bin/bash
#SBATCH --job-name=verify_results
#SBATCH --output=verify_results_%j.log
#SBATCH --error=verify_results_%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Load MATLAB module (adjust based on your HPC environment)
module load matlab

# Change to the project root (directory containing this script), regardless of submit location
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || { echo "[ERROR] Cannot cd to $SCRIPT_DIR"; exit 1; }
echo "[INFO] Running verification from $PWD"

# Run the verification script
matlab -nodisplay -nosplash -r "verify_simulation_results; exit;"
