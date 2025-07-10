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

# Change to the project directory
cd $SLURM_SUBMIT_DIR

# Run the verification script
matlab -nodisplay -nosplash -r "verify_simulation_results; exit;"
