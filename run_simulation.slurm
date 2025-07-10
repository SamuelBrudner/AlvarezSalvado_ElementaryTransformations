#!/bin/bash
#SBATCH --job-name=nav_model
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --partition=compute
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Usage: sbatch run_simulation.slurm /path/to/matlab/Code

# Get MATLAB code directory from command line
MATLAB_CODE_DIR="$1"

if [ -z "$MATLAB_CODE_DIR" ]; then
    echo "Error: Please provide MATLAB code directory as argument"
    echo "Usage: sbatch run_simulation.slurm /path/to/matlab/Code"
    exit 1
fi

# Load MATLAB
module load matlab/R2023b

# Run simulation
matlab -nodisplay -nosplash -batch "addpath('${MATLAB_CODE_DIR}'); try, fprintf('Starting simulation...\n'); out = Elifenavmodel_bilateral(3600, 'crimaldi', 0, 100); save('results_${SLURM_JOB_ID}.mat', 'out', '-v7.3'); fprintf('Results saved to: results_${SLURM_JOB_ID}.mat\n'); catch ME, fprintf('Error: %s\n', ME.message); exit(1); end; exit(0);"