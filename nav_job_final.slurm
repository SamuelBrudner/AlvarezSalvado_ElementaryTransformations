#!/bin/bash
# nav_job_final.slurm - Main SLURM job script for navigation model
#
# Runs navigation_model_vec with Crimaldi plume environment
# Array job: 400 tasks (0-399), 100 concurrent max
# Memory: 82GB per task (required for HDF5 plume data)
#
# Usage: sbatch nav_job_final.slurm
#        sbatch --array=0-0 nav_job_final.slurm  # Single test
#
# Output: results/nav_results_XXXX.mat (where XXXX is array task ID)

#SBATCH --job-name=nav_model
#SBATCH --partition=day
#SBATCH --time=6:00:00
#SBATCH --mem=82G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-399%100
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err

# Load MATLAB module (required on Grace)
module load MATLAB/2023b

# Setup
cd /home/snb6/Documents/AlvarezSalvado_ElementaryTransformations
mkdir -p logs results

# Log info
echo "Job started at: $(date)"
echo "Running on: $(hostname)"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"

# Run MATLAB
matlab -nodisplay -nosplash << 'EOF'
% Add code to path
addpath(genpath('Code'));

% Get array task ID
task_id = str2double(getenv('SLURM_ARRAY_TASK_ID'));
if isnan(task_id), task_id = 0; end

% Run simulation
try
    fprintf('Starting simulation for task %d\n', task_id);
    out = navigation_model_vec(3600, 'Crimaldi', 0, 10);
    
    % Save results
    filename = sprintf('results/nav_results_%04d.mat', task_id);
    save(filename, 'out', '-v7.3');
    fprintf('Results saved to %s\n', filename);
    
catch ME
    fprintf('ERROR: %s\n', ME.message);
    exit(1);
end

exit(0);
EOF

echo "Job completed at: $(date)"