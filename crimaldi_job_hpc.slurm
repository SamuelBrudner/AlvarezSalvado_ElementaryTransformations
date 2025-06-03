#!/bin/bash
#SBATCH --job-name=crimaldi_hpc
#SBATCH --partition=day
#SBATCH --time=6:00:00
#SBATCH --mem=82G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-99%20
#SBATCH --output=logs/crimaldi-%A_%a.out
#SBATCH --error=logs/crimaldi-%A_%a.err

# Set up HPC path fix
export MATLAB_PLUME_FILE="/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations/data/plumes/10302017_10cms_bounded.hdf5"

# Create logs directory
mkdir -p logs results

# Load MATLAB
module load MATLAB/2023b

# Run simulation
matlab -nodisplay -nosplash -r "
addpath(genpath('Code'));

% Get task ID
task_id = str2double(getenv('SLURM_ARRAY_TASK_ID'));
if isnan(task_id), task_id = 0; end

fprintf('Task %d: Starting Crimaldi simulation\n', task_id);

try
    % Run with config duration (300s)
    out = navigation_model_vec('config', 'Crimaldi', 0, 10);
    
    % Save results
    filename = sprintf('results/crimaldi_results_%04d.mat', task_id);
    save(filename, 'out', '-v7.3');
    fprintf('Task %d: Success! Saved to %s\n', task_id, filename);
catch ME
    fprintf('Task %d ERROR: %s\n', task_id, ME.message);
    exit(1);
end
exit(0);
"
