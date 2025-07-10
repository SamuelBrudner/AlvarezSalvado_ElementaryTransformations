#!/bin/bash
# nav_job_crimaldi.slurm - SLURM job script specifically for Crimaldi plume
#
# Runs navigation_model_vec with Crimaldi plume environment
# Array job: 400 tasks (0-399), 100 concurrent max
# Memory: 32GB per task (Crimaldi plume is ~9GB uncompressed)
#
# Usage: sbatch nav_job_crimaldi.slurm
#        sbatch --array=0-0 nav_job_crimaldi.slurm  # Single test
#
# Output: results/nav_results_XXXX.mat (where XXXX is array task ID)

#SBATCH --job-name=nav_crim
#SBATCH --partition=day
#SBATCH --time=6:00:00
#SBATCH --mem=32G         # Higher than smoke due to uncompressed file
#SBATCH --cpus-per-task=1
#SBATCH --array=0-0%1  # Quick test: only one job
#SBATCH --output=logs/crim-%A_%a.out
#SBATCH --error=logs/crim-%A_%a.err

# Ensure no display
unset DISPLAY
export MATLAB_JAVA=$JAVA_HOME/jre

# Load MATLAB module (required on Grace)
module load MATLAB/2023b

# Setup
cd /vast/palmer/home.grace/snb6/Documents/AlvarezSalvado_ElementaryTransformations
mkdir -p logs results

# Force Crimaldi plume configuration
export PLUME=crimaldi

# Log info
echo "======================================"
echo "Crimaldi Plume Navigation Simulation"
echo "======================================"
echo "Job started at: $(date)"
echo "Running on: $(hostname)"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Plume: Crimaldi (15 Hz, 0.74 mm/px)"
echo ""

# Run MATLAB
matlab -nodisplay -nosplash -nodesktop -nojvm << 'EOF'
% Add code to path
addpath(genpath('Code'));

% Get array task ID
task_id = str2double(getenv('SLURM_ARRAY_TASK_ID'));
if isnan(task_id), task_id = 0; end

% Force Crimaldi plume configuration
fprintf('Configuring for Crimaldi plume...\n');
crim_config = fullfile(pwd, 'configs/plumes/crimaldi_10cms_bounded.json');
paths_file = 'configs/paths.json';

% Update paths to use Crimaldi
paths = jsondecode(fileread(paths_file));
paths.plume_config = crim_config;
fid = fopen(paths_file, 'w');
fprintf(fid, '%s', jsonencode(paths));
fclose(fid);

% Verify configuration
[plume_file, plume_config] = get_plume_file();
fprintf('\nActive plume: %s\n', plume_config.plume_id);
fprintf('Frame rate: %.1f Hz\n', plume_config.temporal.frame_rate);
fprintf('Pixel scale: %.3f mm/px\n', plume_config.spatial.mm_per_pixel);

% QUICK TEST: reduce simulation duration and agents
n_agents = 2;
duration_seconds = 5;  % 5 seconds for fast test
n_frames = round(duration_seconds * 15);
fprintf('[QUICK TEST] Running %d agents for %.1f seconds (%d frames at 15 Hz)\n', n_agents, duration_seconds, n_frames);

% Run simulation
try
    fprintf('\nStarting simulation for task %d...\n', task_id);
    tic;
    
    out = navigation_model_vec(n_frames, 'Crimaldi', 0, 10);
    
    elapsed = toc;
    fprintf('Simulation completed in %.1f seconds\n', elapsed);
    
    % Save with standard prefix
    filename = sprintf('results/nav_results_%04d.mat', task_id);
    save(filename, 'out', '-v7.3');
    fprintf('Results saved to %s\n', filename);
    
catch ME
    fprintf('ERROR: %s\n', ME.message);
    exit(1);
end

exit(0);
EOF

echo ""
echo "Job completed at: $(date)"