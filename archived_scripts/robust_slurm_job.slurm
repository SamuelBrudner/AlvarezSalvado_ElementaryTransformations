#!/bin/bash
#SBATCH --job-name=crimaldi_sim
#SBATCH --partition=day
#SBATCH --time=6:00:00
#SBATCH --mem=82G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-399%100
#SBATCH --output=/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations/crimaldi_logs/slurm-%A_%a.out
#SBATCH --error=/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations/crimaldi_logs/slurm-%A_%a.err

# Exit on any error
set -e

# Load MATLAB module if needed (uncomment and adjust for your cluster)
# module load matlab/R2023b

# Change to project directory
PROJECT_DIR="/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations"
cd "$PROJECT_DIR"

# Verify we're in the right directory
echo "Current directory: $(pwd)"
echo "Contents of Code directory:"
ls -la Code/ | head -10

# Set up parameters
TRIAL_LENGTH=3600
ENVIRONMENT="Crimaldi"
AGENTS_PER_JOB=10

# Create a MATLAB script file to avoid command line escaping issues
MATLAB_SCRIPT=$(mktemp /tmp/matlab_nav_XXXXXX.m)

cat > "$MATLAB_SCRIPT" << 'EOF'
% Navigation model simulation script
fprintf('MATLAB simulation started at %s\n', datestr(now));
fprintf('Current directory: %s\n', pwd);

% Add all subdirectories in Code to path
addpath(genpath('Code'));

% Verify the function exists
if exist('navigation_model_vec', 'file') ~= 2
    error('navigation_model_vec function not found in path!');
end

% Set parameters
trial_length = 3600;
environment = 'Crimaldi';
plotting = 0;
num_agents = 10;

% Run simulation
fprintf('Running simulation with parameters:\n');
fprintf('  Trial length: %d\n', trial_length);
fprintf('  Environment: %s\n', environment);
fprintf('  Number of agents: %d\n', num_agents);
fprintf('  Array task ID: %s\n', getenv('SLURM_ARRAY_TASK_ID'));

try
    out = navigation_model_vec(trial_length, environment, plotting, num_agents);
    
    % Save results with array task ID
    task_id = getenv('SLURM_ARRAY_TASK_ID');
    if isempty(task_id)
        task_id = '0';
    end
    
    output_file = sprintf('results_array_%s.mat', task_id);
    save(output_file, 'out', '-v7.3');
    fprintf('Results saved to %s\n', output_file);
    
catch ME
    fprintf('ERROR: %s\n', ME.message);
    fprintf('Stack trace:\n');
    for i = 1:length(ME.stack)
        fprintf('  File: %s\n  Name: %s\n  Line: %d\n', ...
                ME.stack(i).file, ME.stack(i).name, ME.stack(i).line);
    end
    exit(1);
end

fprintf('Simulation completed successfully\n');
exit(0);
EOF

# Run MATLAB with the script file
echo "Running MATLAB script: $MATLAB_SCRIPT"
matlab -nodisplay -nosplash < "$MATLAB_SCRIPT"

# Clean up
rm -f "$MATLAB_SCRIPT"