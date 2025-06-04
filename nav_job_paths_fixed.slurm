#!/bin/bash
#SBATCH --job-name=nav_model
#SBATCH --partition=day
#SBATCH --time=6:00:00
#SBATCH --mem=82G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-99%20
#SBATCH --output=logs/nav-%A_%a.out
#SBATCH --error=logs/nav-%A_%a.err

# Set project directory
PROJECT_ROOT="/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations"

# Change to project directory
cd "$PROJECT_ROOT"

# Create directories
mkdir -p logs results

# Load MATLAB
module load MATLAB/2023b

# Get array task ID
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

# Create temporary MATLAB script
TEMP_MATLAB=$(mktemp /tmp/nav_${SLURM_JOB_ID}_${TASK_ID}_XXXXXX.m)

# Write MATLAB code using here-doc
cat > "$TEMP_MATLAB" << MATLAB_CODE
% Navigation model task ${TASK_ID}
fprintf('\n=== Navigation Model Task ${TASK_ID} ===\n');

% Setup
cd('${PROJECT_ROOT}');
addpath(genpath('Code'));

% Load configuration
try
    paths = load_paths();
    fprintf('Project root: %s\n', paths.project_root);
catch
    fprintf('Note: Could not load paths config\n');
end

% Task parameters
task_id = ${TASK_ID};

try
    % Load plume config
    fprintf('\nLoading configuration...\n');
    [plume_file, plume_config] = get_plume_file();
    
    % Get duration
    if isfield(plume_config, 'simulation') && isfield(plume_config.simulation, 'duration_seconds')
        duration = plume_config.simulation.duration_seconds;
    else
        duration = 300;
    end
    fprintf('Simulation duration: %.0f seconds\n', duration);
    
    % Calculate samples (15 Hz for Crimaldi)
    n_samples = round(duration * 15);
    fprintf('Total samples: %d at 15 Hz\n', n_samples);
    
    % Run simulation
    fprintf('\nRunning navigation simulation...\n');
    tic;
    out = navigation_model_vec(n_samples, 'Crimaldi', 0, 10);
    elapsed = toc;
    
    % Save results
    filename = sprintf('results/nav_results_%04d.mat', task_id);
    save(filename, 'out', '-v7.3');
    
    % Report
    fprintf('\n=== Task %d Complete ===\n', task_id);
    fprintf('Agents: %d\n', size(out.x,2));
    fprintf('Duration: %.1f seconds simulated\n', size(out.x,1)/15);
    fprintf('Runtime: %.1f seconds actual\n', elapsed);
    fprintf('Output: %s\n', filename);
    
    if exist(filename, 'file')
        d = dir(filename);
        fprintf('Size: %.1f MB\n', d.bytes/1e6);
    end
    
    if isfield(out, 'successrate')
        fprintf('Success rate: %.1f%%\n', out.successrate * 100);
    end
    
catch ME
    fprintf('\n=== Task %d FAILED ===\n', task_id);
    fprintf('Error: %s\n', ME.message);
    for i = 1:length(ME.stack)
        fprintf('  %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
    exit(1);
end

% Exit successfully
exit(0);
MATLAB_CODE

# Run MATLAB with the temporary script
matlab -nodisplay -nosplash -r "run('$TEMP_MATLAB')"

# Capture MATLAB exit code
MATLAB_EXIT=$?

# Clean up
rm -f "$TEMP_MATLAB"

# Exit with MATLAB's exit code
exit $MATLAB_EXIT
