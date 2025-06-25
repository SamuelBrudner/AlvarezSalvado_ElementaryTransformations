#!/bin/bash
# nav_job_slurm.sh - SLURM job script for smoke plume navigation simulations
#
# Runs navigation_model_vec with smoke plume environment
# Array job: 400 tasks (0-399), 100 concurrent max
# Memory: 16GB per task (smoke plume is compressed to 0.58GB)
#
# Usage: sbatch slurm/nav_job_slurm.sh
#        sbatch --array=0-0 slurm/nav_job_slurm.sh  # Single test
#
# Output: results/smoke_nav_results_XXXX.mat (where XXXX is array task ID)
#
# Moved from repository root to slurm/ directory per Section 0 requirement
# to consolidate SLURM job management and improve discoverability

#SBATCH --job-name=nav_smoke
#SBATCH --partition=day
#SBATCH --time=2:00:00    # Reduced from 6h - smoke sims are faster
#SBATCH --mem=16G         # Reduced from 82G - compressed file
#SBATCH --cpus-per-task=1
#SBATCH --array=0-399%100
#SBATCH --output=slurm_logs/nav_smoke/nav_smoke-%A_%a.out
#SBATCH --error=slurm_logs/nav_smoke/nav_smoke-%A_%a.err

# Ensure no display
unset DISPLAY
export MATLAB_JAVA=$JAVA_HOME/jre

# Load MATLAB module (required on Grace)
module load MATLAB/2023b

# Get project directory using symlink-safe path
PROJECT_DIR="/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations"
cd "$PROJECT_DIR" || exit 1

# Create necessary directories
mkdir -p logs results slurm_logs/nav_smoke

# Log info
echo "======================================"
echo "Smoke Plume Navigation Simulation"
echo "======================================"
echo "Job started at: $(date)"
echo "Running on: $(hostname)"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Working directory: $(pwd)"
echo "SLURM job script: slurm/nav_job_slurm.sh"
echo ""

# Create MATLAB script in temp file
TEMP_MATLAB=$(mktemp /tmp/smoke_nav_$$.m)
cat > "$TEMP_MATLAB" << 'MATLAB_EOF'
% Smoke plume navigation simulation
try
    % Get project directory and task ID from environment
    project_dir = getenv('PROJECT_DIR');
    task_id = str2double(getenv('SLURM_ARRAY_TASK_ID'));
    if isnan(task_id), task_id = 0; end
    
    % Change to project directory
    cd(project_dir);
    addpath(genpath('Code'));
    
    % Adjust task ID for comparative studies (1000+ becomes 0+)
    actual_task_id = task_id;
    if task_id >= 1000
        actual_task_id = task_id - 1000;
        fprintf('Comparative study mode: task %d -> %d\n', task_id, actual_task_id);
    end
    
    % Force smoke plume configuration
    fprintf('Configuring for smoke plume...\n');
    smoke_config_path = fullfile(project_dir, 'configs/plumes/smoke_1a_backgroundsubtracted.json');
    paths_file = fullfile(project_dir, 'configs/paths.json');
    
    % Update paths to use smoke
    paths_data = jsondecode(fileread(paths_file));
    paths_data.plume_config = smoke_config_path;
    
    fid = fopen(paths_file, 'w');
    fprintf(fid, '%s', jsonencode(paths_data));
    fclose(fid);
    
    % Verify configuration
    [plume_file, plume_config] = get_plume_file();
    fprintf('\nActive plume: %s\n', plume_config.plume_id);
    fprintf('Frame rate: %.1f Hz\n', plume_config.temporal.frame_rate);
    fprintf('Pixel scale: %.3f mm/px\n', plume_config.spatial.mm_per_pixel);
    
    % Check if plume file exists
    if ~exist(plume_file, 'file')
        error('Plume file not found: %s', plume_file);
    end
    
    % Calculate frames for smoke plume at 60 Hz
    duration_seconds = plume_config.simulation.duration_seconds;
    n_frames = round(duration_seconds * 60);
    fprintf('Simulation: %.1f seconds = %d frames at 60 Hz\n', duration_seconds, n_frames);
    
    % Run simulation (note: we use 'Crimaldi' as environment type, which loads from config)
    fprintf('\nStarting simulation for task %d...\n', task_id);
    tic;
    
    % The model will use the configured plume file regardless of environment string
    out = navigation_model_vec(n_frames, 'Crimaldi', 0, 10);
    
    elapsed = toc;
    fprintf('Simulation completed in %.1f seconds\n', elapsed);
    
    % Save with smoke prefix (use original task_id for filename)
    filename = sprintf('results/smoke_nav_results_%04d.mat', task_id);
    save(filename, 'out', '-v7.3');
    fprintf('Results saved to %s\n', filename);
    fprintf('✓ Task %d completed successfully\n', task_id);
    
catch ME
    fprintf('ERROR: %s\n', ME.message);
    fprintf('Stack trace:\n');
    for i = 1:length(ME.stack)
        fprintf('  In %s at line %d\n', ME.stack(i).name, ME.stack(i).line);
    end
    exit(1);
end

exit(0);
MATLAB_EOF

# Export variables for MATLAB
export PROJECT_DIR
export SLURM_ARRAY_TASK_ID

# Run MATLAB with the temp script
matlab -nodisplay -nosplash -nodesktop -nojvm -r "run('$TEMP_MATLAB')"
MATLAB_EXIT=$?

# Clean up
rm -f "$TEMP_MATLAB"

echo ""
echo "Job completed at: $(date)"
echo "Exit code: $MATLAB_EXIT"
echo "Log files: slurm_logs/nav_smoke/nav_smoke-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.{out,err}"

exit $MATLAB_EXIT