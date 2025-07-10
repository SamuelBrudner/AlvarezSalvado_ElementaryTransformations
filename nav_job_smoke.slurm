#!/bin/bash
# nav_job_smoke.slurm - SLURM job script specifically for smoke plume
#
# Runs navigation_model_vec with smoke plume environment
# Array job: 400 tasks (0-399), 100 concurrent max
# Memory: 16GB per task (smoke plume is compressed to 0.58GB)
#
# Usage: sbatch nav_job_smoke.slurm
#        sbatch --array=0-0 nav_job_smoke.slurm  # Single test
#
# Output: results/smoke_nav_results_XXXX.mat (where XXXX is array task ID)

#SBATCH --job-name=nav_smoke
#SBATCH --partition=day
#SBATCH --time=2:00:00    # Reduced from 6h - smoke sims are faster
#SBATCH --mem=16G         # Reduced from 82G - compressed file
#SBATCH --cpus-per-task=1
#SBATCH --array=1000-1000%1  # Quick test: only one job
#SBATCH --output=logs/smoke-%A_%a.out
#SBATCH --error=logs/smoke-%A_%a.err

# Ensure no display
unset DISPLAY
export MATLAB_JAVA=$JAVA_HOME/jre

# Load MATLAB module (required on Grace)
module load MATLAB/2023b

# Get project directory using symlink-safe path
PROJECT_DIR="/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations"
cd "$PROJECT_DIR" || exit 1
mkdir -p logs results

# Log info
echo "======================================"
echo "Smoke Plume Navigation Simulation"
echo "======================================"
echo "Job started at: $(date)"
echo "Running on: $(hostname)"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Working directory: $(pwd)"
echo ""

# Log which config will be used and show its contents
SMOKE_CONFIG_PATH="$PROJECT_DIR/configs/plumes/smoke_1a_backgroundsubtracted.json"
echo "[INFO] Using smoke plume config: $SMOKE_CONFIG_PATH"
echo "[INFO] Contents of config file (first 10 lines):"
head -n 10 "$SMOKE_CONFIG_PATH" || echo "[WARN] Could not read config file"
echo "[INFO] Contents of paths.json before update (first 10 lines):"
head -n 10 "$PROJECT_DIR/configs/paths.json" || echo "[WARN] Could not read paths.json"

# Create MATLAB script in temp file
TEMP_MATLAB=$(mktemp /tmp/smoke_nav_XXXXXX.m)
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
    fprintf('[MATLAB] Configuring for smoke plume...\n');
    smoke_config_path = fullfile(project_dir, 'configs/plumes/smoke_1a_backgroundsubtracted.json');
    paths_file = fullfile(project_dir, 'configs/paths.json');
    
    % Log current plume_config in paths.json before update
    try
        paths_data_pre = jsondecode(fileread(paths_file));
        fprintf('[MATLAB] paths.json before update: plume_config = %s\n', paths_data_pre.plume_config);
    catch ME
        fprintf('[MATLAB][WARN] Could not read paths.json before update: %s\n', ME.message);
    end
    
    % Update paths to use smoke
    paths_data = jsondecode(fileread(paths_file));
    paths_data.plume_config = smoke_config_path;
    fid = fopen(paths_file, 'w');
    fprintf(fid, '%s', jsonencode(paths_data));
    fclose(fid);
    
    % Log contents of paths.json after update
    try
        paths_data_post = jsondecode(fileread(paths_file));
        fprintf('[MATLAB] paths.json after update: plume_config = %s\n', paths_data_post.plume_config);
    catch ME
        fprintf('[MATLAB][WARN] Could not read paths.json after update: %s\n', ME.message);
    end
    
    % Verify configuration
    which get_plume_file
    [plume_file, plume_config] = get_plume_file();
    disp('DEBUG: plume_config after get_plume_file:');
    disp(plume_config);
    disp('DEBUG: Top-level fields after get_plume_file:');
    disp(fieldnames(plume_config));
    fprintf('[MATLAB] Loaded plume_config: %s\n', smoke_config_path);
    fprintf('[MATLAB] Loaded HDF5 file: %s\n', plume_file);
    disp('DEBUG: plume_config struct just before accessing .temporal:');
    disp(plume_config);
    disp('DEBUG: Top-level fields in plume_config:');
    disp(fieldnames(plume_config));
    fprintf('[MATLAB] Frame rate: %.1f Hz\n', plume_config.temporal.frame_rate);
    fprintf('[MATLAB] Pixel scale: %.3f mm/px\n', plume_config.spatial.mm_per_pixel);
    
    % Check if plume file exists and log file size
    if ~exist(plume_file, 'file')
        error('[MATLAB] Plume file not found: %s', plume_file);
    else
        file_info = dir(plume_file);
        fprintf('[MATLAB] Plume file exists. Size: %.2f MB\n', file_info.bytes/1e6);
    end
    
    % QUICK TEST: reduce simulation duration and agents
    n_agents = 2;
    duration_seconds = 5;  % 5 seconds for fast test
    n_frames = round(duration_seconds * 60);
    fprintf('[MATLAB][QUICK TEST] Running %d agents for %.1f seconds (%d frames at 60 Hz)\n', n_agents, duration_seconds, n_frames);
    
    % Run simulation (note: we use 'Crimaldi' as environment type, which loads from config)
    fprintf('\nStarting simulation for task %d...\n', task_id);
    tic;
    
    % Use the correct environment type for smoke plume simulation
    out = navigation_model_vec(n_frames, 'Smoke', 0, 10);
    
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

# Run MATLAB with the temp script (guaranteed non-interactive)
matlab -nodisplay -nosplash -nodesktop -nojvm -batch "run('$TEMP_MATLAB')"
MATLAB_EXIT=$?

# Clean up
rm -f "$TEMP_MATLAB"

echo ""
echo "Job completed at: $(date)"
echo "Exit code: $MATLAB_EXIT"

exit $MATLAB_EXIT