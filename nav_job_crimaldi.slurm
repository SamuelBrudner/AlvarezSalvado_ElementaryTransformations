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
#SBATCH --output=logs/crimaldi/nav_crim_%A_%a.out
#SBATCH --error=logs/crimaldi/nav_crim_%A_%a.err

# Ensure no display
unset DISPLAY
export MATLAB_JAVA=$JAVA_HOME/jre

# Load MATLAB module (required on Grace)
module load MATLAB/2023b

# Get project directory using symlink-safe path
export PROJECT_DIR="/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations"
cd "$PROJECT_DIR" || exit 1
mkdir -p logs results
echo "[INFO] Set PROJECT_DIR=$PROJECT_DIR"

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

# Log which config will be used and show its contents
CRIM_CONFIG_PATH="$PROJECT_DIR/configs/plumes/crimaldi_10cms_bounded.json"
echo "[INFO] Using Crimaldi plume config: $CRIM_CONFIG_PATH"
echo "[INFO] Contents of config file (first 10 lines):"
head -n 10 "$CRIM_CONFIG_PATH" || echo "[WARN] Could not read config file"
echo "[INFO] Contents of paths.json before update (first 10 lines):"
head -n 10 "$PROJECT_DIR/configs/paths.json" || echo "[WARN] Could not read paths.json"

# Create MATLAB script in temp file
TEMP_MATLAB=$(mktemp /tmp/crimaldi_nav_XXXXXX.m)
cat > "$TEMP_MATLAB" << MATLAB_EOF
% Crimaldi plume navigation simulation
try
    % Get project directory and task ID from environment
    project_dir = '$PROJECT_DIR';
    task_id = str2double(getenv('SLURM_ARRAY_TASK_ID'));
    if isnan(task_id), task_id = 0; end
    
    fprintf('Using project directory: %s\n', project_dir);
    
    % Change to project directory
    cd(project_dir);
    addpath(genpath('Code'));
    
    % Force Crimaldi plume configuration
    fprintf('[MATLAB] Configuring for Crimaldi plume...\n');
    crim_config_path = fullfile(project_dir, 'configs/plumes/crimaldi_10cms_bounded.json');
    paths_file = fullfile(project_dir, 'configs/paths.json');
    
    % Log current plume_config in paths.json before update
    try
        paths_data_pre = jsondecode(fileread(paths_file));
        fprintf('[MATLAB] paths.json before update: plume_config = %s\n', paths_data_pre.plume_config);
    catch ME
        fprintf('[MATLAB][WARN] Could not read paths.json before update: %s\n', ME.message);
    end
    
    % Update paths to use Crimaldi
    paths_data = jsondecode(fileread(paths_file));
    paths_data.plume_config = crim_config_path;
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
    
    % Print config path and raw JSON to help with debugging
    fprintf('DEBUG: Config path:\n%s\n', crim_config_path);
    fid = fopen(crim_config_path, 'r');
    if fid ~= -1
        raw_json = fread(fid, '*char')';
        fclose(fid);
        fprintf('DEBUG: Raw JSON:\n%s\n', raw_json);
    end
    
    % Load plume config and show details
    [plume_file, plume_config] = get_plume_file();
    disp('DEBUG: Loaded config struct:');
    disp(jsondecode(raw_json));
    
    disp('DEBUG: Top-level fields:');
    disp(fieldnames(jsondecode(raw_json)));

    % Print contents of plume_config after loading
    disp('DEBUG: plume_config after get_plume_file:');
    disp(plume_config);
    disp('DEBUG: Top-level fields after get_plume_file:');
    disp(fieldnames(plume_config));
    
    fprintf('[MATLAB] Loaded plume_config: %s\n', crim_config_path);
    fprintf('[MATLAB] Loaded HDF5 file: %s\n', plume_file);
    
    % Print plume_config before accessing temporal field
    disp('DEBUG: plume_config struct just before accessing .temporal:');
    disp(plume_config);
    disp('DEBUG: Top-level fields in plume_config:');
    disp(fieldnames(plume_config));
    
    % Access and print key parameters
    fprintf('[MATLAB] Frame rate: %.1f Hz\n', plume_config.temporal.frame_rate);
    fprintf('[MATLAB] Pixel scale: %.3f mm/px\n', plume_config.spatial.mm_per_pixel);
    
    % Check if plume file exists and log file size
    if ~exist(plume_file, 'file')
        error('[MATLAB] Plume file not found: %s', plume_file);
    else
        file_info = dir(plume_file);
        fprintf('[MATLAB] Plume file exists. Size: %.2f MB\n', file_info.bytes/1e6);
    end
    
    % Determine simulation parameters from config
    n_agents = plume_config.simulation.agent_initialization.n_agents_per_job;
    duration_seconds = plume_config.simulation.duration_seconds;
    n_frames = round(duration_seconds * plume_config.temporal.frame_rate);
    fprintf('[MATLAB] Running %d agents for %.1f seconds (%d frames at %.1f Hz)\n', ...
        n_agents, duration_seconds, n_frames, plume_config.temporal.frame_rate);
    
    % Run simulation
    fprintf('\nStarting simulation for task %d...\n', task_id);
    tic;
    
    out = navigation_model_vec(n_frames, 'Crimaldi', 0, n_agents);
    
    elapsed = toc;
    fprintf('Simulation completed in %.1f seconds\n', elapsed);
    
    % Save with standard prefix
    filename = sprintf('results/nav_results_%04d.mat', task_id);
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

# Save a copy of the MATLAB script for debugging
echo "[INFO] MATLAB script content saved to logs/crimaldi_matlab_${SLURM_ARRAY_TASK_ID}.m"
cp "$TEMP_MATLAB" "${PROJECT_DIR}/logs/crimaldi_matlab_${SLURM_ARRAY_TASK_ID}.m" || echo "[WARN] Could not save MATLAB script copy"

# Export environment variables to ensure they're available to MATLAB
echo "[INFO] Exporting environment variables for MATLAB"
export PROJECT_DIR
export SLURM_ARRAY_TASK_ID
export PLUME

# Execute the temp MATLAB script (output will go directly to SLURM log)
echo "[INFO] Running MATLAB script: $TEMP_MATLAB"
echo "[INFO] MATLAB output will appear below"
echo "===== MATLAB EXECUTION START ====="
matlab -nodisplay -nosplash -nodesktop -batch "run('$TEMP_MATLAB');"
MATLAB_EXIT=$?
echo "===== MATLAB EXECUTION COMPLETE ====="

echo ""
echo "Job completed at: $(date)"
echo "MATLAB exit code: $MATLAB_EXIT"

exit $MATLAB_EXIT