#!/bin/bash
# update_to_heredoc.sh - Update nav_job_paths.slurm to use here-docs

echo "=== Updating SLURM Script to Use Here-Docs ==="
echo ""

# Backup current version
if [ -f "nav_job_paths.slurm" ]; then
    cp nav_job_paths.slurm nav_job_paths.slurm.backup_$(date +%Y%m%d_%H%M%S)
    echo "✓ Backed up current script"
fi

# Create the new version with here-docs
cat > nav_job_paths.slurm << 'EOF'
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

# Create temporary MATLAB script using here-doc
MATLAB_SCRIPT=$(mktemp /tmp/nav_task_${SLURM_JOB_ID}_${TASK_ID}_XXXXXX.m)

cat > "$MATLAB_SCRIPT" << 'MATLAB_EOF'
% Navigation model task script
fprintf('\n=== Navigation Model Task %d ===\n', $TASK_ID);

% Setup paths
cd('$PROJECT_ROOT');
addpath(genpath('Code'));

% Load paths configuration
try
    paths = load_paths();
    fprintf('Using paths from: %s\n', paths.project_root);
catch ME
    fprintf('Warning: Could not load paths: %s\n', ME.message);
end

% Get task ID
task_id = $TASK_ID;

try
    % Load config to check duration
    fprintf('\nLoading plume configuration...\n');
    [plume_file, plume_config] = get_plume_file();
    
    % Check if config has duration
    if isfield(plume_config, 'simulation') && isfield(plume_config.simulation, 'duration_seconds')
        duration_seconds = plume_config.simulation.duration_seconds;
        fprintf('Using duration from config: %.0f seconds\n', duration_seconds);
    else
        duration_seconds = 300;  % Default 5 minutes
        fprintf('Using default duration: %.0f seconds\n', duration_seconds);
    end
    
    % Calculate samples based on environment (Crimaldi runs at 15 Hz)
    n_samples = round(duration_seconds * 15);
    fprintf('Running simulation for %d samples at 15 Hz\n', n_samples);
    
    % Run simulation
    fprintf('\nStarting navigation_model_vec...\n');
    tic;
    out = navigation_model_vec(n_samples, 'Crimaldi', 0, 10);
    elapsed = toc;
    
    % Save results
    filename = sprintf('results/nav_results_%04d.mat', task_id);
    save(filename, 'out', '-v7.3');
    
    % Report success
    fprintf('\n✓ Task %d COMPLETE!\n', task_id);
    fprintf('  Simulated: %d samples = %.1f seconds\n', size(out.x,1), size(out.x,1)/15);
    fprintf('  Computation time: %.1f seconds\n', elapsed);
    fprintf('  Saved to: %s\n', filename);
    
    % Check file size
    if exist(filename, 'file')
        d = dir(filename);
        fprintf('  File size: %.1f MB\n', d.bytes/1024/1024);
    end
    
    % Success metrics
    if isfield(out, 'successrate')
        fprintf('  Success rate: %.1f%%\n', out.successrate * 100);
    end
    
catch ME
    fprintf('\n✗ Task %d FAILED!\n', task_id);
    fprintf('Error: %s\n', ME.message);
    fprintf('Stack trace:\n');
    for i = 1:length(ME.stack)
        fprintf('  In %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
    exit(1);
end

% Exit successfully
fprintf('\nTask %d completed successfully\n', task_id);
exit(0);
MATLAB_EOF

# Substitute variables in the MATLAB script
sed -i "s/\$TASK_ID/$TASK_ID/g" "$MATLAB_SCRIPT"
sed -i "s|\$PROJECT_ROOT|$PROJECT_ROOT|g" "$MATLAB_SCRIPT"

# Run MATLAB with the script
matlab -nodisplay -nosplash < "$MATLAB_SCRIPT"

# Clean up
rm -f "$MATLAB_SCRIPT"
EOF

chmod +x nav_job_paths.slurm

echo ""
echo "✓ Updated nav_job_paths.slurm to use here-docs"
echo ""
echo "Key improvements:"
echo "  - Uses here-doc for clean MATLAB code"
echo "  - Creates temporary script file"
echo "  - No escaping issues"
echo "  - Easy to read and modify"
echo "  - Proper variable substitution"
echo ""
echo "To test:"
echo "  sbatch --array=0-0 nav_job_paths.slurm"
echo ""
echo "To run full array:"
echo "  sbatch nav_job_paths.slurm"