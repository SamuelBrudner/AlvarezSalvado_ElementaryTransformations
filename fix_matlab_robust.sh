#!/bin/bash
# fix_matlab_robust.sh - Create a robust SLURM script that properly uses here-docs

echo "=== Creating Robust MATLAB Here-Doc Solution ==="
echo ""

# Create the SLURM script
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

# Create and run MATLAB script using here-doc piped to MATLAB
# This approach avoids temporary files and runs cleanly
matlab -nodisplay -nosplash -nodesktop << MATLAB_SCRIPT
% Navigation model task script
fprintf('\n=== Navigation Model Task $TASK_ID ===\n');

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
    
    fprintf('\nTask %d completed - exiting MATLAB\n', task_id);
    
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
exit(0);
MATLAB_SCRIPT
EOF

chmod +x nav_job_paths.slurm

echo "✓ Created robust SLURM script using direct here-doc"
echo ""
echo "This approach:"
echo "  - Uses here-doc piped directly to MATLAB (no temp files)"
echo "  - Variables are substituted by bash before MATLAB sees them"
echo "  - No issues with function definitions"
echo "  - Clean exit handling"
echo ""
echo "Test with a single job:"
echo "  sbatch --array=0-0 nav_job_paths.slurm"
echo ""
echo "Check output:"
echo "  tail -f logs/nav-*_0.out"