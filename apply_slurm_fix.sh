#!/bin/bash
# apply_slurm_fix.sh - Fix the nav_job_paths.slurm script

echo "=== Fixing SLURM Script ==="
echo ""

# Check current issue
echo "1. Current problem in nav_job_paths.slurm:"
grep -n "source.*json" nav_job_paths.slurm 2>/dev/null && echo "   ✗ Found problematic 'source' command for JSON file" || echo "   ✓ No JSON sourcing found"

# Backup current script
if [ -f "nav_job_paths.slurm" ]; then
    cp nav_job_paths.slurm nav_job_paths.slurm.broken_$(date +%Y%m%d_%H%M%S)
    echo ""
    echo "2. Backed up broken script"
fi

# Create fixed version
echo ""
echo "3. Creating fixed SLURM script..."

cat > nav_job_paths_fixed.slurm << 'EOF'
#!/bin/bash
#SBATCH --job-name=nav_model
#SBATCH --partition=day
#SBATCH --time=6:00:00
#SBATCH --mem=82G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-99%20
#SBATCH --output=logs/nav-%A_%a.out
#SBATCH --error=logs/nav-%A_%a.err

# Set project directory (DO NOT source JSON files!)
PROJECT_ROOT="/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations"

# Change to project directory
cd "$PROJECT_ROOT"

# Create directories
mkdir -p logs results

# Load MATLAB
module load MATLAB/2023b

# Get array task ID
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

# Run simulation with complete MATLAB command
matlab -nodisplay -nosplash -r "
% Add Code to path
addpath(genpath('Code'));

% Load paths configuration
try
    paths = load_paths();
    fprintf('Using paths from: %s\\n', paths.project_root);
catch ME
    fprintf('Warning: Could not load paths: %s\\n', ME.message);
end

% Get task ID
task_id = $TASK_ID;
fprintf('\\n=== Task %d: Starting simulation ===\\n', task_id);

try
    % Load config to check duration
    [plume_file, plume_config] = get_plume_file();
    
    % Check if config has duration
    if isfield(plume_config, 'simulation') && isfield(plume_config.simulation, 'duration_seconds')
        duration_seconds = plume_config.simulation.duration_seconds;
        fprintf('Using duration from config: %.0f seconds\\n', duration_seconds);
    else
        duration_seconds = 300;  % Default 5 minutes
        fprintf('Using default duration: %.0f seconds\\n', duration_seconds);
    end
    
    % Calculate samples based on environment (Crimaldi runs at 15 Hz)
    n_samples = round(duration_seconds * 15);
    fprintf('Running simulation for %d samples\\n', n_samples);
    
    % Run simulation
    fprintf('Starting navigation_model_vec...\\n');
    tic;
    out = navigation_model_vec(n_samples, 'Crimaldi', 0, 10);
    elapsed = toc;
    
    % Save results
    filename = sprintf('results/nav_results_%04d.mat', task_id);
    save(filename, 'out', '-v7.3');
    
    % Report success
    fprintf('\\n✓ Task %d COMPLETE!\\n', task_id);
    fprintf('  Simulated: %d samples = %.1f seconds\\n', size(out.x,1), size(out.x,1)/15);
    fprintf('  Computation time: %.1f seconds\\n', elapsed);
    fprintf('  Saved to: %s\\n', filename);
    
    % Check file size if it exists
    if exist(filename, 'file')
        d = dir(filename);
        fprintf('  File size: %.1f MB\\n', d.bytes/1024/1024);
    end
    
catch ME
    fprintf('\\n✗ Task %d FAILED!\\n', task_id);
    fprintf('Error: %s\\n', ME.message);
    fprintf('Stack trace:\\n');
    for i = 1:length(ME.stack)
        fprintf('  In %s (line %d)\\n', ME.stack(i).name, ME.stack(i).line);
    end
    exit(1);
end

% Exit successfully
fprintf('\\nTask %d exiting normally\\n', task_id);
exit(0);
"
EOF

# Replace the broken script
mv nav_job_paths_fixed.slurm nav_job_paths.slurm
chmod +x nav_job_paths.slurm

echo "✓ Fixed SLURM script created"

echo ""
echo "4. Verifying the fix:"
echo "   Checking for JSON sourcing in new script..."
grep -n "source.*json" nav_job_paths.slurm 2>/dev/null && echo "   ✗ Still has JSON sourcing!" || echo "   ✓ No JSON sourcing - GOOD!"

echo ""
echo "5. Key changes made:"
echo "   - Removed 'source configs/paths.json' line (JSON files can't be sourced!)"
echo "   - Added explicit duration handling from config (300 seconds)"
echo "   - Added proper error handling and progress reporting"
echo "   - Made MATLAB code more verbose for debugging"

echo ""
echo "=== Ready to Test ==="
echo ""
echo "To test a single job:"
echo "  sbatch --array=0-0 nav_job_paths.slurm"
echo ""
echo "To submit full array:"
echo "  sbatch nav_job_paths.slurm"
echo ""
echo "Monitor with:"
echo "  squeue -u $USER"
echo "  tail -f logs/nav-*_0.out"