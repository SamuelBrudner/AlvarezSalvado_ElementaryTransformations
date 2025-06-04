#!/bin/bash
# fix_all_nav_issues.sh - Fix all navigation job issues

echo "=== Comprehensive Navigation Job Fix ==="
echo ""

# Step 1: Make get_plume_file quiet
echo "Step 1: Quieting verbose output..."
if grep -q "fprintf('Loading plume config" Code/get_plume_file.m; then
    sed -i.backup_verbose "s/fprintf('Loading plume config from %s/% fprintf('Loading plume config from %s/" Code/get_plume_file.m
    sed -i "s/fprintf('Using plume file: %s/% fprintf('Using plume file: %s/" Code/get_plume_file.m
    echo "✓ Made get_plume_file.m quieter"
else
    echo "✓ Already quiet"
fi

# Step 2: Fix the SLURM script
echo ""
echo "Step 2: Creating working SLURM script..."

cat > nav_job_working.slurm << 'EOF'
#!/bin/bash
#SBATCH --job-name=nav_model
#SBATCH --partition=day
#SBATCH --time=6:00:00
#SBATCH --mem=82G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/nav-%A_%a.out
#SBATCH --error=logs/nav-%A_%a.err

# Note: Array specification will be added at submission time

# Setup
cd /home/snb6/Documents/AlvarezSalvado_ElementaryTransformations
mkdir -p logs results

# Load MATLAB
module load MATLAB/2023b

# Get array task ID
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

# Create MATLAB script for this task
MATLAB_SCRIPT="/tmp/nav_task_${SLURM_JOB_ID}_${TASK_ID}.m"

cat > "$MATLAB_SCRIPT" << 'MATLAB_EOF'
% Navigation model task
fprintf('\n=== Navigation Model Task %d ===\n', $TASK_ID);

% Setup paths
cd('/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations');
addpath(genpath('Code'));

% Get task ID
task_id = $TASK_ID;

try
    % Load config to check duration
    [~, pc] = get_plume_file();
    if isfield(pc, 'simulation') && isfield(pc.simulation, 'duration_seconds')
        duration_seconds = pc.simulation.duration_seconds;
        n_samples = round(duration_seconds * 15);  % 15 Hz for Crimaldi
        fprintf('Using duration from config: %.0f seconds = %d samples\n', duration_seconds, n_samples);
    else
        n_samples = 4500;  % Default 300s at 15Hz
        fprintf('Using default duration: 300 seconds = 4500 samples\n');
    end
    
    % Run simulation with explicit sample count
    fprintf('Starting simulation...\n');
    tic;
    out = navigation_model_vec(n_samples, 'Crimaldi', 0, 10);
    elapsed = toc;
    
    % Save results
    filename = sprintf('results/nav_results_%04d.mat', task_id);
    save(filename, 'out', '-v7.3');
    
    % Report success
    fprintf('\n✓ Task %d complete!\n', task_id);
    fprintf('  Simulated: %d samples = %.1f seconds\n', size(out.x,1), size(out.x,1)/15);
    fprintf('  Computation time: %.1f seconds\n', elapsed);
    fprintf('  Saved to: %s\n', filename);
    fprintf('  File size: %.1f MB\n', dir(filename).bytes/1024/1024);
    
catch ME
    fprintf('\n✗ Task %d FAILED!\n', task_id);
    fprintf('Error: %s\n', ME.message);
    fprintf('Stack trace:\n');
    for i = 1:length(ME.stack)
        fprintf('  %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
    exit(1);
end

exit(0);
MATLAB_EOF

# Make script executable and run MATLAB
chmod +x "$MATLAB_SCRIPT"
matlab -nodisplay -nosplash < "$MATLAB_SCRIPT"

# Clean up
rm -f "$MATLAB_SCRIPT"
EOF

echo "✓ Created nav_job_working.slurm"

# Step 3: Create test script
echo ""
echo "Step 3: Creating test script..."

cat > test_nav_working.sh << 'EOF'
#!/bin/bash
# Test single navigation job

echo "Testing navigation job locally..."

# Simulate SLURM environment
export SLURM_ARRAY_TASK_ID=999
export SLURM_JOB_ID=99999

# Run the job script
bash nav_job_working.slurm
EOF

chmod +x test_nav_working.sh

# Step 4: Show summary
echo ""
echo "=== Fix Complete ==="
echo ""
echo "What was fixed:"
echo "1. ✓ Quieted verbose plume loading messages"
echo "2. ✓ Fixed SLURM script (no JSON sourcing, proper MATLAB execution)"
echo "3. ✓ Made it read duration from config (300s) but fall back to hardcoded if needed"
echo ""
echo "To test locally:"
echo "  ./test_nav_working.sh"
echo ""
echo "To submit array job:"
echo "  sbatch --array=0-99%50 nav_job_working.slurm"
echo ""
echo "To submit single test job:"
echo "  sbatch --array=0-0 nav_job_working.slurm"