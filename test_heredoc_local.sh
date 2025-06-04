#!/bin/bash
# test_heredoc_local.sh - Test the here-doc approach locally before SLURM submission

echo "=== Testing Here-Doc MATLAB Execution ==="
echo ""

# Set variables as if we're in SLURM
PROJECT_ROOT="/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations"
TASK_ID=999  # Test task ID

echo "Project root: $PROJECT_ROOT"
echo "Test task ID: $TASK_ID"
echo ""

# Create temporary MATLAB script
MATLAB_SCRIPT=$(mktemp /tmp/test_heredoc_XXXXXX.m)
echo "Creating temporary script: $MATLAB_SCRIPT"

# Write MATLAB code using here-doc
cat > "$MATLAB_SCRIPT" << 'MATLAB_EOF'
% Test script for here-doc approach
fprintf('\n=== Here-Doc Test for Task %d ===\n', $TASK_ID);

% Setup paths
cd('$PROJECT_ROOT');
addpath(genpath('Code'));

% Quick tests
fprintf('\n1. Path test:\n');
fprintf('   Current directory: %s\n', pwd);
fprintf('   Code in path: %s\n', iif(exist('navigation_model_vec', 'file')==2, 'YES', 'NO'));

fprintf('\n2. Config test:\n');
try
    [plume_file, plume_config] = get_plume_file();
    fprintf('   ✓ Config loaded\n');
    fprintf('   Plume file: %s\n', plume_file);
    
    if isfield(plume_config, 'simulation') && isfield(plume_config.simulation, 'duration_seconds')
        fprintf('   ✓ Duration: %.0f seconds\n', plume_config.simulation.duration_seconds);
        n_samples = round(plume_config.simulation.duration_seconds * 15);
        fprintf('   ✓ Samples: %d (at 15 Hz)\n', n_samples);
    else
        fprintf('   ✗ No duration in config\n');
    end
catch ME
    fprintf('   ✗ Config error: %s\n', ME.message);
end

fprintf('\n3. Quick simulation test:\n');
try
    tic;
    out = navigation_model_vec(150, 'Crimaldi', 0, 1);  % 10 seconds, 1 agent
    elapsed = toc;
    fprintf('   ✓ Test simulation successful!\n');
    fprintf('   Generated %d samples in %.1f seconds\n', size(out.x,1), elapsed);
catch ME
    fprintf('   ✗ Simulation failed: %s\n', ME.message);
end

fprintf('\n✓ Here-doc test complete for task %d\n', $TASK_ID);

% Helper function
function r = iif(c, t, f)
    if c, r = t; else, r = f; end
end

exit(0);
MATLAB_EOF

# Substitute variables
echo ""
echo "Substituting variables..."
sed -i "s/\$TASK_ID/$TASK_ID/g" "$MATLAB_SCRIPT"
sed -i "s|\$PROJECT_ROOT|$PROJECT_ROOT|g" "$MATLAB_SCRIPT"

echo "Running MATLAB with here-doc script..."
echo ""

# Run MATLAB
matlab -nodisplay -nosplash < "$MATLAB_SCRIPT" 2>&1 | grep -v "^>>" | grep -v "^$"

# Clean up
rm -f "$MATLAB_SCRIPT"

echo ""
echo "=== Test Complete ==="
echo ""
echo "If the test was successful, you can now run:"
echo "  ./update_to_heredoc.sh     # Update the SLURM script"
echo "  sbatch --array=0-0 nav_job_paths.slurm  # Test single job"