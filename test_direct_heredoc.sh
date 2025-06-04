#!/bin/bash
# test_direct_heredoc.sh - Test direct here-doc approach before SLURM submission

echo "=== Testing Direct Here-Doc Approach ==="
echo ""

# Set test variables
PROJECT_ROOT="/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations"
TASK_ID=999

echo "Testing with:"
echo "  PROJECT_ROOT: $PROJECT_ROOT"
echo "  TASK_ID: $TASK_ID"
echo ""

# Test the direct here-doc approach
matlab -nodisplay -nosplash -nodesktop << MATLAB_SCRIPT
% Quick test of direct here-doc
fprintf('\n=== Direct Here-Doc Test ===\n');
fprintf('Task ID: $TASK_ID\n');
fprintf('Project: $PROJECT_ROOT\n');

% Change directory and add paths
cd('$PROJECT_ROOT');
addpath(genpath('Code'));

% Quick functionality test
fprintf('\nChecking environment:\n');
fprintf('  Current dir: %s\n', pwd);
fprintf('  nav function exists: %s\n', exist('navigation_model_vec', 'file') == 2 ? 'YES' : 'NO');

% Test config loading
try
    [pf, pc] = get_plume_file();
    fprintf('  Config loaded: YES\n');
    if isfield(pc, 'simulation') && isfield(pc.simulation, 'duration_seconds')
        fprintf('  Duration: %.0f seconds\n', pc.simulation.duration_seconds);
    end
catch
    fprintf('  Config loaded: NO\n');
end

% Quick 1-second simulation test
fprintf('\nRunning 1-second test simulation...\n');
try
    tic;
    out = navigation_model_vec(15, 'Crimaldi', 0, 1);  % 1 second, 1 agent
    elapsed = toc;
    fprintf('✓ Success! Generated %d samples in %.2f seconds\n', size(out.x,1), elapsed);
catch ME
    fprintf('✗ Failed: %s\n', ME.message);
end

fprintf('\n✓ Direct here-doc test complete!\n');
exit(0);
MATLAB_SCRIPT

echo ""
echo "Test complete. If successful, the SLURM script should work!"