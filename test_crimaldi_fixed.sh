#!/bin/bash
# test_crimaldi_fixed.sh - Test with correct crimaldi file
set -euo pipefail

# First, create the symlink if it doesn't exist
if [ ! -f "data/10302017_10cms_bounded.hdf5" ]; then
    echo "Creating symlink for Crimaldi data..."
    ln -s 10302017_10cms_bounded_2.h5 data/10302017_10cms_bounded.hdf5
fi

# Create test directory
TEST_DIR="test_output/test_crimaldi_fixed_$(date +%s)"
mkdir -p "$TEST_DIR"

# Create test MATLAB script
MATLAB_SCRIPT=$(mktemp test_crim_fixed_XXXX.m)
cat > "$MATLAB_SCRIPT" << EOF
addpath('Code');

% Create config for crimaldi environment
cfg = struct();
cfg.environment = 'crimaldi';
cfg.triallength = 100;
cfg.plotting = 0;
cfg.ntrials = 1;
cfg.outputDir = '$TEST_DIR';

fprintf('Running with outputDir: %s\n', cfg.outputDir);

try
    % Run the navigation
    fprintf('Calling run_navigation_cfg...\n');
    tic;
    R = run_navigation_cfg(cfg);
    elapsed = toc;
    fprintf('run_navigation_cfg completed in %.3f seconds\n', elapsed);
    
    % Check what we got back
    if isempty(R)
        error('run_navigation_cfg returned empty result');
    elseif ~isstruct(R)
        error('run_navigation_cfg did not return a struct');
    else
        fprintf('Got structure with fields:\n');
        disp(fieldnames(R));
    end
    
    % Save the result
    resultPath = fullfile(cfg.outputDir, 'result.mat');
    fprintf('\nSaving to: %s\n', resultPath);
    
    save(resultPath, '-struct', 'R', '-v7');
    
    % Verify
    if exist(resultPath, 'file')
        fprintf('SUCCESS: File saved\n');
        d = dir(resultPath);
        fprintf('File size: %d bytes\n', d.bytes);
    else
        fprintf('ERROR: File not found after save!\n');
    end
    
catch ME
    fprintf('ERROR: %s\n', ME.message);
    for i = 1:length(ME.stack)
        fprintf('  at %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
end

% List directory contents
fprintf('\nDirectory contents:\n');
system(['ls -la ' cfg.outputDir]);

exit;
EOF

# Run it
echo "Running fixed crimaldi test..."
matlab -nodisplay -nosplash -r "run('$MATLAB_SCRIPT')"

# Check results
echo -e "\n=== Checking results ==="
echo "Directory contents of $TEST_DIR:"
ls -la "$TEST_DIR" 2>/dev/null || echo "Directory not found"

# Cleanup
rm -f "$MATLAB_SCRIPT"