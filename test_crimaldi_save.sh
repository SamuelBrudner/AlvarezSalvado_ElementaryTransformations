#!/bin/bash
# test_crimaldi_save.sh - Test with crimaldi environment
set -euo pipefail

# Create test directory
TEST_DIR="test_output/test_crimaldi_$(date +%s)"
mkdir -p "$TEST_DIR"

# Create test MATLAB script
MATLAB_SCRIPT=$(mktemp test_crim_XXXX.m)
cat > "$MATLAB_SCRIPT" << EOF
addpath('Code');

% Create a simple config for crimaldi environment
cfg = struct();
cfg.environment = 'crimaldi';
cfg.triallength = 100;
cfg.plotting = 0;
cfg.ntrials = 1;
cfg.outputDir = '$TEST_DIR';

fprintf('Running with outputDir: %s\n', cfg.outputDir);

try
    % Run the navigation
    R = run_navigation_cfg(cfg);
    
    % Check what we got back
    fprintf('run_navigation_cfg returned:\n');
    if isempty(R)
        fprintf('  EMPTY result!\n');
    elseif ~isstruct(R)
        fprintf('  NOT A STRUCT! Class: %s\n', class(R));
    else
        fprintf('  Structure with fields:\n');
        fields = fieldnames(R);
        for i = 1:length(fields)
            fprintf('    %s\n', fields{i});
        end
    end
    
    % Try to save it
    if ~isempty(R) && isstruct(R)
        resultPath = fullfile(cfg.outputDir, 'result.mat');
        fprintf('\nSaving to: %s\n', resultPath);
        save(resultPath, '-struct', 'R', '-v7');
        
        if exist(resultPath, 'file')
            fprintf('SUCCESS: File created\n');
            d = dir(resultPath);
            fprintf('File size: %d bytes\n', d.bytes);
        else
            fprintf('FAILED: File not created\n');
        end
    end
    
catch ME
    fprintf('ERROR: %s\n', ME.message);
end

% List what's in the directory
fprintf('\nDirectory contents:\n');
system(['ls -la ' cfg.outputDir]);

exit;
EOF

# Run it
echo "Running crimaldi test..."
matlab -nodisplay -nosplash -r "run('$MATLAB_SCRIPT')"

# Check results
echo -e "\n=== Checking results ==="
echo "Directory contents:"
ls -la "$TEST_DIR" 2>/dev/null || echo "Directory not found"

# Cleanup
rm -f "$MATLAB_SCRIPT"c