#!/bin/bash
# test_matlab_save.sh - Diagnostic script to test MATLAB saving
set -euo pipefail

echo "=== MATLAB Save Test ==="

# Create a test MATLAB script
MATLAB_SCRIPT=$(mktemp test_save_XXXX.m)
cat > "$MATLAB_SCRIPT" << 'EOF'
fprintf('Current working directory: %s\n', pwd);
fprintf('MATLAB version: %s\n', version);

% Test 1: Simple save in current directory
testData.x = 1:10;
testData.y = rand(1,10);
save('test_simple.mat', 'testData');
if exist('test_simple.mat', 'file')
    fprintf('SUCCESS: test_simple.mat created in current directory\n');
else
    fprintf('FAILED: test_simple.mat not created\n');
end

% Test 2: Save with relative path
testDir = 'test_output/test_save';
if ~exist(testDir, 'dir')
    mkdir(testDir);
end
save(fullfile(testDir, 'test_relative.mat'), 'testData');
if exist(fullfile(testDir, 'test_relative.mat'), 'file')
    fprintf('SUCCESS: test_relative.mat created\n');
else
    fprintf('FAILED: test_relative.mat not created\n');
end

% Test 3: Save with absolute path
absDir = fullfile(pwd, 'test_output', 'test_absolute');
if ~exist(absDir, 'dir')
    mkdir(absDir);
end
save(fullfile(absDir, 'test_absolute.mat'), 'testData');
if exist(fullfile(absDir, 'test_absolute.mat'), 'file')
    fprintf('SUCCESS: test_absolute.mat created at %s\n', absDir);
else
    fprintf('FAILED: test_absolute.mat not created\n');
end

% Test 4: Run the actual navigation function
addpath('Code');
cfg = struct();
cfg.environment = 'gaussian';
cfg.triallength = 100;
cfg.plotting = 0;
cfg.ntrials = 1;
cfg.outputDir = fullfile(pwd, 'test_output', 'test_nav');
if ~exist(cfg.outputDir, 'dir')
    mkdir(cfg.outputDir);
end

fprintf('\nTesting run_navigation_cfg...\n');
try
    result = run_navigation_cfg(cfg);
    fprintf('run_navigation_cfg returned a structure with fields:\n');
    disp(fieldnames(result));
    
    % Try to save it
    save(fullfile(cfg.outputDir, 'result.mat'), '-struct', 'result');
    if exist(fullfile(cfg.outputDir, 'result.mat'), 'file')
        fprintf('SUCCESS: Navigation result saved\n');
    else
        fprintf('FAILED: Navigation result not saved\n');
    end
catch ME
    fprintf('ERROR in run_navigation_cfg: %s\n', ME.message);
end

% List all created files
fprintf('\nFiles created:\n');
system('find test_output -name "*.mat" -ls 2>/dev/null || echo "No .mat files found"');
system('find . -maxdepth 1 -name "*.mat" -ls 2>/dev/null || echo "No .mat files in current dir"');

exit;
EOF

# Load MATLAB module if needed
if ! command -v matlab &> /dev/null; then
    module load MATLAB/2023b || module load matlab/2023b || true
fi

# Run the test
echo "Running MATLAB test script..."
matlab -nodisplay -nosplash -r "run('$MATLAB_SCRIPT')"

# Check what was created
echo -e "\n=== Checking file system ==="
echo "Current directory: $(pwd)"
echo "Files in current directory:"
ls -la *.mat 2>/dev/null || echo "No .mat files in current directory"

echo -e "\nFiles in test_output:"
find test_output -name "*.mat" -ls 2>/dev/null || echo "No .mat files found"

# Cleanup
rm -f "$MATLAB_SCRIPT"