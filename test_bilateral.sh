#!/bin/bash
# test_bilateral.sh - Test bilateral sensing with both plumes
set -euo pipefail

# Create test directory
TEST_DIR="test_output/test_bilateral_$(date +%s)"
mkdir -p "$TEST_DIR"

# Use absolute path for metadata
PLUME_METADATA="/home/snb6/palmer_scratch/plume/smoke_1a_orig_backgroundsubtracted_meta.yaml"

echo "🧪 Testing Bilateral Sensing"
echo "============================"

# Test 1: Crimaldi plume with bilateral
echo -e "\n1️⃣ Testing Crimaldi plume with bilateral sensing..."
MATLAB_SCRIPT=$(mktemp test_bilateral_crim_XXXX.m)
cat > "$MATLAB_SCRIPT" << EOF
addpath('Code');

cfg = struct();
cfg.environment = 'crimaldi';
cfg.triallength = 3600;
cfg.bilateral = true;  % Enable bilateral sensing
cfg.ntrials = 1;
cfg.plotting = 0;
cfg.randomSeed = 42;
cfg.outputDir = '$TEST_DIR/crimaldi_bilateral';

fprintf('\\n=== Crimaldi + Bilateral ===\\n');
fprintf('Environment: %s\\n', cfg.environment);
fprintf('Bilateral: %d\\n', cfg.bilateral);
fprintf('Output: %s\\n', cfg.outputDir);

try
    mkdir(cfg.outputDir);
    tic;
    R = run_navigation_cfg(cfg);
    elapsed = toc;
    fprintf('Completed in %.2f seconds\\n', elapsed);
    
    save(fullfile(cfg.outputDir, 'result.mat'), '-struct', 'R', '-v7');
    fprintf('✓ SUCCESS: Crimaldi bilateral test passed\\n');
    
catch ME
    fprintf('✗ ERROR: %s\\n', ME.message);
end
EOF

matlab -nodisplay -nosplash -r "run('$MATLAB_SCRIPT'); exit;"
rm -f "$MATLAB_SCRIPT"

# Test 2: Custom plume with bilateral
echo -e "\n2️⃣ Testing Custom plume with bilateral sensing..."
MATLAB_SCRIPT=$(mktemp test_bilateral_custom_XXXX.m)
cat > "$MATLAB_SCRIPT" << EOF
addpath('Code');

cfg = struct();
cfg.environment = 'video';
cfg.plume_metadata = '$PLUME_METADATA';
cfg.triallength = 3600;
cfg.bilateral = true;  % Enable bilateral sensing
cfg.ntrials = 1;
cfg.plotting = 0;
cfg.randomSeed = 42;
cfg.outputDir = '$TEST_DIR/custom_bilateral';

fprintf('\\n=== Custom + Bilateral ===\\n');
fprintf('Environment: %s\\n', cfg.environment);
fprintf('Metadata: %s\\n', cfg.plume_metadata);
fprintf('Bilateral: %d\\n', cfg.bilateral);
fprintf('Output: %s\\n', cfg.outputDir);

try
    mkdir(cfg.outputDir);
    tic;
    R = run_navigation_cfg(cfg);
    elapsed = toc;
    fprintf('Completed in %.2f seconds\\n', elapsed);
    
    save(fullfile(cfg.outputDir, 'result.mat'), '-struct', 'R', '-v7');
    fprintf('✓ SUCCESS: Custom bilateral test passed\\n');
    
catch ME
    fprintf('✗ ERROR: %s\\n', ME.message);
end
EOF

matlab -nodisplay -nosplash -r "run('$MATLAB_SCRIPT'); exit;"
rm -f "$MATLAB_SCRIPT"

# Check results
echo -e "\n📊 Test Results:"
echo "=================="
if [ -f "$TEST_DIR/crimaldi_bilateral/result.mat" ]; then
    echo "✅ Crimaldi bilateral: PASSED"
    ls -lh "$TEST_DIR/crimaldi_bilateral/result.mat"
else
    echo "❌ Crimaldi bilateral: FAILED"
fi

if [ -f "$TEST_DIR/custom_bilateral/result.mat" ]; then
    echo "✅ Custom bilateral: PASSED"
    ls -lh "$TEST_DIR/custom_bilateral/result.mat"
else
    echo "❌ Custom bilateral: FAILED"
fi

echo -e "\nOutput directory: $TEST_DIR"