#!/bin/bash
# test_both.sh - Test both Crimaldi and Gaussian with config duration

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PLUME_CONFIG="$PROJECT_DIR/configs/plumes/crimaldi_10cms_bounded.json"

cat > /tmp/test_both.m << MATLAB_EOF
cd('$PROJECT_DIR');
addpath(genpath('Code'));

fprintf('=== Testing Config Duration ===\n\n');

% Test Crimaldi
fprintf('1. Crimaldi (15 Hz):\n');
try
    out1 = navigation_model_vec('config', 'Crimaldi', 0, 1);
    fprintf('   Samples: %d\n', size(out1.x, 1));
    fprintf('   Duration: %.1f seconds\n', size(out1.x, 1) / 15);
catch ME
    fprintf('   ERROR: %s\n', ME.message);
end

fprintf('\n2. Gaussian (50 Hz):\n');
try
    out2 = navigation_model_vec('config', 'gaussian', 0, 1);
    fprintf('   Samples: %d\n', size(out2.x, 1));
    fprintf('   Duration: %.1f seconds\n', size(out2.x, 1) / 50);
catch ME
    fprintf('   ERROR: %s\n', ME.message);
end

fprintf('\nBoth should show 300 seconds.\n');
exit;
MATLAB_EOF

matlab -nodisplay -nosplash -r "run('/tmp/test_both.m')" 2>&1 | tail -25
rm -f /tmp/test_both.m
