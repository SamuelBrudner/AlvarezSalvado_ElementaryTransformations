#!/bin/bash
# test_crimaldi_300s.sh - Test that Crimaldi runs for 300 seconds

echo "=== Testing Crimaldi 300s Duration ==="
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PLUME_CONFIG="$PROJECT_DIR/configs/plumes/crimaldi_10cms_bounded.json"

cat > /tmp/test_crimaldi.m << MATLAB_EOF
cd('$PROJECT_DIR');
addpath(genpath('Code'));

fprintf('Testing Crimaldi environment with config duration...\n');
try
    tic;
    out = navigation_model_vec('config', 'Crimaldi', 0, 1);
    elapsed = toc;
    
    n_samples = size(out.x, 1);
    duration_seconds = n_samples / 15;
    
    fprintf('\n=== RESULTS ===\n');
    fprintf('Samples: %d\n', n_samples);
    fprintf('Duration: %.1f seconds\n', duration_seconds);
    fprintf('Expected: 300.0 seconds\n');
    if abs(duration_seconds - 300) < 1
        fprintf('Match: YES ✓\n');
    else
        fprintf('Match: NO ✗\n');
    end
    fprintf('Computation time: %.1f seconds\n', elapsed);
catch ME
    fprintf('ERROR: %s\n', ME.message);
end
exit;
MATLAB_EOF

matlab -nodisplay -nosplash -r "run('/tmp/test_crimaldi.m')"
rm -f /tmp/test_crimaldi.m
