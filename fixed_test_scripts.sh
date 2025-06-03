#!/bin/bash
# fixed_test_scripts.sh - Create all test scripts with proper CD

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Creating fixed test scripts..."

# 1. Simple duration test
cat > simple_duration_test.sh << 'EOF'
#!/bin/bash
# simple_duration_test.sh - Simple test for duration config

echo "=== Simple Duration Test ==="
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
CONFIG_PATH="$PROJECT_ROOT/configs/plumes/crimaldi_10cms_bounded.json"

echo "Config: $CONFIG_PATH"
echo "Duration: $(grep duration_seconds "$CONFIG_PATH")"

cat > /tmp/duration_test.m << MATLAB_EOF
cd('$PROJECT_ROOT');
addpath(genpath('Code'));
setenv('PLUME_CONFIG', '$CONFIG_PATH');

fprintf('Loading config...\n');
try
    [pf, pc] = get_plume_file();
    if isfield(pc, 'simulation') && isfield(pc.simulation, 'duration_seconds')
        fprintf('✓ Duration from config: %.1f seconds\n', pc.simulation.duration_seconds);
    else
        fprintf('✗ No duration field found\n');
    end
catch ME
    fprintf('✗ Error: %s\n', ME.message);
end

fprintf('\nTesting Crimaldi simulation...\n');
try
    out = navigation_model_vec('config', 'Crimaldi', 0, 1);
    fprintf('✓ Result: %d samples at 15Hz = %.1f seconds\n', ...
            size(out.x,1), size(out.x,1)/15);
catch ME
    fprintf('✗ Error: %s\n', ME.message);
end
exit;
MATLAB_EOF

matlab -nodisplay -nosplash -r "run('/tmp/duration_test.m')" 2>&1 | tail -20
rm -f /tmp/duration_test.m
EOF

# 2. Test Crimaldi 300s
cat > test_crimaldi_300s.sh << 'EOF'
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
EOF

# 3. Quick test both environments
cat > test_both.sh << 'EOF'
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
EOF

chmod +x simple_duration_test.sh test_crimaldi_300s.sh test_both.sh check_duration.sh

echo ""
echo "Fixed test scripts created:"
echo "  ./check_duration.sh      - Quick duration check"
echo "  ./simple_duration_test.sh - Simple test with output"
echo "  ./test_crimaldi_300s.sh  - Detailed Crimaldi test"
echo "  ./test_both.sh           - Test both environments"