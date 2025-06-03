#!/bin/bash
# analyze_run_test.sh - Check what run_test.sh is doing

echo "=== Analyzing run_test.sh ==="
echo ""

echo "1. Current run_test.sh content (first 20 lines):"
head -20 run_test.sh 2>/dev/null || echo "File not found"
echo ""

echo "2. Let's create a working test script:"
cat > test_duration.sh << 'TESTSCRIPT'
#!/bin/bash
# test_duration.sh - Test config duration loading

echo "=== Testing Config Duration ==="
echo ""

# Ensure we're in the project root
cd "$(dirname "$0")"

# Set config path explicitly
export PLUME_CONFIG="$(pwd)/configs/plumes/crimaldi_10cms_bounded.json"
echo "Config: $PLUME_CONFIG"
echo "Duration setting: $(grep duration_seconds "$PLUME_CONFIG")"
echo ""

# Run MATLAB test
matlab -nodisplay -nosplash -r "
% Add paths
addpath(genpath('Code'));

fprintf('\\nTest 1 - Load config:\\n');
try
    [pf, pc] = get_plume_file();
    fprintf('  ✓ Config loaded\\n');
    if isfield(pc, 'simulation') && isfield(pc.simulation, 'duration_seconds')
        fprintf('  ✓ Duration: %.1f seconds\\n', pc.simulation.duration_seconds);
    else
        fprintf('  ✗ No duration field\\n');
    end
catch ME
    fprintf('  ✗ Error: %s\\n', ME.message);
end

fprintf('\\nTest 2 - Run Crimaldi simulation:\\n');
try
    out = navigation_model_vec('config', 'Crimaldi', 0, 1);
    n_samples = size(out.x, 1);
    duration = n_samples / 15;  % Crimaldi runs at 15 Hz
    fprintf('  ✓ Success: %d samples = %.1f seconds\\n', n_samples, duration);
catch ME
    fprintf('  ✗ Error: %s\\n', ME.message);
end

fprintf('\\nTest 3 - Run Gaussian simulation:\\n');
try
    out = navigation_model_vec('config', 'gaussian', 0, 1);
    n_samples = size(out.x, 1);
    duration = n_samples / 50;  % Gaussian runs at 50 Hz
    fprintf('  ✓ Success: %d samples = %.1f seconds\\n', n_samples, duration);
catch ME
    fprintf('  ✗ Error: %s\\n', ME.message);
end

exit;
" 2>&1 | grep -v ">>"
TESTSCRIPT

chmod +x test_duration.sh
echo "Created test_duration.sh"
echo ""
echo "Run it with: ./test_duration.sh"