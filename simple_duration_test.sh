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
