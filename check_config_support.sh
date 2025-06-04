#!/bin/bash
# check_config_support.sh - Check if navigation_model_vec supports 'config' parameter

echo "Checking if navigation_model_vec supports 'config' parameter..."
echo ""

# Check if the function handles string input
echo "Looking for string/config handling in navigation_model_vec.m:"
grep -n -A5 "triallength" Code/navigation_model_vec.m | grep -E "(ischar|isstring|config)" | head -10

echo ""
echo "Looking for 'duration_seconds' usage:"
grep -n "duration_seconds" Code/navigation_model_vec.m

echo ""
echo "Quick test to see what happens with 'config':"

cat > test_config_param.m << 'EOF'
fprintf('\nTesting config parameter support...\n');
addpath(genpath('Code'));

% Test 1: Check if config loads duration
[~, pc] = get_plume_file();
if isfield(pc, 'simulation') && isfield(pc.simulation, 'duration_seconds')
    fprintf('Config has duration: %.0f seconds\n', pc.simulation.duration_seconds);
else
    fprintf('Config missing duration\n');
end

% Test 2: Try a very short simulation with 'config'
fprintf('\nTrying navigation_model_vec with different parameters:\n');

% First with a number
try
    out1 = navigation_model_vec(150, 'Crimaldi', 0, 1);  % 10 seconds at 15Hz
    fprintf('Numeric param: %d samples = %.1f seconds\n', size(out1.x,1), size(out1.x,1)/15);
catch ME
    fprintf('Numeric param failed: %s\n', ME.message);
end

% Then with 'config'
try
    out2 = navigation_model_vec('config', 'Crimaldi', 0, 1);
    fprintf('Config param: %d samples = %.1f seconds\n', size(out2.x,1), size(out2.x,1)/15);
catch ME
    fprintf('Config param failed: %s\n', ME.message);
    % Maybe it's not implemented - try 0 which might trigger config reading
    try
        out2 = navigation_model_vec(0, 'Crimaldi', 0, 1);
        fprintf('Zero param: %d samples = %.1f seconds\n', size(out2.x,1), size(out2.x,1)/15);
    catch ME2
        fprintf('Zero param also failed: %s\n', ME2.message);
    end
end

exit;
EOF

matlab -nodisplay -nosplash < test_config_param.m