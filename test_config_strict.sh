#!/bin/bash
# test_config_strict.sh - Test strict config behavior

echo "=== Testing strict config behavior ==="

# Test 1: Default (should use crimaldi_10cms_bounded.json)
echo -e "\n1. Default (should succeed):"
unset PLUME_CONFIG
matlab -nodisplay -nosplash -nojvm -r "
addpath('Code');
try
    [pf, pc] = get_plume_file();
    fprintf('✓ Success - using config from plumes directory\n');
catch ME
    fprintf('✗ Failed: %s\n', ME.message);
end
exit
" 2>&1 | grep -E "(Loading|✓|✗)"

# Test 2: Bad config path (should fail)
echo -e "\n2. Bad config path (should fail):"
export PLUME_CONFIG="/bad/path/config.json"
matlab -nodisplay -nosplash -nojvm -r "
addpath('Code');
try
    [pf, pc] = get_plume_file();
    fprintf('✗ Unexpected success!\n');
catch ME
    fprintf('✓ Failed as expected: %s\n', ME.identifier);
end
exit
" 2>&1 | grep -E "(Loading|✓|✗)"

# Test 3: Verify simulation fails with bad config
echo -e "\n3. Simulation with bad config (should fail):"
timeout 10s matlab -nodisplay -nosplash -nojvm -r "
addpath('Code');
try
    out = navigation_model_vec(100, 'Crimaldi', 0, 1);
    fprintf('✗ Unexpected success!\n');
catch ME
    fprintf('✓ Failed as expected\n');
end
exit
" 2>&1 | grep -E "(✓|✗|ERROR)"