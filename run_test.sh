#!/bin/bash
# run_test.sh - Simple test of config duration

echo "Testing config-based duration..."

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Quick MATLAB test
TEMP=$(mktemp /tmp/test_XXXXXX.m)
cat > "$TEMP" << EOF
% CRITICAL: Change to project directory first!
cd('$PROJECT_DIR');
addpath(genpath('Code'));

fprintf('\n1. Config check: ');
try
    [~,pc] = get_plume_file();
    if isfield(pc,'simulation') && isfield(pc.simulation, 'duration_seconds')
        fprintf('%.0f seconds\n', pc.simulation.duration_seconds);
    else
        fprintf('not set\n');
    end
catch ME
    fprintf('Error: %s\n', ME.message);
end

fprintf('\n2. Test run: ');
try
    out = navigation_model_vec('config', 'gaussian', 0, 2);
    fprintf('SUCCESS! %d samples\n', size(out.x,1));
catch
    try
        out = navigation_model_vec(0, 'gaussian', 0, 2);
        fprintf('SUCCESS with 0! %d samples\n', size(out.x,1));
    catch ME
        fprintf('FAILED: %s\n', ME.message);
    end
end
exit;
EOF

matlab -nodisplay -nosplash -nojvm -r "run('$TEMP')" 2>&1 | grep -v ">>"
rm "$TEMP"