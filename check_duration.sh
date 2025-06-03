#!/bin/bash
# check_duration.sh - Fixed version with proper directory handling

# Get absolute project path
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Set paths
export PLUME_CONFIG="$PROJECT_DIR/configs/plumes/crimaldi_10cms_bounded.json"

# Create minimal test
cat > /tmp/check.m << EOF
% CRITICAL: Change to project directory first!
cd('$PROJECT_DIR');

% Now add Code to path
addpath(genpath('Code'));

try
    [~, pc] = get_plume_file();
    if isfield(pc, 'simulation') && isfield(pc.simulation, 'duration_seconds')
        fprintf('SUCCESS: Duration = %.0f seconds\n', pc.simulation.duration_seconds);
    else
        fprintf('FAIL: No duration field\n');
    end
catch ME
    fprintf('ERROR: %s\n', ME.message);
end
exit;
EOF

# Run it
echo "Checking duration loading..."
matlab -nodisplay -nosplash -nojvm -r "run('/tmp/check.m')" 2>&1 | grep -E "(SUCCESS|FAIL|ERROR)"

rm -f /tmp/check.m