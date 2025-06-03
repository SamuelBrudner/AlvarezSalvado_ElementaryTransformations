#!/bin/bash
# matlab_results_check.sh - Check results using MATLAB (no Python needed)
# 
# Usage: ./matlab_results_check.sh [results_file.mat]
# Default: results/nav_results_0000.mat
#
# This script provides a quick summary of navigation model results using MATLAB

RESULT_FILE="${1:-results/nav_results_0000.mat}"

if [ ! -f "$RESULT_FILE" ]; then
    echo "Error: $RESULT_FILE not found"
    exit 1
fi

echo "=== MATLAB Results Check ==="
echo "File: $RESULT_FILE"
echo ""

# Use MATLAB for analysis
matlab -nodisplay -nosplash << EOF 2>/dev/null | grep -v "^>>" | grep -v "^ $"
try
    % Load results
    data = load('$RESULT_FILE');
    out = data.out;
    
    % Basic info
    fprintf('Environment: %s\n', out.environment);
    
    % Get dimensions
    [n_samples, n_agents] = size(out.x);
    if n_agents == 1 && n_samples > 1000
        % Might be transposed for single agent
        n_agents = 1;
    end
    
    fprintf('Agents: %d\n', n_agents);
    fprintf('Duration: %.1f seconds\n\n', n_samples/15);
    
    % Success metrics
    if isfield(out, 'successrate')
        fprintf('✓ Success rate: %.1f%%\n', out.successrate * 100);
    end
    
    if isfield(out, 'latency')
        successful = sum(~isnan(out.latency));
        fprintf('✓ Successful agents: %d/%d\n', successful, n_agents);
        if successful > 0
            fprintf('✓ Mean time to target: %.1f seconds\n', nanmean(out.latency));
            fprintf('✓ Range: %.1f - %.1f seconds\n', ...
                    nanmin(out.latency), nanmax(out.latency));
        end
    end
    
    % Trajectory summary
    fprintf('\nTrajectory Summary:\n');
    fprintf('Starting Y positions: %.1f to %.1f cm\n', ...
            min(out.y(1,:)), max(out.y(1,:)));
    fprintf('Ending Y positions: %.1f to %.1f cm\n', ...
            min(out.y(end,:)), max(out.y(end,:)));
    
    % Distance for first agent
    dx = diff(out.x(:,1));
    dy = diff(out.y(:,1));
    dist = sum(sqrt(dx.^2 + dy.^2));
    fprintf('Agent 1 distance traveled: %.1f cm\n', dist);
    
catch ME
    fprintf('Error: %s\n', ME.message);
end
exit;
EOF

echo ""
echo "For visualizations, run MATLAB interactively and use:"
echo "  load('$RESULT_FILE')"
echo "  plot(out.x, out.y)"