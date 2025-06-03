#!/bin/bash
# quick_check.sh - Quick check of your single result file (FIXED)

echo "=== Quick Results Check ==="
echo ""

# First check if file exists
if [ ! -f "results/nav_results_0000.mat" ]; then
    echo "Error: results/nav_results_0000.mat not found!"
    echo "Run the simulation first: ./run_nav_model.sh test"
    exit 1
fi

# Kill any hanging MATLAB processes
pkill -f matlab 2>/dev/null

# Use timeout and proper MATLAB syntax
timeout 30s matlab -nodisplay -nosplash -nojvm << 'EOF' 2>&1 | grep -v "^>>" | grep -v "^$"
try
    fprintf('Loading results file...\n');
    load('results/nav_results_0000.mat');
    
    fprintf('\nFile: nav_results_0000.mat\n');
    fprintf('Environment: %s\n', out.environment);
    fprintf('Agents: %d\n', size(out.x, 2));
    fprintf('Duration: %.1f seconds\n', size(out.x, 1) / 15);
    
    if isfield(out, 'successrate')
        fprintf('Success rate: %.1f%%\n', out.successrate * 100);
    end
    
    if isfield(out, 'latency')
        successful = sum(~isnan(out.latency));
        fprintf('Successful agents: %d out of %d\n', successful, length(out.latency));
        if successful > 0
            fprintf('Mean time to target: %.1f seconds\n', nanmean(out.latency));
            fprintf('Range: %.1f - %.1f seconds\n', nanmin(out.latency), nanmax(out.latency));
        end
    end
    
    % Quick distance check
    if size(out.x, 1) > 1
        dist1 = sum(sqrt(diff(out.x(:,1)).^2 + diff(out.y(:,1)).^2));
        fprintf('\nAgent 1 traveled: %.1f cm\n', dist1);
        fprintf('Started at: (%.1f, %.1f) cm\n', out.x(1,1), out.y(1,1));
        fprintf('Ended at: (%.1f, %.1f) cm\n', out.x(end,1), out.y(end,1));
    end
    
    fprintf('\n✓ Check complete\n');
catch ME
    fprintf('Error: %s\n', ME.message);
end
exit(0);
EOF

STATUS=$?
if [ $STATUS -eq 124 ]; then
    echo "✗ Check timed out after 30 seconds"
    echo "  The results file might be corrupted or too large"
else
    echo ""
    echo "For plots, use: python view_results.py results/nav_results_0000.mat"
fi