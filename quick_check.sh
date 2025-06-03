#!/bin/bash
# quick_check.sh - Quick check of your single result file

echo "=== Quick Results Check ==="
echo ""

# Use MATLAB to get the key stats
matlab -nodisplay -nosplash -r "
try
    load('results/nav_results_0000.mat');
    fprintf('\nFile: nav_results_0000.mat\n');
    fprintf('Environment: %s\n', out.environment);
    fprintf('Agents: %d\n', size(out.x, 2));
    fprintf('Duration: %.1f seconds\n', size(out.x, 1) / 15);
    fprintf('Success rate: %.1f%%\n', out.successrate * 100);
    
    successful = sum(~isnan(out.latency));
    fprintf('Successful agents: %d out of %d\n', successful, length(out.latency));
    
    if successful > 0
        fprintf('Mean time to target: %.1f seconds\n', nanmean(out.latency));
        fprintf('Range: %.1f - %.1f seconds\n', nanmin(out.latency), nanmax(out.latency));
    end
    
    % Quick distance check
    dist1 = sum(sqrt(diff(out.x(:,1)).^2 + diff(out.y(:,1)).^2));
    fprintf('\nAgent 1 traveled: %.1f cm\n', dist1);
    fprintf('Started at: (%.1f, %.1f) cm\n', out.x(1,1), out.y(1,1));
    fprintf('Ended at: (%.1f, %.1f) cm\n', out.x(end,1), out.y(end,1));
    
catch ME
    fprintf('Error: %s\n', ME.message);
end
exit;
" 2>/dev/null | grep -v "^>>" | tail -n +11

echo ""
echo "For plots, use: matlab -r \"load('results/nav_results_0000.mat'); plot(out.x, out.y)\""