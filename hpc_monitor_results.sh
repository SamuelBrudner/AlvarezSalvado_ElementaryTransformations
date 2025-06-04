#!/bin/bash
# hpc_monitor_results.sh - Monitor progress and analyze results for HPC runs
#
# Usage: ./hpc_monitor_results.sh [MODE]
#        MODE: status, results, compare, or watch (default: status)

MODE="${1:-status}"
RESULTS_DIR="results"

case $MODE in
    status)
        echo "=== HPC Job Status ==="
        echo ""
        
        # Show running jobs
        RUNNING=$(squeue -u $USER -h -o "%j %T %M %l %C %m" | grep -E "(nav_|crim|smoke)")
        if [ -n "$RUNNING" ]; then
            echo "Running navigation jobs:"
            echo "NAME                STATE     TIME     TIMELIMIT  CPUS  MEMORY"
            echo "$RUNNING"
        else
            echo "No navigation jobs currently running"
        fi
        
        echo ""
        echo "=== Results Summary ==="
        
        # Count results
        CRIM_COUNT=$(ls -1 $RESULTS_DIR/nav_results_*.mat 2>/dev/null | grep -v smoke | wc -l)
        SMOKE_COUNT=$(ls -1 $RESULTS_DIR/smoke_nav_results_*.mat 2>/dev/null | wc -l)
        
        echo "Crimaldi results: $CRIM_COUNT files"
        echo "Smoke results: $SMOKE_COUNT files"
        
        # Recent jobs
        echo ""
        echo "Recent job history:"
        sacct -u $USER --format=JobName%20,State,ExitCode,Elapsed,Start \
              --noheader -S $(date -d '1 day ago' +%Y-%m-%d) | \
              grep -E "(nav_|crim|smoke)" | head -10
        ;;
        
    results)
        echo "=== Quick Results Analysis ==="
        
        # Create temporary MATLAB script
        cat > /tmp/analyze_recent_$$.m << 'EOF'
% Analyze recent results
fprintf('\nAnalyzing recent results...\n\n');

% Find result files
crim_files = dir('results/nav_results_*.mat');
smoke_files = dir('results/smoke_nav_results_*.mat');

% Remove smoke files from crimaldi list
crim_files = crim_files(~contains({crim_files.name}, 'smoke'));

fprintf('Found %d Crimaldi files\n', length(crim_files));
fprintf('Found %d Smoke files\n\n', length(smoke_files));

% Analyze last 10 of each
n_analyze = min(10, min(length(crim_files), length(smoke_files)));

if n_analyze > 0
    crim_success = zeros(n_analyze, 1);
    smoke_success = zeros(n_analyze, 1);
    
    % Get most recent files
    [~, idx] = sort([crim_files.datenum], 'descend');
    crim_files = crim_files(idx(1:n_analyze));
    
    [~, idx] = sort([smoke_files.datenum], 'descend');
    smoke_files = smoke_files(idx(1:n_analyze));
    
    % Load and analyze
    for i = 1:n_analyze
        c = load(fullfile('results', crim_files(i).name));
        if isfield(c.out, 'successrate')
            crim_success(i) = c.out.successrate * 100;
        end
        
        s = load(fullfile('results', smoke_files(i).name));
        if isfield(s.out, 'successrate')
            smoke_success(i) = s.out.successrate * 100;
        end
    end
    
    fprintf('=== Last %d Results ===\n', n_analyze);
    fprintf('Crimaldi: %.1f%% ± %.1f%%\n', mean(crim_success), std(crim_success));
    fprintf('Smoke:    %.1f%% ± %.1f%%\n', mean(smoke_success), std(smoke_success));
    
    % Show individual results
    fprintf('\nIndividual success rates:\n');
    fprintf('Task   Crimaldi   Smoke\n');
    fprintf('----   --------   -----\n');
    for i = 1:n_analyze
        fprintf('%3d    %6.1f%%   %6.1f%%\n', i, crim_success(i), smoke_success(i));
    end
else
    fprintf('No results found to analyze\n');
end
EOF
        
        # Run analysis
        cd /vast/palmer/home.grace/snb6/Documents/AlvarezSalvado_ElementaryTransformations
        matlab -batch "run('/tmp/analyze_recent_$$.m')" 2>/dev/null | grep -v "Loading" | tail -n +2
        rm -f /tmp/analyze_recent_$$.m
        ;;
        
    compare)
        echo "=== Comparative Analysis ==="
        
        # Find matching pairs (task 0 vs 1000, 1 vs 1001, etc.)
        echo "Analyzing matched pairs..."
        
        matlab -batch "
            cd('/vast/palmer/home.grace/snb6/Documents/AlvarezSalvado_ElementaryTransformations');
            
            % Find pairs
            crim_success = [];
            smoke_success = [];
            
            for i = 0:99
                crim_file = sprintf('results/nav_results_%04d.mat', i);
                smoke_file = sprintf('results/smoke_nav_results_%04d.mat', i+1000);
                
                if exist(crim_file, 'file') && exist(smoke_file, 'file')
                    c = load(crim_file);
                    s = load(smoke_file);
                    
                    if isfield(c.out, 'successrate') && isfield(s.out, 'successrate')
                        crim_success(end+1) = c.out.successrate * 100;
                        smoke_success(end+1) = s.out.successrate * 100;
                    end
                end
            end
            
            if ~isempty(crim_success)
                fprintf('\nFound %d matched pairs\n', length(crim_success));
                fprintf('Crimaldi: %.1f%% ± %.1f%%\n', mean(crim_success), std(crim_success));
                fprintf('Smoke:    %.1f%% ± %.1f%%\n', mean(smoke_success), std(smoke_success));
                
                % Paired t-test
                [h, p] = ttest(crim_success, smoke_success);
                fprintf('\nPaired t-test p-value: %.4f\n', p);
                
                % Win/loss analysis
                crim_wins = sum(crim_success > smoke_success);
                smoke_wins = sum(smoke_success > crim_success);
                ties = sum(crim_success == smoke_success);
                
                fprintf('\nHead-to-head:\n');
                fprintf('  Crimaldi wins: %d (%.1f%%)\n', crim_wins, 100*crim_wins/length(crim_success));
                fprintf('  Smoke wins: %d (%.1f%%)\n', smoke_wins, 100*smoke_wins/length(smoke_success));
                fprintf('  Ties: %d\n', ties);
            else
                fprintf('No matched pairs found\n');
            end
        " 2>/dev/null | grep -v "Loading" | tail -n +2
        ;;
        
    watch)
        echo "Monitoring results in real-time..."
        echo "Press Ctrl+C to stop"
        echo ""
        
        watch -n 10 "
            echo '=== Live Results Monitor ==='
            echo 'Time: '\$(date)
            echo ''
            echo 'Running jobs:'
            squeue -u $USER -h -o '%j %T %M' | grep -E '(nav_|crim|smoke)' | head -5
            echo ''
            echo 'Results count:'
            echo '  Crimaldi: '\$(ls -1 results/nav_results_*.mat 2>/dev/null | grep -v smoke | wc -l)
            echo '  Smoke: '\$(ls -1 results/smoke_nav_results_*.mat 2>/dev/null | wc -l)
            echo ''
            echo 'Latest files:'
            ls -lt results/*nav_results_*.mat 2>/dev/null | head -5 | awk '{print \$9}'
        "
        ;;
        
    *)
        echo "Usage: $0 {status|results|compare|watch}"
        echo ""
        echo "  status  - Show job status and counts"
        echo "  results - Analyze recent results"
        echo "  compare - Compare matched Crimaldi/Smoke pairs"
        echo "  watch   - Live monitoring (updates every 10s)"
        ;;
esac