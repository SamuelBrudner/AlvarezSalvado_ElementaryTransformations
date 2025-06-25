#!/bin/bash
# hpc_monitor_results.sh - Monitor progress and analyze results for HPC runs
#
# Usage: ./scripts/hpc_monitor_results.sh [MODE] [-v|--verbose]
#        MODE: status, results, compare, or watch (default: status)
#        -v, --verbose: Enable detailed trace output

# Initialize variables
MODE=""
VERBOSE=0
RESULTS_DIR="results"
SCRIPT_NAME="hpc_monitor_results"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        status|results|compare|watch)
            MODE="$1"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [MODE] [-v|--verbose]"
            echo ""
            echo "MODE options:"
            echo "  status  - Show job status and counts (default)"
            echo "  results - Analyze recent results"
            echo "  compare - Compare matched Crimaldi/Smoke pairs"
            echo "  watch   - Live monitoring (updates every 10s)"
            echo ""
            echo "OPTIONS:"
            echo "  -v, --verbose  Enable detailed trace output"
            echo "  -h, --help     Show this help message"
            exit 0
            ;;
        *)
            if [[ -z "$MODE" ]]; then
                MODE="$1"
            else
                echo "Unknown option: $1" >&2
                echo "Use -h or --help for usage information" >&2
                exit 1
            fi
            shift
            ;;
    esac
done

# Set default mode if not specified
MODE="${MODE:-status}"

# Setup logging infrastructure
LOG_DIR="logs"
if [[ ! -d "$LOG_DIR" ]]; then
    mkdir -p "$LOG_DIR"
fi

LOG_FILE="${LOG_DIR}/${SCRIPT_NAME}_$(date +%Y%m%d_%H%M%S).log"

# Logging function - outputs to both stdout and log file when verbose
log_verbose() {
    local message="[$(date '+%Y-%m-%d %H:%M:%S')] [$SCRIPT_NAME] $1"
    if [[ $VERBOSE -eq 1 ]]; then
        echo "$message"
        echo "$message" >> "$LOG_FILE"
    fi
}

# Standard logging function - always outputs to log file, conditionally to stdout
log_info() {
    local message="[$(date '+%Y-%m-%d %H:%M:%S')] [$SCRIPT_NAME] INFO: $1"
    echo "$message" >> "$LOG_FILE"
    if [[ $VERBOSE -eq 1 ]]; then
        echo "$message"
    fi
}

# Error logging function
log_error() {
    local message="[$(date '+%Y-%m-%d %H:%M:%S')] [$SCRIPT_NAME] ERROR: $1"
    echo "$message" >&2
    echo "$message" >> "$LOG_FILE"
}

# Initialize logging
log_info "Starting HPC monitoring script in mode: $MODE"
log_verbose "Verbose logging enabled"
log_verbose "Results directory: $RESULTS_DIR"
log_verbose "Log file: $LOG_FILE"

case $MODE in
    status)
        log_verbose "Executing status monitoring mode"
        echo "=== HPC Job Status ==="
        echo ""
        
        log_verbose "Querying running jobs for user: $USER"
        # Show running jobs
        RUNNING=$(squeue -u $USER -h -o "%j %T %M %l %C %m" | grep -E "(nav_|crim|smoke)")
        if [ -n "$RUNNING" ]; then
            log_verbose "Found running navigation jobs"
            echo "Running navigation jobs:"
            echo "NAME                STATE     TIME     TIMELIMIT  CPUS  MEMORY"
            echo "$RUNNING"
            log_info "Active jobs found: $(echo "$RUNNING" | wc -l)"
        else
            log_verbose "No running navigation jobs found"
            echo "No navigation jobs currently running"
            log_info "No active navigation jobs"
        fi
        
        echo ""
        echo "=== Results Summary ==="
        
        log_verbose "Counting results in $RESULTS_DIR directory"
        # Count results
        CRIM_COUNT=$(ls -1 $RESULTS_DIR/nav_results_*.mat 2>/dev/null | grep -v smoke | wc -l)
        SMOKE_COUNT=$(ls -1 $RESULTS_DIR/smoke_nav_results_*.mat 2>/dev/null | wc -l)
        
        log_verbose "Found $CRIM_COUNT Crimaldi results and $SMOKE_COUNT Smoke results"
        echo "Crimaldi results: $CRIM_COUNT files"
        echo "Smoke results: $SMOKE_COUNT files"
        log_info "Results summary: Crimaldi=$CRIM_COUNT, Smoke=$SMOKE_COUNT"
        
        # Recent jobs
        echo ""
        echo "Recent job history:"
        log_verbose "Querying job history for the last 24 hours"
        sacct -u $USER --format=JobName%20,State,ExitCode,Elapsed,Start \
              --noheader -S $(date -d '1 day ago' +%Y-%m-%d) | \
              grep -E "(nav_|crim|smoke)" | head -10
        log_verbose "Status monitoring completed"
        ;;
        
    results)
        log_verbose "Executing results analysis mode"
        echo "=== Quick Results Analysis ==="
        
        log_verbose "Creating temporary MATLAB analysis script"
        # Create temporary MATLAB script
        TEMP_SCRIPT="/tmp/analyze_recent_$$.m"
        log_verbose "Temporary script path: $TEMP_SCRIPT"
        
        cat > "$TEMP_SCRIPT" << 'EOF'
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
        
        log_verbose "Running MATLAB analysis script"
        # Run analysis
        cd /vast/palmer/home.grace/snb6/Documents/AlvarezSalvado_ElementaryTransformations
        log_verbose "Changed to working directory: $(pwd)"
        
        matlab -batch "run('$TEMP_SCRIPT')" 2>/dev/null | grep -v "Loading" | tail -n +2
        
        log_verbose "Cleaning up temporary script"
        rm -f "$TEMP_SCRIPT"
        log_verbose "Results analysis completed"
        ;;
        
    compare)
        log_verbose "Executing comparative analysis mode"
        echo "=== Comparative Analysis ==="
        
        # Find matching pairs (task 0 vs 1000, 1 vs 1001, etc.)
        echo "Analyzing matched pairs..."
        log_verbose "Starting paired comparison analysis between Crimaldi and Smoke results"
        
        log_verbose "Running MATLAB comparative analysis"
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
        log_verbose "Comparative analysis completed"
        ;;
        
    watch)
        log_verbose "Executing watch mode for real-time monitoring"
        echo "Monitoring results in real-time..."
        echo "Press Ctrl+C to stop"
        echo ""
        
        log_verbose "Starting watch mode with 10-second intervals"
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
        log_verbose "Watch mode terminated"
        ;;
        
    *)
        log_error "Invalid mode specified: $MODE"
        echo "Usage: $0 {status|results|compare|watch} [-v|--verbose]"
        echo ""
        echo "  status  - Show job status and counts"
        echo "  results - Analyze recent results"
        echo "  compare - Compare matched Crimaldi/Smoke pairs"
        echo "  watch   - Live monitoring (updates every 10s)"
        echo ""
        echo "  -v, --verbose  Enable detailed trace output"
        echo "  -h, --help     Show help message"
        exit 1
        ;;
esac

log_info "HPC monitoring script completed successfully"
log_verbose "Script execution finished"