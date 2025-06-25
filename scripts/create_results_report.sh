#!/bin/bash
# create_results_report.sh - Generate comprehensive results summary using MATLAB
#
# Usage: ./scripts/create_results_report.sh [-v|--verbose] [-h|--help]
#
# Options:
#   -v, --verbose    Enable detailed trace output and logging
#   -h, --help       Show this help message
#
# Analyzes all nav_results_*.mat files in results/ directory
# Creates a summary report with statistics

set -euo pipefail

# Parse command line arguments for verbose logging
VERBOSE=0
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [-v|--verbose] [-h|--help]"
            echo ""
            echo "Generate comprehensive results summary using MATLAB"
            echo ""
            echo "Options:"
            echo "  -v, --verbose    Enable detailed trace output and logging"
            echo "  -h, --help       Show this help message"
            echo ""
            echo "Description:"
            echo "  Analyzes all nav_results_*.mat files in results/ directory"
            echo "  Creates a summary report with statistics and performance metrics"
            echo "  Output is written to stdout for pipeline integration"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Determine project root (script is in scripts/ subdirectory)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE=""

# Initialize logging if verbose mode is enabled
if [[ $VERBOSE -eq 1 ]]; then
    LOG_FILE="$PROJECT_ROOT/logs/create_results_report_${TIMESTAMP}.log"
    mkdir -p "$PROJECT_ROOT/logs"
    echo "[$(date)] Starting verbose logging for create_results_report.sh" | tee -a "$LOG_FILE"
    echo "[$(date)] Log file: $LOG_FILE" | tee -a "$LOG_FILE"
fi

# Verbose logging function
log_verbose() {
    if [[ $VERBOSE -eq 1 ]]; then
        echo "[$(date)] $1" | tee -a "$LOG_FILE"
    fi
}

# Standard output function that respects verbose mode
log_info() {
    echo "$1"
    if [[ $VERBOSE -eq 1 ]]; then
        echo "[$(date)] INFO: $1" >> "$LOG_FILE"
    fi
}

log_verbose "Results report generation started with verbose logging enabled"
log_verbose "Project Root: $PROJECT_ROOT"
log_verbose "Timestamp: $TIMESTAMP"

# Configuration
RESULTS_DIR="$PROJECT_ROOT/results"
REPORT_FILE="results_summary_${TIMESTAMP}.txt"

log_verbose "Configuration initialized:"
log_verbose "  Results Directory: $RESULTS_DIR"
log_verbose "  Report File: $REPORT_FILE"

# Create report header
log_verbose "Creating report header"
echo "=== Navigation Model Results Summary ===" > "$REPORT_FILE"
echo "Generated: $(date)" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

log_verbose "Report file initialized successfully"

# Count result files
log_verbose "Scanning for result files in $RESULTS_DIR"
N_FILES=$(ls -1 "$RESULTS_DIR"/nav_results_*.mat 2>/dev/null | wc -l)
echo "Total result files: $N_FILES" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

log_verbose "Found $N_FILES result files to analyze"

if [ $N_FILES -eq 0 ]; then
    log_verbose "No result files found, creating empty report"
    echo "No result files found in $RESULTS_DIR/" >> "$REPORT_FILE"
    cat "$REPORT_FILE"
    log_verbose "Empty report generated and displayed"
    if [[ $VERBOSE -eq 1 ]]; then
        log_verbose "Verbose logging completed. Log file saved to: $LOG_FILE"
    fi
    exit 0
fi

log_verbose "Proceeding with analysis of $N_FILES result files"

# Create MATLAB analysis script
TEMP_SCRIPT="temp_analyze_all_${TIMESTAMP}.m"
log_verbose "Creating temporary MATLAB analysis script: $TEMP_SCRIPT"

cat > "$TEMP_SCRIPT" << 'EOF'
% Temporary script to analyze all results
% Enhanced version with comprehensive statistics and error handling

results_dir = 'results';
files = dir(fullfile(results_dir, 'nav_results_*.mat'));

fprintf('\nIndividual Results:\n');
fprintf('==================\n');

% Initialize aggregated data structures
all_success_rates = [];
all_latencies = [];
all_agent_counts = [];
all_file_info = {};
total_agents = 0;
processed_files = 0;
failed_files = 0;

% Process each result file
for i = 1:length(files)
    filepath = fullfile(files(i).folder, files(i).name);
    fprintf('\nFile: %s\n', files(i).name);
    
    try
        % Load and validate data structure
        data = load(filepath);
        
        % Check for expected data structure
        if ~isfield(data, 'out')
            fprintf('  Warning: Missing "out" field in data structure\n');
            failed_files = failed_files + 1;
            continue;
        end
        
        out = data.out;
        processed_files = processed_files + 1;
        
        % Extract basic metrics
        [n_samples, n_agents] = size(out.x);
        total_agents = total_agents + n_agents;
        all_agent_counts(end+1) = n_agents;
        
        fprintf('  Agents: %d\n', n_agents);
        fprintf('  Samples: %d\n', n_samples);
        
        % Process success rate
        if isfield(out, 'successrate')
            success_rate = out.successrate;
            fprintf('  Success rate: %.1f%%\n', success_rate * 100);
            all_success_rates(end+1) = success_rate;
        else
            fprintf('  Success rate: Not available\n');
        end
        
        % Process latency data
        if isfield(out, 'latency')
            latencies = out.latency;
            successful = ~isnan(latencies);
            n_success = sum(successful);
            
            if n_success > 0
                mean_lat = mean(latencies(successful));
                median_lat = median(latencies(successful));
                std_lat = std(latencies(successful));
                
                fprintf('  Successful navigations: %d\n', n_success);
                fprintf('  Mean latency: %.1f s\n', mean_lat);
                fprintf('  Median latency: %.1f s\n', median_lat);
                fprintf('  Latency std: %.1f s\n', std_lat);
                fprintf('  Latency range: %.1f - %.1f s\n', ...
                        min(latencies(successful)), max(latencies(successful)));
                
                all_latencies = [all_latencies, latencies(successful)];
            else
                fprintf('  No successful navigations found\n');
            end
        else
            fprintf('  Latency data: Not available\n');
        end
        
        % Additional analysis if trajectory data is available
        if isfield(out, 'x') && isfield(out, 'y')
            % Calculate trajectory statistics
            trajectory_lengths = zeros(1, n_agents);
            for agent = 1:n_agents
                x_traj = out.x(:, agent);
                y_traj = out.y(:, agent);
                valid_points = ~isnan(x_traj) & ~isnan(y_traj);
                if sum(valid_points) > 1
                    dx = diff(x_traj(valid_points));
                    dy = diff(y_traj(valid_points));
                    trajectory_lengths(agent) = sum(sqrt(dx.^2 + dy.^2));
                end
            end
            
            valid_trajectories = trajectory_lengths > 0;
            if any(valid_trajectories)
                mean_traj_length = mean(trajectory_lengths(valid_trajectories));
                fprintf('  Mean trajectory length: %.2f units\n', mean_traj_length);
            end
        end
        
        % Store file information for summary
        all_file_info{end+1} = struct('name', files(i).name, ...
                                      'agents', n_agents, ...
                                      'processed', true);
        
    catch ME
        fprintf('  Error: %s\n', ME.message);
        fprintf('  Stack trace: %s\n', ME.stack(1).name);
        failed_files = failed_files + 1;
        
        % Store failed file information
        all_file_info{end+1} = struct('name', files(i).name, ...
                                      'agents', 0, ...
                                      'processed', false, ...
                                      'error', ME.message);
    end
end

% Generate comprehensive summary statistics
fprintf('\n\nOverall Statistics:\n');
fprintf('==================\n');
fprintf('Files processed successfully: %d/%d\n', processed_files, length(files));
fprintf('Files with errors: %d\n', failed_files);
fprintf('Total agents simulated: %d\n', total_agents);

if ~isempty(all_agent_counts)
    fprintf('Agent count per file - Mean: %.1f, Range: %d - %d\n', ...
            mean(all_agent_counts), min(all_agent_counts), max(all_agent_counts));
end

% Success rate analysis
if ~isempty(all_success_rates)
    fprintf('\nSuccess Rate Analysis:\n');
    fprintf('  Mean success rate: %.1f%%\n', mean(all_success_rates) * 100);
    fprintf('  Median success rate: %.1f%%\n', median(all_success_rates) * 100);
    fprintf('  Success rate std: %.1f%%\n', std(all_success_rates) * 100);
    fprintf('  Success rate range: %.1f%% - %.1f%%\n', ...
            min(all_success_rates)*100, max(all_success_rates)*100);
    
    % Performance categorization
    high_performance = sum(all_success_rates > 0.8);
    medium_performance = sum(all_success_rates > 0.5 & all_success_rates <= 0.8);
    low_performance = sum(all_success_rates <= 0.5);
    
    fprintf('  Performance distribution:\n');
    fprintf('    High (>80%%): %d files\n', high_performance);
    fprintf('    Medium (50-80%%): %d files\n', medium_performance);
    fprintf('    Low (<=50%%): %d files\n', low_performance);
end

% Latency analysis
if ~isempty(all_latencies)
    fprintf('\nLatency Analysis:\n');
    fprintf('  Total successful navigations: %d\n', length(all_latencies));
    fprintf('  Overall mean latency: %.1f seconds\n', mean(all_latencies));
    fprintf('  Overall median latency: %.1f seconds\n', median(all_latencies));
    fprintf('  Latency standard deviation: %.1f seconds\n', std(all_latencies));
    fprintf('  Fastest navigation: %.1f s\n', min(all_latencies));
    fprintf('  Slowest navigation: %.1f s\n', max(all_latencies));
    
    % Latency distribution analysis
    fast_navigations = sum(all_latencies < 50);
    medium_navigations = sum(all_latencies >= 50 & all_latencies < 100);
    slow_navigations = sum(all_latencies >= 100);
    
    fprintf('  Navigation speed distribution:\n');
    fprintf('    Fast (<50s): %d (%.1f%%)\n', fast_navigations, ...
            fast_navigations/length(all_latencies)*100);
    fprintf('    Medium (50-100s): %d (%.1f%%)\n', medium_navigations, ...
            medium_navigations/length(all_latencies)*100);
    fprintf('    Slow (>=100s): %d (%.1f%%)\n', slow_navigations, ...
            slow_navigations/length(all_latencies)*100);
end

% Quality assessment
fprintf('\nQuality Assessment:\n');
if processed_files > 0
    quality_score = (processed_files / length(files)) * 100;
    fprintf('  Data quality score: %.1f%% (based on successful file processing)\n', quality_score);
    
    if ~isempty(all_success_rates)
        mean_success = mean(all_success_rates) * 100;
        if mean_success > 70
            fprintf('  Performance quality: Excellent (%.1f%% mean success)\n', mean_success);
        elseif mean_success > 50
            fprintf('  Performance quality: Good (%.1f%% mean success)\n', mean_success);
        else
            fprintf('  Performance quality: Needs improvement (%.1f%% mean success)\n', mean_success);
        end
    end
end

% Recommendations
fprintf('\nRecommendations:\n');
if failed_files > 0
    fprintf('  - Investigate %d failed files for data integrity issues\n', failed_files);
end
if ~isempty(all_success_rates) && mean(all_success_rates) < 0.7
    fprintf('  - Consider parameter tuning to improve success rates\n');
end
if ~isempty(all_latencies) && mean(all_latencies) > 100
    fprintf('  - Analyze high latency cases for optimization opportunities\n');
end

exit;
EOF

log_verbose "MATLAB analysis script created successfully"

# Change to project root for proper relative paths
log_verbose "Changing to project root directory for MATLAB execution"
cd "$PROJECT_ROOT"

# Run MATLAB analysis
log_verbose "Starting MATLAB analysis of result files"
log_info "Analyzing results with MATLAB..." >&2

# Execute MATLAB with comprehensive error handling
if [[ $VERBOSE -eq 1 ]]; then
    log_verbose "Running MATLAB in verbose mode with detailed output capture"
    matlab -nodisplay -nosplash < "$TEMP_SCRIPT" 2>&1 | \
    tee -a "$LOG_FILE" | \
    grep -v "^>>" | \
    tail -n +11 >> "$REPORT_FILE"
else
    matlab -nodisplay -nosplash < "$TEMP_SCRIPT" 2>/dev/null | \
    grep -v "^>>" | \
    tail -n +11 >> "$REPORT_FILE"
fi

MATLAB_EXIT_CODE=${PIPESTATUS[0]}
log_verbose "MATLAB execution completed with exit code: $MATLAB_EXIT_CODE"

if [[ $MATLAB_EXIT_CODE -ne 0 ]]; then
    log_verbose "MATLAB execution failed, but continuing with available output"
    echo "" >> "$REPORT_FILE"
    echo "Warning: MATLAB analysis encountered errors during execution" >> "$REPORT_FILE"
fi

# Clean up temporary files
log_verbose "Cleaning up temporary MATLAB script: $TEMP_SCRIPT"
rm -f "$TEMP_SCRIPT"

# Add execution metadata to report
log_verbose "Adding execution metadata to report"
echo "" >> "$REPORT_FILE"
echo "Report Generation Info:" >> "$REPORT_FILE"
echo "Generated by: $(basename "$0")" >> "$REPORT_FILE"
echo "Execution time: $(date)" >> "$REPORT_FILE"
echo "Analysis duration: $((SECONDS)) seconds" >> "$REPORT_FILE"

# Display report to stdout (primary output)
log_verbose "Displaying generated report to stdout"
cat "$REPORT_FILE"

# Clean up report file (output goes to stdout)
log_verbose "Cleaning up temporary report file: $REPORT_FILE"
rm -f "$REPORT_FILE"

log_verbose "Results report generation completed successfully"

if [[ $VERBOSE -eq 1 ]]; then
    log_verbose "Verbose logging completed. Log file saved to: $LOG_FILE"
fi