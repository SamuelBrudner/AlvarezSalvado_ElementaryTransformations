#!/bin/bash
# create_results_report.sh - Generate results summary using MATLAB
#
# Usage: ./create_results_report.sh [output_report_file.txt]
#
# Analyzes all nav_results_*.mat files in results/ directory
# Creates a summary report with statistics

set -euo pipefail

# Directory constants
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$PROJECT_ROOT/results"
LOGS_DIR="$PROJECT_ROOT/logs"
CONFIG_DIR="$PROJECT_ROOT/configs"

# Check if required directories exist
for dir in "$RESULTS_DIR" "$LOGS_DIR"; do
    if [ ! -d "$dir" ]; then
        echo "[WARNING] Creating missing directory: $dir"
        mkdir -p "$dir"
    fi
done

# Check for MATLAB and required project directories
if ! command -v matlab &> /dev/null; then
    echo "[ERROR] MATLAB executable not found in PATH"
    exit 1
fi

if [ ! -d "$PROJECT_ROOT/Code" ]; then
    echo "[ERROR] Required Code directory is missing"
    exit 1
fi

# Set up report file
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
REPORT_FILE="${1:-$RESULTS_DIR/results_summary_${TIMESTAMP}.txt}"
LOG_FILE="$LOGS_DIR/report_gen_${TIMESTAMP}.log"

echo "[INFO] Creating report at $REPORT_FILE"

# Initialize report
echo "=== Navigation Model Results Summary ===" > "$REPORT_FILE"
echo "Generated: $(date)" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# Count result files with safeguards
if [ ! -d "$RESULTS_DIR" ]; then
    echo "[WARNING] Results directory not found"
    N_FILES=0
else
    N_FILES=$(find "$RESULTS_DIR" -name "nav_results_*.mat" -type f | wc -l | tr -d ' ')
fi

echo "Total result files: $N_FILES" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

if [ "$N_FILES" -eq 0 ]; then
    echo "No result files found in $RESULTS_DIR/" >> "$REPORT_FILE"
    echo "[INFO] Report contains no results - no .mat files found"
    cat "$REPORT_FILE"
    exit 0
fi

# Create MATLAB analysis script with improved robustness
cat > temp_analyze_all.m << 'EOF'
% Set up proper error reporting and path handling
try
    addpath(genpath('Code'));
    
    % Current directory check
    current_dir = pwd;
    fprintf('Working directory: %s\n', current_dir);
    
    % Define project paths correctly
    project_root = pwd;
    results_dir = fullfile(project_root, 'results');
    config_dir = fullfile(project_root, 'configs');
    
    % Load any necessary configurations from paths.json if it exists
    paths_config = struct();
    paths_json = fullfile(config_dir, 'paths.json');
    if exist(paths_json, 'file')
        try
            paths_config = jsondecode(fileread(paths_json));
            fprintf('Successfully loaded paths config from %s\n', paths_json);
        catch ME
            warning('Failed to load paths config: %s', ME.message);
        end
    else
        warning('paths.json not found at: %s', paths_json);
    end
    
    % Verify results directory
    if ~exist(results_dir, 'dir')
        warning('Results directory not found, creating: %s', results_dir);
        mkdir(results_dir);
    end
    
    % Find all result files
    files = dir(fullfile(results_dir, 'nav_results_*.mat'));
    
    fprintf('\nIndividual Results:\n');
    fprintf('==================\n');
    
    if isempty(files)
        fprintf('No result files found in %s\n', results_dir);
        exit(0);
    end
    
    all_success_rates = [];
    all_latencies = [];
    total_agents = 0;
    
    for i = 1:length(files)
        filepath = fullfile(files(i).folder, files(i).name);
        fprintf('\nFile: %s\n', files(i).name);
        
        try
            data = load(filepath);
            if ~isfield(data, 'out')
                warning('Missing "out" structure in %s', files(i).name);
                continue;
            end
            out = data.out;
            
            if ~isfield(out, 'x') || isempty(out.x)
                warning('Missing or empty trajectory data in %s', files(i).name);
                continue;
            end
            
            [n_samples, n_agents] = size(out.x);
            total_agents = total_agents + n_agents;
            
            fprintf('  Agents: %d\n', n_agents);
            
            if isfield(out, 'successrate')
                fprintf('  Success rate: %.1f%%\n', out.successrate * 100);
                all_success_rates(end+1) = out.successrate;
            else
                fprintf('  Success rate: N/A (not available in this file)\n');
            end
            
            if isfield(out, 'latency')
                successful = ~isnan(out.latency);
                n_success = sum(successful);
                if n_success > 0
                    mean_latency = mean(out.latency(successful));
                    fprintf('  Mean latency: %.1f s\n', mean_latency);
                    all_latencies = [all_latencies, out.latency(successful)];
                else
                    fprintf('  Mean latency: N/A (no successful trials)\n');
                end
            else
                fprintf('  Latency: N/A (not available in this file)\n');
            end
        catch ME
            fprintf('  Error processing %s: %s\n', files(i).name, getReport(ME, 'extended'));
        end
    end
    
    fprintf('\nAggregate Results:\n');
    fprintf('=================\n');
    fprintf('Total agents analyzed: %d\n', total_agents);
    
    if ~isempty(all_success_rates)
        mean_success = mean(all_success_rates);
        fprintf('Mean success rate: %.1f%%\n', mean_success * 100);
    else
        fprintf('Mean success rate: N/A (no data available)\n');
    end
    
    if ~isempty(all_latencies)
        mean_latency = mean(all_latencies);
        fprintf('Mean latency: %.1f s\n', mean_latency);
    else
        fprintf('Mean latency: N/A (no data available)\n');
    end
catch ME
    fprintf('ERROR in analysis: %s\n', getReport(ME, 'extended'));
    exit(1);
end
EOF

# Run MATLAB analysis with timeout and proper error handling
echo "[INFO] Analyzing results with MATLAB..." >&2
echo "[LOG] Running analysis with 5 minute timeout" >&2

# Run MATLAB with timeout and capture exit code
timeout 300s matlab -batch "try; temp_analyze_all; catch ME; fprintf('Error: %s\n', getReport(ME, 'extended')); exit(1); end; exit(0);" 2>&1 | grep -v "^>>" | tee -a "$LOG_FILE" >> "$REPORT_FILE"
MATLAB_EXIT=${PIPESTATUS[0]}

# Check if MATLAB execution was successful
if [ $MATLAB_EXIT -ne 0 ]; then
    if [ $MATLAB_EXIT -eq 124 ]; then
        echo "[ERROR] MATLAB analysis timed out after 5 minutes" | tee -a "$REPORT_FILE" "$LOG_FILE" >&2
    else
        echo "[ERROR] MATLAB analysis failed with exit code $MATLAB_EXIT" | tee -a "$REPORT_FILE" "$LOG_FILE" >&2
    fi
fi

# Validate report file
if [ ! -s "$REPORT_FILE" ]; then
    echo "[WARNING] Report file is empty or was not created properly" >&2
else
    REPORT_SIZE=$(wc -l < "$REPORT_FILE")
    if [ "$REPORT_SIZE" -lt 5 ]; then
        echo "[WARNING] Report file seems incomplete (only $REPORT_SIZE lines)" >&2
    fi
fi

# Clean up temporary files with error handling
if [ -f temp_analyze_all.m ]; then
    rm -f temp_analyze_all.m
fi

# Display report
echo "[SUCCESS] Report saved to: $REPORT_FILE" >&2
echo "" >&2
cat "$REPORT_FILE"