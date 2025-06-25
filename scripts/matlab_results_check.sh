#!/bin/bash
# matlab_results_check.sh - Check results using MATLAB (no Python needed)
# 
# Usage: ./scripts/matlab_results_check.sh [-v|--verbose] [results_file.mat]
# Default: results/nav_results_0000.mat
#
# This script provides a quick summary of navigation model results using MATLAB
# Options:
#   -v, --verbose    Enable verbose logging with detailed trace output

# Initialize variables
VERBOSE=0
RESULT_FILE=""
SCRIPT_NAME="matlab_results_check.sh"

# Function for verbose logging
log_verbose() {
    [[ $VERBOSE -eq 1 ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] $SCRIPT_NAME: $1"
}

# Function to log to file if logs directory exists
log_to_file() {
    if [[ $VERBOSE -eq 1 ]] && [[ -d "logs" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $SCRIPT_NAME: $1" >> "logs/matlab_results_check.log"
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=1
            log_verbose "Verbose logging enabled"
            log_to_file "Verbose logging enabled"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [-v|--verbose] [results_file.mat]"
            echo "Options:"
            echo "  -v, --verbose    Enable verbose logging with detailed trace output"
            echo "  -h, --help       Show this help message"
            echo ""
            echo "Default results file: results/nav_results_0000.mat"
            exit 0
            ;;
        -*)
            echo "Error: Unknown option $1" >&2
            echo "Use -h or --help for help" >&2
            exit 1
            ;;
        *)
            if [[ -z "$RESULT_FILE" ]]; then
                RESULT_FILE="$1"
            else
                echo "Error: Too many arguments" >&2
                echo "Use -h or --help for help" >&2
                exit 1
            fi
            shift
            ;;
    esac
done

# Set default result file if not provided
RESULT_FILE="${RESULT_FILE:-results/nav_results_0000.mat}"

log_verbose "Starting MATLAB results validation"
log_to_file "Starting MATLAB results validation"
log_verbose "Target results file: $RESULT_FILE"
log_to_file "Target results file: $RESULT_FILE"

# Check if results file exists
if [ ! -f "$RESULT_FILE" ]; then
    echo "Error: $RESULT_FILE not found"
    log_verbose "ERROR: Results file not found: $RESULT_FILE"
    log_to_file "ERROR: Results file not found: $RESULT_FILE"
    exit 1
fi

log_verbose "Results file found, proceeding with MATLAB analysis"
log_to_file "Results file found, proceeding with MATLAB analysis"

echo "=== MATLAB Results Check ==="
echo "File: $RESULT_FILE"
echo ""

log_verbose "Launching MATLAB for results analysis"
log_to_file "Launching MATLAB for results analysis"

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

MATLAB_EXIT_CODE=$?

log_verbose "MATLAB analysis completed with exit code: $MATLAB_EXIT_CODE"
log_to_file "MATLAB analysis completed with exit code: $MATLAB_EXIT_CODE"

echo ""
echo "For visualizations, run MATLAB interactively and use:"
echo "  load('$RESULT_FILE')"
echo "  plot(out.x, out.y)"

log_verbose "Results validation completed successfully"
log_to_file "Results validation completed successfully"

exit $MATLAB_EXIT_CODE