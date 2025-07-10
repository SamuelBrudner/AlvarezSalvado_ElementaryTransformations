#!/bin/bash
# run_plot_results.sh - Generate figures from navigation results
#
# Usage: ./run_plot_results.sh [results_file.mat]
#        Default: results/nav_results_0000.mat
#
# Creates PDF figures in the results directory

set -euo pipefail

# Directory constants
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$PROJECT_ROOT/results"

# Check if required directories exist
if [ ! -d "$RESULTS_DIR" ]; then
    echo "[WARNING] Creating missing results directory"
    mkdir -p "$RESULTS_DIR"
fi

# Check if Code directory exists
if [ ! -d "$PROJECT_ROOT/Code" ]; then
    echo "[ERROR] Required Code directory is missing"
    exit 1
fi

# Get result file path
RESULT_FILE="${1:-$RESULTS_DIR/nav_results_0000.mat}"

if [ ! -f "$RESULT_FILE" ]; then
    echo "[ERROR] Result file not found: $RESULT_FILE"
    exit 1
fi

# Check if result file is readable
if [ ! -r "$RESULT_FILE" ]; then
    echo "[ERROR] Result file is not readable: $RESULT_FILE"
    exit 1
fi

echo "[INFO] Creating figures for $RESULT_FILE..."

# Create a timestamp for logs
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$RESULTS_DIR/plot_results_${TIMESTAMP}.log"

# Run MATLAB plotting function with timeout
echo "[LOG] Running MATLAB plotting with 5 minute timeout"
echo "[LOG] Command: matlab -batch \"try; addpath(genpath('Code')); plot_results('$RESULT_FILE'); catch ME; fprintf('Error: %s\\n', getReport(ME, 'extended')); exit(1); end; exit(0);\""

# Run with a timeout and capture exit code
timeout 300s matlab -batch "try; addpath(genpath('Code')); plot_results('$RESULT_FILE'); catch ME; fprintf('Error: %s\n', getReport(ME, 'extended')); exit(1); end; exit(0);" 2>&1 | grep -v "^>>" | tee "$LOG_FILE"
MATLAB_EXIT=${PIPESTATUS[0]}

if [ $MATLAB_EXIT -ne 0 ]; then
    if [ $MATLAB_EXIT -eq 124 ]; then
        echo "[ERROR] MATLAB execution timed out after 5 minutes"
    else
        echo "[ERROR] MATLAB execution failed with exit code $MATLAB_EXIT"
    fi
    echo "[ERROR] Check log file: $LOG_FILE"
    exit $MATLAB_EXIT
fi

# Check if figures were actually generated
FIGURE_COUNT=$(find "$RESULTS_DIR" -name "*_$(basename "$RESULT_FILE" .mat)_*.pdf" -type f | wc -l)
if [ $FIGURE_COUNT -eq 0 ]; then
    echo "[WARNING] No figures appear to have been generated"
else
    echo "[SUCCESS] Generated $FIGURE_COUNT figures"
    echo ""
    echo "To view figures:"
    echo "  ls -la $RESULTS_DIR/*.pdf"
    echo ""
    echo "To display on cluster (if X11 forwarding enabled):"
    echo "  display $RESULTS_DIR/$(basename "$RESULT_FILE" .mat)_trajectories.pdf"
fi

exit 0