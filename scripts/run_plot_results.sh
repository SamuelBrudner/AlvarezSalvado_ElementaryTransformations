#!/bin/bash
# run_plot_results.sh - Generate figures from navigation results
#
# Usage: ./run_plot_results.sh [results_file.mat] [--verbose|-v]
#        Default: results/nav_results_0000.mat
#
# Creates 4 PNG figures in the results directory
#
# Options:
#   -v, --verbose    Enable detailed trace output for debugging
#
# Examples:
#   ./run_plot_results.sh                                    # Use default file
#   ./run_plot_results.sh results/custom_results.mat        # Specify results file
#   ./run_plot_results.sh --verbose                         # Verbose with default file
#   ./run_plot_results.sh results/data.mat -v               # Verbose with custom file
#   ./run_plot_results.sh -v > logs/plotting_$(date +%Y%m%d_%H%M%S).log 2>&1  # Log to file

# Initialize variables
VERBOSE=0
RESULT_FILE=""

# Function to display usage information
show_usage() {
    echo "Usage: $0 [results_file.mat] [--verbose|-v]"
    echo ""
    echo "Generate PNG figures from MATLAB navigation results"
    echo ""
    echo "Arguments:"
    echo "  results_file.mat    Path to results file (default: results/nav_results_0000.mat)"
    echo ""
    echo "Options:"
    echo "  -v, --verbose       Enable detailed trace output for debugging"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                           # Use default file"
    echo "  $0 results/experiment_001.mat                # Specify results file"
    echo "  $0 --verbose                                 # Verbose with default file"
    echo "  $0 results/data.mat -v                       # Verbose with custom file"
    echo ""
    echo "Output: Creates PNG files in results/ directory"
    echo "        - *_trajectories.png    Agent trajectory visualization"
    echo "        - *_statistics.png      Performance statistics"
    echo "        - *_heatmap.png         Spatial search distribution"
    echo "        - *_temporal.png        Time-series analysis"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=1
            [[ $VERBOSE -eq 1 ]] && echo "[$(date)] run_plot_results.sh: Verbose logging enabled"
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        -*)
            echo "Error: Unknown option $1"
            echo ""
            show_usage
            exit 1
            ;;
        *)
            if [[ -z "$RESULT_FILE" ]]; then
                RESULT_FILE="$1"
                [[ $VERBOSE -eq 1 ]] && echo "[$(date)] run_plot_results.sh: Results file specified: $RESULT_FILE"
            else
                echo "Error: Multiple result files specified"
                echo ""
                show_usage
                exit 1
            fi
            shift
            ;;
    esac
done

# Set default result file if not specified
if [[ -z "$RESULT_FILE" ]]; then
    RESULT_FILE="results/nav_results_0000.mat"
    [[ $VERBOSE -eq 1 ]] && echo "[$(date)] run_plot_results.sh: Using default results file: $RESULT_FILE"
fi

[[ $VERBOSE -eq 1 ]] && echo "[$(date)] run_plot_results.sh: Starting figure generation process"
[[ $VERBOSE -eq 1 ]] && echo "[$(date)] run_plot_results.sh: Target results file: $RESULT_FILE"

# Verify results file exists
if [ ! -f "$RESULT_FILE" ]; then
    echo "Error: $RESULT_FILE not found"
    [[ $VERBOSE -eq 1 ]] && echo "[$(date)] run_plot_results.sh: File validation failed - results file does not exist"
    [[ $VERBOSE -eq 1 ]] && echo "[$(date)] run_plot_results.sh: Checked path: $(pwd)/$RESULT_FILE"
    [[ $VERBOSE -eq 1 ]] && echo "[$(date)] run_plot_results.sh: Available files in results/: $(ls -la results/ 2>/dev/null || echo 'results/ directory not found')"
    exit 1
fi

[[ $VERBOSE -eq 1 ]] && echo "[$(date)] run_plot_results.sh: Results file validation successful"
[[ $VERBOSE -eq 1 ]] && echo "[$(date)] run_plot_results.sh: File size: $(stat -f%z "$RESULT_FILE" 2>/dev/null || stat -c%s "$RESULT_FILE" 2>/dev/null || echo 'unknown') bytes"

# Ensure results directory exists for output figures
if [ ! -d "results" ]; then
    [[ $VERBOSE -eq 1 ]] && echo "[$(date)] run_plot_results.sh: Creating results directory for output figures"
    mkdir -p results
fi

echo "Creating figures for $RESULT_FILE..."
[[ $VERBOSE -eq 1 ]] && echo "[$(date)] run_plot_results.sh: Launching MATLAB for figure generation"
echo ""

# Prepare MATLAB command with error handling
[[ $VERBOSE -eq 1 ]] && echo "[$(date)] run_plot_results.sh: Preparing MATLAB execution environment"
[[ $VERBOSE -eq 1 ]] && echo "[$(date)] run_plot_results.sh: MATLAB command: plot_results('$RESULT_FILE')"

# Run MATLAB plotting function with enhanced error handling
[[ $VERBOSE -eq 1 ]] && echo "[$(date)] run_plot_results.sh: Executing MATLAB plotting function"
matlab_output=$(matlab -nodisplay -nosplash -r "
try
    fprintf('MATLAB: Starting plot generation for %s\n', '$RESULT_FILE');
    plot_results('$RESULT_FILE');
    fprintf('MATLAB: Plot generation completed successfully\n');
catch ME
    fprintf('MATLAB Error: %s\n', ME.message);
    fprintf('MATLAB Error Stack:\n');
    for i = 1:length(ME.stack)
        fprintf('  File: %s, Function: %s, Line: %d\n', ME.stack(i).file, ME.stack(i).name, ME.stack(i).line);
    end
    exit(1);
end
exit(0);
" 2>&1)

# Process MATLAB output
matlab_exit_code=$?
filtered_output=$(echo "$matlab_output" | grep -v "^>>")

if [[ $VERBOSE -eq 1 ]]; then
    echo "[$(date)] run_plot_results.sh: MATLAB execution completed with exit code: $matlab_exit_code"
    echo "[$(date)] run_plot_results.sh: MATLAB output:"
    echo "$filtered_output" | sed 's/^/[$(date)] MATLAB: /'
fi

# Display only clean output in non-verbose mode
if [[ $VERBOSE -eq 0 ]]; then
    echo "$filtered_output"
fi

# Check MATLAB execution success
if [[ $matlab_exit_code -ne 0 ]]; then
    echo ""
    echo "Error: MATLAB execution failed"
    [[ $VERBOSE -eq 1 ]] && echo "[$(date)] run_plot_results.sh: MATLAB plotting function failed with exit code $matlab_exit_code"
    exit 1
fi

[[ $VERBOSE -eq 1 ]] && echo "[$(date)] run_plot_results.sh: MATLAB execution successful"

# Verify output figures were created
[[ $VERBOSE -eq 1 ]] && echo "[$(date)] run_plot_results.sh: Checking for generated figure files"
generated_figures=$(find results/ -name "*.png" -newer "$RESULT_FILE" 2>/dev/null | wc -l)
[[ $VERBOSE -eq 1 ]] && echo "[$(date)] run_plot_results.sh: Found $generated_figures newly generated PNG files"

if [[ $generated_figures -eq 0 ]]; then
    [[ $VERBOSE -eq 1 ]] && echo "[$(date)] run_plot_results.sh: Warning - No new PNG files detected after MATLAB execution"
fi

echo ""
echo "To view figures:"
echo "  ls -la results/*.png"
[[ $VERBOSE -eq 1 ]] && echo "[$(date)] run_plot_results.sh: Listing available PNG files:"
[[ $VERBOSE -eq 1 ]] && ls -la results/*.png 2>/dev/null || [[ $VERBOSE -eq 1 ]] && echo "[$(date)] run_plot_results.sh: No PNG files found in results/ directory"

echo ""
echo "To display on cluster (if X11 forwarding enabled):"
echo "  display results/nav_results_0000_trajectories.png"

[[ $VERBOSE -eq 1 ]] && echo "[$(date)] run_plot_results.sh: Figure generation workflow completed successfully"
[[ $VERBOSE -eq 1 ]] && echo "[$(date)] run_plot_results.sh: Script execution finished"