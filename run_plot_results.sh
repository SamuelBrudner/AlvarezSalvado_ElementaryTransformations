#!/bin/bash
# run_plot_results.sh - Generate figures from navigation results
#
# Usage: ./run_plot_results.sh [results_file.mat]
#        Default: results/nav_results_0000.mat
#
# Creates 4 PNG figures in the results directory

RESULT_FILE="${1:-results/nav_results_0000.mat}"

if [ ! -f "$RESULT_FILE" ]; then
    echo "Error: $RESULT_FILE not found"
    exit 1
fi

echo "Creating figures for $RESULT_FILE..."
echo ""

# Run MATLAB plotting function
matlab -nodisplay -nosplash -r "
try
    plot_results('$RESULT_FILE');
catch ME
    fprintf('Error: %s\n', ME.message);
    exit(1);
end
exit(0);
" 2>&1 | grep -v "^>>"

echo ""
echo "To view figures:"
echo "  ls -la results/*.png"
echo ""
echo "To display on cluster (if X11 forwarding enabled):"
echo "  display results/nav_results_0000_trajectories.png"