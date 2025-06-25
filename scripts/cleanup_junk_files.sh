#!/bin/bash
# cleanup_junk_files.sh - Remove temporary files from debugging
# Enhanced with verbose logging support per CLI requirements

# Initialize verbose logging
VERBOSE=0

# Parse command-line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -v|--verbose)
                VERBOSE=1
                shift
                ;;
            -h|--help)
                echo "Usage: $0 [-v|--verbose] [-h|--help]"
                echo ""
                echo "Options:"
                echo "  -v, --verbose    Enable detailed trace output"
                echo "  -h, --help       Show this help message"
                echo ""
                echo "Description:"
                echo "  Utility script for cleaning up temporary and junk files from the project directory."
                echo "  Removes temporary MATLAB scripts, configs, results, and plots generated during debugging."
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                echo "Use -h or --help for usage information"
                exit 1
                ;;
        esac
    done
}

# Verbose logging function
log_verbose() {
    [[ $VERBOSE -eq 1 ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] cleanup_junk_files: $1"
}

# Create logs directory if it doesn't exist
ensure_logs_dir() {
    if [[ ! -d "logs" ]]; then
        log_verbose "Creating logs directory"
        mkdir -p logs
    fi
}

# Parse arguments
parse_args "$@"

# Ensure logs directory exists if verbose mode is enabled
[[ $VERBOSE -eq 1 ]] && ensure_logs_dir

log_verbose "Starting cleanup process"
log_verbose "Verbose logging enabled - trace output will be shown"

echo "=== Cleaning Up Temporary Files ==="

# Remove temporary MATLAB scripts
log_verbose "Beginning removal of temporary MATLAB scripts"
echo "Removing temporary scripts..."

log_verbose "Removing temporary dimension check scripts"
rm -f check_all_dimensions.m
rm -f check_dimensions.m
rm -f check_arena_calc.m
rm -f check_json_raw.m

log_verbose "Removing temporary config fix scripts"
rm -f fix_configs_direct.m
rm -f fix_smoke_coordinates.m
rm -f fix_smoke_like_crimaldi.m
rm -f generate_minimal_configs.m
rm -f fix_existing_configs.m

log_verbose "Removing temporary smoke-related scripts using wildcard patterns"
rm -f fix_smoke_*.m
rm -f smoke_test*.m
rm -f test_smoke*.m
rm -f visualize_smoke*.m
rm -f run_smoke_*.m
rm -f check_smoke_*.m

# Remove backup and temporary configs
log_verbose "Beginning removal of temporary configuration files"
echo "Removing temporary configs..."

log_verbose "Removing fixed/corrected config variants from configs/plumes/"
rm -f configs/plumes/*_FIXED.json
rm -f configs/plumes/*_CORRECT.json
rm -f configs/plumes/*_minimal.json
rm -f configs/plumes/*.backup*

log_verbose "Removing temporary smoke shell scripts"
rm -f smoke_*.sh

# Remove temporary result files
log_verbose "Beginning removal of temporary result files"
echo "Removing temporary results..."

log_verbose "Removing test/temporary result files from results/ directory"
rm -f results/smoke_nav_results_TEST.mat
rm -f results/smoke_nav_results_PROPER.mat
rm -f results/smoke_nav_results_CORRECTED.mat
rm -f results/smoke_nav_results_CRIMALDI_STYLE.mat

log_verbose "Removing nav_results_0001.mat (keeping only 0000)"
rm -f results/nav_results_0001.mat  # Keep only 0000

# Remove broken adaptive model
log_verbose "Removing broken adaptive model files from Code/ directory"
rm -f Code/Elifenavmodel_bilateral_adaptive.m
rm -f Code/Elifenavmodel_bilateral_smoke_temp.m

# Remove temporary plots
log_verbose "Beginning removal of temporary plot files"
rm -f smoke_*.png

log_verbose "Removing comparison plot (keeping if desired for reference)"
rm -f both_plumes_comparison.png  # Keep if you want

log_verbose "Cleanup operations completed successfully"

echo ""
echo "✓ Cleanup complete!"
echo ""
echo "Essential files remaining:"
echo "  configs/plumes/crimaldi_10cms_bounded.json"
echo "  configs/plumes/smoke_1a_backgroundsubtracted.json"
echo "  generate_clean_configs.m"
echo "  plot_both_plumes.m"
echo "  run_both_plumes_test.m"
echo "  results/nav_results_0000.mat"
echo "  results/smoke_nav_results_1000.mat (if exists)"

log_verbose "Script execution completed - all temporary files cleaned up"
[[ $VERBOSE -eq 1 ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] cleanup_junk_files: Verbose logging session ended"