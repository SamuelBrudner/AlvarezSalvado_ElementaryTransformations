#!/bin/bash

# Fix analysis logic issues in the simulation pipeline
# This script corrects the conditional logic for analysis decision-making
# Moved to scripts/ directory as part of repository reorganization

# Initialize verbose logging
VERBOSE=0
SCRIPT_NAME="$(basename "$0")"
LOG_DIR="logs"

# Function to log verbose messages
log_verbose() {
    if [[ $VERBOSE -eq 1 ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$SCRIPT_NAME] $1" | tee -a "$LOG_DIR/fix_analysis_logic.log"
    fi
}

# Function to log regular messages
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$SCRIPT_NAME] $1"
    if [[ $VERBOSE -eq 1 ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$SCRIPT_NAME] $1" >> "$LOG_DIR/fix_analysis_logic.log"
    fi
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Fix analysis logic issues in the simulation pipeline.

OPTIONS:
    -v, --verbose    Enable verbose logging with detailed trace output
    -h, --help       Show this help message and exit

DESCRIPTION:
    This script fixes conditional logic issues in setup_smoke_plume_config.sh
    related to analysis decision-making. It corrects variable assignments
    and ensures proper handling of analysis options (skip, quick, full).

EXAMPLES:
    $0                 # Run with default logging
    $0 -v              # Run with verbose logging
    $0 --verbose       # Run with verbose logging (long form)

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option '$1'"
            echo "Use '$0 --help' for usage information."
            exit 1
            ;;
    esac
done

# Create logs directory if it doesn't exist
if [[ $VERBOSE -eq 1 ]]; then
    mkdir -p "$LOG_DIR"
    log_verbose "Logs directory created/verified at: $LOG_DIR"
fi

log_verbose "Starting analysis logic fix process"
log_verbose "Target file: setup_smoke_plume_config.sh"

# Check if target file exists
if [[ ! -f "setup_smoke_plume_config.sh" ]]; then
    log_info "ERROR: Target file 'setup_smoke_plume_config.sh' not found in current directory"
    log_verbose "Current directory: $(pwd)"
    log_verbose "Available files: $(ls -la)"
    exit 1
fi

log_verbose "Target file found, proceeding with logic fixes"

# First Perl operation: Fix the logic after user input
log_verbose "Step 1: Fixing analysis decision logic and variable assignments"
perl -i -pe '
# Fix the logic after user input
if (/^if \[\[.*RUN_ANALYSIS.*== "n" \]\]; then$/) {
    # This is where it checks for "n" to skip
    # Keep this section but make sure variables are set
    next;
}

# Make sure QUICK_ANALYSIS is properly set
if (/^elif \[\[.*RUN_ANALYSIS.*== "quick" \]\]; then$/) {
    $_ = "elif [[ \"\$RUN_ANALYSIS\" == \"quick\" ]]; then\n";
    $_ .= "    QUICK_ANALYSIS=1\n";
    $_ .= "    echo \"Will run quick analysis...\"\n";
}

# Fix the else clause
if (/^else$/ && $prev_line =~ /QUICK_ANALYSIS=1/) {
    $_ = "else\n";
    $_ .= "    QUICK_ANALYSIS=0\n";
    $_ .= "    echo \"Will run full analysis...\"\n";
}

$prev_line = $_;
' setup_smoke_plume_config.sh

if [[ $? -eq 0 ]]; then
    log_verbose "Step 1 completed successfully"
else
    log_info "ERROR: Step 1 failed"
    exit 1
fi

# Second Perl operation: Fix Step 4 to handle all three cases properly
log_verbose "Step 2: Fixing Step 4 analysis handling for all cases (skip/quick/full)"
perl -i -pe '
if (/^# Step 4: Run analysis based on user choice$/) {
    $_ = "# Step 4: Run analysis based on user choice\n";
    $_ .= "if [[ \"\$RUN_ANALYSIS\" == \"n\" ]]; then\n";
    $_ .= "    echo \"\"\n";
    $_ .= "    echo \"Step 4: Skipping analysis (using defaults)\"\n";
    $_ .= "    # Set default values\n";
    $_ .= "    width=1024\n";
    $_ .= "    height=1024\n";
    $_ .= "    frames=36000\n";
    $_ .= "    dataset=\"/dataset2\"\n";
    $_ .= "    data_min=0.0\n";
    $_ .= "    data_max=1.0\n";
    $_ .= "    data_mean=0.1\n";
    $_ .= "    data_std=0.1\n";
    $_ .= "    source_x_cm=0.0\n";
    $_ .= "    source_y_cm=0.0\n";
    $_ .= "    arena_width_cm=\$(awk \"BEGIN {printf \\\"%.1f\\\", 1024 * 0.15299877600979192 / 10}\")\n";
    $_ .= "    arena_height_cm=\$(awk \"BEGIN {printf \\\"%.1f\\\", 1024 * 0.15299877600979192 / 10}\")\n";
    $_ .= "    temporal_scale=4.0\n";
    $_ .= "    spatial_scale=0.207\n";
    $_ .= "    beta_suggestion=0.01\n";
    $_ .= "    normalized=1\n";
    $_ .= "else\n";
    # Skip to the original content
    while (<>) {
        last if /^    echo ""/;
    }
    $_ = "    echo \"\"\n";
}
' setup_smoke_plume_config.sh

if [[ $? -eq 0 ]]; then
    log_verbose "Step 2 completed successfully"
else
    log_info "ERROR: Step 2 failed"
    exit 1
fi

log_verbose "All Perl operations completed successfully"
log_verbose "Analysis logic fixes applied to setup_smoke_plume_config.sh"

# Verify the file was modified
if [[ -f "setup_smoke_plume_config.sh" ]]; then
    file_size=$(stat -c%s "setup_smoke_plume_config.sh" 2>/dev/null || stat -f%z "setup_smoke_plume_config.sh" 2>/dev/null || echo "unknown")
    log_verbose "Modified file size: $file_size bytes"
    log_verbose "File modification timestamp: $(stat -c%y "setup_smoke_plume_config.sh" 2>/dev/null || stat -f%m "setup_smoke_plume_config.sh" 2>/dev/null)"
fi

log_info "Analysis logic fixed!"
log_verbose "Fix analysis logic operation completed successfully"

# Log summary if verbose
if [[ $VERBOSE -eq 1 ]]; then
    log_verbose "=== OPERATION SUMMARY ==="
    log_verbose "Script: $SCRIPT_NAME"
    log_verbose "Target: setup_smoke_plume_config.sh"
    log_verbose "Operations: 2 Perl transformations"
    log_verbose "Status: SUCCESS"
    log_verbose "Log file: $LOG_DIR/fix_analysis_logic.log"
    log_verbose "=========================="
fi