#!/bin/bash

# add_matlab_logging.sh - Enhanced MATLAB logging injection script
# 
# Purpose: Adds comprehensive logging capabilities to MATLAB sections in pipeline scripts
# Usage: ./scripts/add_matlab_logging.sh [OPTIONS] [TARGET_SCRIPT]
# 
# This script uses Perl to inject timestamped logging functions and enhanced error reporting
# into MATLAB code sections within pipeline scripts, particularly setup_smoke_plume_config.sh
#
# Enhanced Features:
# - Verbose logging support with -v/--verbose flags
# - Structured output to logs/ directory
# - Timestamped execution traces
# - Enhanced error reporting and debugging capabilities

# Initialize verbose logging flag
VERBOSE=0

# Script metadata
SCRIPT_NAME="$(basename "$0")"
LOG_PREFIX="[$SCRIPT_NAME]"

# Logging function
log_verbose() {
    if [[ $VERBOSE -eq 1 ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $LOG_PREFIX $1"
    fi
}

# Error logging function
log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $LOG_PREFIX ERROR: $1" >&2
}

# Info logging function (always shown)
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $LOG_PREFIX $1"
}

# Display usage information
show_usage() {
    cat << EOF
Usage: $SCRIPT_NAME [OPTIONS] [TARGET_SCRIPT]

Add comprehensive MATLAB logging capabilities to pipeline scripts.

OPTIONS:
    -v, --verbose       Enable verbose logging output
    -h, --help         Show this help message

ARGUMENTS:
    TARGET_SCRIPT      Target script to modify (default: setup_smoke_plume_config.sh)

EXAMPLES:
    $SCRIPT_NAME                                    # Apply to default target
    $SCRIPT_NAME --verbose                          # With verbose output
    $SCRIPT_NAME -v > logs/matlab_logging.log      # Redirect verbose to log file
    $SCRIPT_NAME custom_script.sh                  # Apply to custom target

DESCRIPTION:
    This script enhances MATLAB sections in pipeline scripts by injecting:
    - Timestamped logging functions
    - Performance timing for HDF5 operations
    - Enhanced error reporting
    - Frame sampling progress tracking
    - Detailed execution traces

OUTPUT:
    The target script is modified in-place with comprehensive logging capabilities.
    When verbose mode is enabled, detailed execution steps are displayed.

EOF
}

# Parse command line arguments
TARGET_SCRIPT="setup_smoke_plume_config.sh"

while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=1
            log_verbose "Verbose logging enabled"
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        -*)
            log_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
        *)
            TARGET_SCRIPT="$1"
            shift
            ;;
    esac
done

# Validate environment and dependencies
log_verbose "Starting MATLAB logging enhancement process"
log_verbose "Target script: $TARGET_SCRIPT"

# Check if target script exists
if [[ ! -f "$TARGET_SCRIPT" ]]; then
    log_error "Target script not found: $TARGET_SCRIPT"
    exit 1
fi

log_verbose "Target script found and accessible"

# Check if perl is available
if ! command -v perl &> /dev/null; then
    log_error "Perl is required but not installed or not in PATH"
    exit 1
fi

log_verbose "Perl interpreter available"

# Create logs directory if it doesn't exist
if [[ ! -d "logs" ]]; then
    log_verbose "Creating logs/ directory for future log output"
    mkdir -p logs
fi

# Backup original file
BACKUP_FILE="${TARGET_SCRIPT}.backup.$(date +%Y%m%d_%H%M%S)"
log_verbose "Creating backup: $BACKUP_FILE"
cp "$TARGET_SCRIPT" "$BACKUP_FILE"

# Apply MATLAB logging enhancements using Perl
log_verbose "Applying MATLAB logging enhancements..."
log_info "Injecting comprehensive logging into MATLAB sections"

perl -i -pe '
# Add timestamp function at the beginning of MATLAB script
if (/^    % Change to project root to ensure all relative paths work$/) {
    print "    % Helper function for timestamped logging\n";
    print "    log_msg = @(msg) fprintf('"'"'[%s] %s\\n'"'"', datestr(now, '"'"'HH:MM:SS'"'"'), msg);\n";
    print "    \n";
    print "    log_msg('"'"'=== MATLAB Analysis Started ==='"'"');\n";
}

# Add logging after each major step
s/fprintf\('"'"'Changed to project root: %s\\n'"'"', pwd\);/log_msg(sprintf('"'"'Changed to project root: %s'"'"', pwd));/;

s/fprintf\('"'"'Added Code directory to path\\n'"'"'\);/log_msg('"'"'Added Code directory to path'"'"');/;

s/fprintf\('"'"'\\nReading smoke config from JSON\.\.\.\\n'"'"'\);/log_msg('"'"'Reading smoke config from JSON...'"'"');/;

s/fprintf\('"'"'Configuration:\\n'"'"'\);/log_msg('"'"'Configuration loaded successfully'"'"');/;

s/fprintf\('"'"'\\nChecking HDF5 file\.\.\.\\n'"'"'\);/log_msg('"'"'Checking HDF5 file accessibility...'"'"');/;

# Add timing for h5info
if (/info = h5info\(plume_file\);/) {
    print "        log_msg('"'"'Calling h5info - this may take time for large files...'"'"');\n";
    print "        h5_start = tic;\n";
    $_ = "        info = h5info(plume_file);\n";
    $_ .= "        log_msg(sprintf('"'"'h5info completed in %.1f seconds'"'"', toc(h5_start)));\n";
}

# Add timing for dataset info
if (/ds_info = h5info\(plume_file, dataset_name\);/) {
    print "        log_msg(sprintf('"'"'Reading dataset info: %s'"'"', dataset_name));\n";
    print "        ds_start = tic;\n";
    $_ = "        ds_info = h5info(plume_file, dataset_name);\n";
    $_ .= "        log_msg(sprintf('"'"'Dataset info retrieved in %.1f seconds'"'"', toc(ds_start)));\n";
}

# Add logging for frame sampling
s/fprintf\('"'"'\\nSampling frames\.\.\.\\n'"'"'\);/log_msg(sprintf('"'"'Starting frame sampling (%d frames from %d total)...'"'"', n_samples, n_frames));/;

# Add logging in the frame reading loop
if (/frame = h5read\(plume_file, dataset_name,/) {
    print "        if i == 1\n";
    print "            log_msg('"'"'Reading first frame - timing this operation...'"'"');\n";
    print "            frame_start = tic;\n";
    print "        end\n";
    print "        \n";
    $_ = "        frame = h5read(plume_file, dataset_name, ...\n";
    $_ .= "                       [1 1 sample_indices(i)], [inf inf 1]);\n";
    $_ .= "        \n";
    $_ .= "        if i == 1\n";
    $_ .= "            log_msg(sprintf('"'"'First frame read in %.1f seconds'"'"', toc(frame_start)));\n";
    $_ .= "        end\n";
}

# Add final success logging
if (/fprintf\('"'"'\\n✓ Analysis complete\\n'"'"'\);/) {
    print "    log_msg('"'"'Analysis completed successfully!'"'"');\n";
}

# Add logging to error handler
if (/fprintf\('"'"'\\n\\nERROR in MATLAB analysis:\\n'"'"'\);/) {
    print "    log_msg('"'"'ERROR occurred - see details below'"'"');\n";
}
' "$TARGET_SCRIPT"

# Check if Perl command succeeded
if [[ $? -eq 0 ]]; then
    log_info "MATLAB logging capabilities successfully added to $TARGET_SCRIPT"
    log_verbose "Enhanced logging features include:"
    log_verbose "  - Timestamped log_msg() function"
    log_verbose "  - Performance timing for HDF5 operations"
    log_verbose "  - Enhanced error reporting"
    log_verbose "  - Frame sampling progress tracking"
    log_verbose "  - Detailed execution traces"
    
    # Provide usage examples
    if [[ $VERBOSE -eq 1 ]]; then
        log_verbose ""
        log_verbose "Enhanced MATLAB sections will now provide:"
        log_verbose "  - Automatic timestamping of all operations"
        log_verbose "  - Performance metrics for file operations"
        log_verbose "  - Clear progress indicators"
        log_verbose "  - Enhanced error context"
        log_verbose ""
        log_verbose "Backup created: $BACKUP_FILE"
    fi
    
    log_info "Enhancement complete! The script now includes comprehensive MATLAB logging."
else
    log_error "Failed to apply MATLAB logging enhancements"
    log_error "Restoring original file from backup..."
    cp "$BACKUP_FILE" "$TARGET_SCRIPT"
    exit 1
fi

log_verbose "MATLAB logging enhancement process completed successfully"