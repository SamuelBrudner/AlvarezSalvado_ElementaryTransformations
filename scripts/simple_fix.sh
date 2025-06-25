#!/bin/bash

# Simple fix utility script for common troubleshooting tasks
# Relocated to scripts/ directory with enhanced verbose logging support

# Initialize verbose mode
VERBOSE=0
LOG_FILE=""

# Function to display usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Simple fix utility for common troubleshooting tasks in setup_smoke_plume_config.sh"
    echo ""
    echo "Options:"
    echo "  -v, --verbose    Enable verbose logging with detailed trace output"
    echo "  -h, --help       Display this help message"
    echo ""
    echo "This script applies several fixes to setup_smoke_plume_config.sh:"
    echo "  - Fix wrapper script variable expansion"
    echo "  - Fix timeout command syntax"
    echo "  - Fix mode display variable checking"
    echo "  - Ensure N_SAMPLE_FRAMES is properly set"
}

# Function for verbose logging
log_verbose() {
    if [[ $VERBOSE -eq 1 ]]; then
        local message="[$(date '+%Y-%m-%d %H:%M:%S')] simple_fix.sh: $1"
        echo "$message"
        if [[ -n "$LOG_FILE" && -w "$(dirname "$LOG_FILE")" ]]; then
            echo "$message" >> "$LOG_FILE"
        fi
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=1
            # Create logs directory if it doesn't exist
            if [[ ! -d "logs" ]]; then
                mkdir -p logs
            fi
            # Set log file path
            LOG_FILE="logs/simple_fix_$(date '+%Y%m%d_%H%M%S').log"
            log_verbose "Verbose mode enabled, logging to $LOG_FILE"
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option '$1'"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Main script execution
log_verbose "Starting simple fix utility"

# Check if target file exists
if [[ ! -f "setup_smoke_plume_config.sh" ]]; then
    echo "Error: setup_smoke_plume_config.sh not found in current directory"
    log_verbose "Error: Target file setup_smoke_plume_config.sh not found"
    exit 1
fi

log_verbose "Target file setup_smoke_plume_config.sh found, applying fixes"

# Fix 1: Find the line that creates the wrapper script and ensure variables are expanded
log_verbose "Applying fix 1: Updating wrapper script heredoc to allow variable expansion"
sed -i "s/cat > \"\$TEMP_DIR\/run_analysis.sh\" << 'WRAPPER_EOF'/cat > \"\$TEMP_DIR\/run_analysis.sh\" << WRAPPER_EOF/" setup_smoke_plume_config.sh
if [[ $? -eq 0 ]]; then
    log_verbose "Fix 1 applied successfully"
else
    log_verbose "Warning: Fix 1 may not have been applied (target pattern not found or already fixed)"
fi

# Fix 2: Fix the timeout line to ensure TIMEOUT_SECONDS is expanded
log_verbose "Applying fix 2: Correcting timeout command variable expansion"
sed -i 's/timeout \${TIMEOUT_SECONDS}s/timeout ${TIMEOUT_SECONDS}s/g' setup_smoke_plume_config.sh
if [[ $? -eq 0 ]]; then
    log_verbose "Fix 2 applied successfully"
else
    log_verbose "Warning: Fix 2 may not have been applied (target pattern not found or already fixed)"
fi

# Fix 3: Fix the mode display to use the correct variable check
log_verbose "Applying fix 3: Fixing mode display variable checking syntax"
sed -i 's/Mode: \$(if \[ \"\$RUN_ANALYSIS\" == \"quick\" \]/Mode: $(if [ "$RUN_ANALYSIS" == "quick" ]/g' setup_smoke_plume_config.sh
if [[ $? -eq 0 ]]; then
    log_verbose "Fix 3 applied successfully"
else
    log_verbose "Warning: Fix 3 may not have been applied (target pattern not found or already fixed)"
fi

# Fix 4: Ensure N_SAMPLE_FRAMES is properly set in the MATLAB script
log_verbose "Applying fix 4: Ensuring N_SAMPLE_FRAMES variable is properly set with default value"
sed -i "s/n_samples = min('\$N_SAMPLE_FRAMES', n_frames);/n_samples = min(${N_SAMPLE_FRAMES:-100}, n_frames);/" setup_smoke_plume_config.sh
if [[ $? -eq 0 ]]; then
    log_verbose "Fix 4 applied successfully"
else
    log_verbose "Warning: Fix 4 may not have been applied (target pattern not found or already fixed)"
fi

log_verbose "All fixes have been processed"
echo "Simple fix applied!"
log_verbose "Simple fix utility completed successfully"

# Log completion summary if verbose mode is enabled
if [[ $VERBOSE -eq 1 ]]; then
    echo "Verbose logging completed. Log file: $LOG_FILE"
fi