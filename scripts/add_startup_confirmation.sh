#!/bin/bash

# Add a startup message right at the beginning of the MATLAB script
# Enhanced with verbose logging support per Section 7.2 CLI requirements

# Initialize verbose logging flag
VERBOSE=0

# Parse command line arguments for verbose flag
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        *)
            # Pass through other arguments (maintain CLI compatibility)
            break
            ;;
    esac
done

# Verbose logging function
log_verbose() {
    [[ $VERBOSE -eq 1 ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] add_startup_confirmation.sh: $1"
}

# Create logs directory if it doesn't exist and verbose logging is enabled
[[ $VERBOSE -eq 1 ]] && mkdir -p logs

log_verbose "Starting MATLAB startup confirmation injection process"
log_verbose "Target script: setup_smoke_plume_config.sh"

# Check if target script exists
if [[ ! -f "setup_smoke_plume_config.sh" ]]; then
    log_verbose "ERROR: Target script 'setup_smoke_plume_config.sh' not found in current directory"
    echo "Error: setup_smoke_plume_config.sh not found" >&2
    exit 1
fi

log_verbose "Target script found, proceeding with Perl-based injection"

# Add a startup message right at the beginning of the MATLAB script
log_verbose "Executing Perl script to inject startup confirmation messages"
perl -i -pe '
# Find the beginning of the MATLAB script and add startup confirmation
if (/^% Comprehensive analysis script with error handling$/) {
    $_ .= "fprintf('"'"'\\n=== MATLAB STARTED SUCCESSFULLY at %s ===\\n'"'"', datestr(now));\n";
    $_ .= "fprintf('"'"'Running from: %s\\n'"'"', pwd);\n";
    $_ .= "fprintf('"'"'MATLAB version: %s\\n\\n'"'"', version);\n";
}

# Also add a message right before the try block
if (/^try$/ && !$done_try) {
    print "fprintf('"'"'Starting analysis try block...\\n'"'"');\n";
    $done_try = 1;
}
' setup_smoke_plume_config.sh

# Check if Perl execution was successful
if [[ $? -eq 0 ]]; then
    log_verbose "Perl injection completed successfully"
    echo "Startup confirmation added!"
    log_verbose "Process completed successfully"
else
    log_verbose "ERROR: Perl injection failed with exit code $?"
    echo "Error: Failed to add startup confirmation" >&2
    exit 1
fi

log_verbose "Script execution finished"