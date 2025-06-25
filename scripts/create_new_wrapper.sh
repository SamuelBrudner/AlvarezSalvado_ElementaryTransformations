#!/bin/bash

# Script for creating new wrapper scripts in the project environment
# Relocated to scripts/ directory with enhanced verbose logging support
# 
# Usage: create_new_wrapper.sh [-v|--verbose] [target_file]
# 
# This script modifies setup_smoke_plume_config.sh by replacing the wrapper
# script creation section with an enhanced version that handles timeouts
# and errors for MATLAB analysis operations.

# Initialize verbose mode flag
VERBOSE=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [-v|--verbose] [target_file]"
            echo "Creates/updates wrapper scripts with enhanced timeout and error handling"
            echo ""
            echo "Options:"
            echo "  -v, --verbose    Enable verbose logging output"
            echo "  -h, --help       Show this help message"
            echo ""
            echo "If no target_file is specified, defaults to setup_smoke_plume_config.sh"
            exit 0
            ;;
        -*)
            echo "Unknown option $1" >&2
            exit 1
            ;;
        *)
            TARGET_FILE="$1"
            shift
            ;;
    esac
done

# Set default target file if not specified
TARGET_FILE="${TARGET_FILE:-setup_smoke_plume_config.sh}"

# Create logs directory if it doesn't exist
[[ $VERBOSE -eq 1 ]] && echo "[$(date)] create_new_wrapper.sh: Creating logs directory if needed"
mkdir -p logs

# Verbose logging function
log_verbose() {
    [[ $VERBOSE -eq 1 ]] && echo "[$(date)] create_new_wrapper.sh: $1"
}

# Main script execution with logging
log_verbose "Starting wrapper creation process"
log_verbose "Target file: $TARGET_FILE"
log_verbose "Verbose mode: enabled"

# Check if target file exists
if [[ ! -f "$TARGET_FILE" ]]; then
    log_verbose "ERROR: Target file '$TARGET_FILE' not found"
    echo "ERROR: Target file '$TARGET_FILE' not found" >&2
    exit 1
fi

log_verbose "Target file found, proceeding with wrapper replacement"
log_verbose "Using AWK to replace wrapper script section in $TARGET_FILE"

# Find where the wrapper script is created and replace it entirely
awk '
/# Create a wrapper script to handle timeouts and errors/ {
    print "    # Create a wrapper script to handle timeouts and errors"
    print "    cat > \"$TEMP_DIR/run_analysis.sh\" << WRAPPER_EOF"
    print "#!/bin/bash"
    print "MATLAB_SCRIPT=\"\\$1\""
    print ""
    print "echo \"Running MATLAB analysis with ${TIMEOUT_SECONDS}s timeout...\""
    print "if [ \"${RUN_ANALYSIS}\" == \"quick\" ]; then"
    print "    echo \"  Mode: QUICK (10 frames)\""
    print "else"
    print "    echo \"  Mode: FULL (100 frames)\""
    print "fi"
    print ""
    print "# Run MATLAB with timeout"
    print "if command -v timeout &> /dev/null; then"
    print "    timeout ${TIMEOUT_SECONDS}s matlab -nodisplay -nosplash -r \"run('"'"'\\${MATLAB_SCRIPT}'"'"')\" 2>&1"
    print "    MATLAB_EXIT=\\$?"
    print "else"
    print "    perl -e \"alarm ${TIMEOUT_SECONDS}; exec @ARGV\" matlab -nodisplay -nosplash -r \"run('"'"'\\${MATLAB_SCRIPT}'"'"')\" 2>&1"
    print "    MATLAB_EXIT=\\$?"
    print "fi"
    print ""
    print "if [ \\$MATLAB_EXIT -eq 124 ] || [ \\$MATLAB_EXIT -eq 142 ]; then"
    print "    echo \"\""
    print "    echo \"ERROR: MATLAB analysis timed out after ${TIMEOUT_SECONDS} seconds!\""
    print "    echo \"Possible causes:\""
    print "    echo \"  - HDF5 file is too large or on slow storage\""
    print "    echo \"  - Try running with '"'"'quick'"'"' mode or '"'"'n'"'"' to skip\""
    print "    exit 1"
    print "elif [ \\$MATLAB_EXIT -ne 0 ]; then"
    print "    echo \"\""
    print "    echo \"ERROR: MATLAB exited with code \\$MATLAB_EXIT\""
    print "    exit \\$MATLAB_EXIT"
    print "fi"
    print "WRAPPER_EOF"
    # Skip until we find WRAPPER_EOF
    while (getline && $0 !~ /^WRAPPER_EOF/) {}
    next
}
{ print }
' "$TARGET_FILE" > "${TARGET_FILE}.new"

# Check if AWK processing was successful
if [[ $? -eq 0 ]]; then
    log_verbose "AWK processing completed successfully"
    log_verbose "Moving processed file to replace original"
    mv "${TARGET_FILE}.new" "$TARGET_FILE"
    
    if [[ $? -eq 0 ]]; then
        log_verbose "File replacement completed successfully"
    else
        log_verbose "ERROR: Failed to replace original file"
        echo "ERROR: Failed to replace original file" >&2
        exit 1
    fi
else
    log_verbose "ERROR: AWK processing failed"
    echo "ERROR: AWK processing failed" >&2
    # Clean up temporary file if it exists
    [[ -f "${TARGET_FILE}.new" ]] && rm -f "${TARGET_FILE}.new"
    exit 1
fi

# Set executable permissions
log_verbose "Setting executable permissions on $TARGET_FILE"
chmod +x "$TARGET_FILE"

if [[ $? -eq 0 ]]; then
    log_verbose "Executable permissions set successfully"
else
    log_verbose "WARNING: Failed to set executable permissions"
    echo "WARNING: Failed to set executable permissions on $TARGET_FILE" >&2
fi

# Final success message
log_verbose "Wrapper creation process completed successfully"
echo "Wrapper creation fixed!"

# Log to file if verbose mode is enabled
if [[ $VERBOSE -eq 1 ]]; then
    log_file="logs/create_new_wrapper_$(date +%Y%m%d_%H%M%S).log"
    echo "[$(date)] create_new_wrapper.sh: Successfully updated wrapper in $TARGET_FILE" >> "$log_file"
    echo "[$(date)] create_new_wrapper.sh: Log file written to $log_file"
fi

log_verbose "Script execution completed"