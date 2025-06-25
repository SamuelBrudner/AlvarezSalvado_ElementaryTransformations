#!/bin/bash

# debug_and_fix.sh - Debugging and troubleshooting script for simulation environment
# Relocated to scripts/ directory with enhanced verbose logging support
#
# This script identifies and fixes common issues in the simulation environment,
# specifically addressing command substitution problems in wrapper script creation.
#
# Usage: debug_and_fix.sh [-v|--verbose]
#   -v, --verbose    Enable detailed trace output for debugging

# Initialize verbose mode flag
VERBOSE=0

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [-v|--verbose]"
            echo ""
            echo "Debugging and troubleshooting script for identifying and fixing common issues"
            echo "in the simulation environment."
            echo ""
            echo "Options:"
            echo "  -v, --verbose    Enable detailed trace output for debugging"
            echo "  -h, --help       Show this help message"
            exit 0
            ;;
        *)
            echo "Error: Unknown option '$1'"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Verbose logging function
log_verbose() {
    if [[ $VERBOSE -eq 1 ]]; then
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] DEBUG: $*"
    fi
}

# Ensure logs directory exists
if [[ ! -d "logs" ]]; then
    mkdir -p logs
    log_verbose "Created logs directory"
fi

# Main execution starts here
log_verbose "Starting debug_and_fix.sh execution"
log_verbose "Verbose mode enabled, will output detailed trace information"

echo "Debug and Fix Script - Addressing Wrapper Creation Issues"
echo "========================================================"

log_verbose "Preparing to create new wrapper script to fix command substitution issues"

# Create a completely new wrapper section that avoids the command substitution issue
log_verbose "Creating temporary wrapper script: create_new_wrapper.sh"

cat > create_new_wrapper.sh << 'NEWWRAPPER'
#!/bin/bash

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
' setup_smoke_plume_config.sh > setup_smoke_plume_config.new && mv setup_smoke_plume_config.new setup_smoke_plume_config.sh

chmod +x setup_smoke_plume_config.sh
echo "Wrapper creation fixed!"
NEWWRAPPER

log_verbose "Temporary wrapper script created successfully"

# Verify the target script exists before attempting to fix it
if [[ ! -f "setup_smoke_plume_config.sh" ]]; then
    echo "Warning: Target script 'setup_smoke_plume_config.sh' not found in current directory"
    log_verbose "Target script setup_smoke_plume_config.sh not found, checking for alternative locations"
    
    # Check if it exists in scripts directory (might be called from wrong directory)
    if [[ -f "scripts/setup_smoke_plume_config.sh" ]]; then
        echo "Found script in scripts/ directory, adjusting path reference"
        log_verbose "Found target script in scripts/ directory"
        # Update the awk command to target the correct file
        sed -i 's|setup_smoke_plume_config.sh|scripts/setup_smoke_plume_config.sh|g' create_new_wrapper.sh
    else
        echo "Error: Could not locate setup_smoke_plume_config.sh in expected locations"
        log_verbose "Failed to locate setup_smoke_plume_config.sh in current directory or scripts/ directory"
        echo "Please ensure you are running this script from the repository root or that the target script exists"
        log_verbose "Cleaning up temporary files before exit"
        rm -f create_new_wrapper.sh
        exit 1
    fi
fi

log_verbose "Making temporary wrapper script executable"
chmod +x create_new_wrapper.sh

log_verbose "Executing wrapper creation fix"
echo "Applying wrapper script fixes..."

# Execute the wrapper creation fix
bash create_new_wrapper.sh

# Check if the fix was successful
if [[ $? -eq 0 ]]; then
    echo "✓ Debug and fix operation completed successfully"
    log_verbose "Wrapper creation fix applied successfully"
else
    echo "✗ Error occurred during wrapper creation fix"
    log_verbose "Error occurred during wrapper creation fix execution"
    echo "Check the output above for specific error details"
    log_verbose "Cleaning up temporary files after error"
    rm -f create_new_wrapper.sh
    exit 1
fi

# Clean up temporary files
log_verbose "Cleaning up temporary wrapper script"
rm -f create_new_wrapper.sh

log_verbose "Debug and fix operation completed successfully"
echo ""
echo "Debug and fix process completed. The wrapper script creation issues"
echo "in setup_smoke_plume_config.sh should now be resolved."

if [[ $VERBOSE -eq 1 ]]; then
    echo ""
    echo "Verbose logging was enabled. Detailed trace information was displayed above."
    echo "For future runs, you can capture this output to a log file:"
    echo "  $0 --verbose > logs/debug_fix_\$(date +%Y%m%d_%H%M%S).log 2>&1"
fi

log_verbose "debug_and_fix.sh execution completed successfully"