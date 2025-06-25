#!/bin/bash

# test_wrapper_matlab.sh - MATLAB wrapper test script for validating wrapper functionality
# Relocated to scripts/ directory with enhanced verbose logging support
#
# Usage: ./scripts/test_wrapper_matlab.sh [OPTIONS]
# Options:
#   -v, --verbose    Enable verbose logging with detailed trace output
#   -h, --help       Show this help message
#
# The script creates a minimal MATLAB wrapper test to validate wrapper functionality.
# When verbose mode is enabled, detailed execution steps are logged to stdout and logs/ directory.

# Initialize verbose flag
VERBOSE=0

# Function to log verbose messages
log_verbose() {
    if [[ $VERBOSE -eq 1 ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] test_wrapper_matlab: $1" | tee -a logs/test_wrapper_matlab_$(date +%Y%m%d).log 2>/dev/null || echo "[$(date '+%Y-%m-%d %H:%M:%S')] test_wrapper_matlab: $1"
    fi
}

# Function to show help
show_help() {
    echo "MATLAB wrapper test script for validating wrapper functionality"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -v, --verbose    Enable verbose logging with detailed trace output"
    echo "  -h, --help       Show this help message"
    echo ""
    echo "This script creates a minimal MATLAB wrapper test to validate wrapper functionality."
    echo "When verbose mode is enabled, detailed execution steps are logged to stdout and logs/ directory."
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=1
            log_verbose "Verbose logging enabled"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information."
            exit 1
            ;;
    esac
done

# Ensure logs directory exists for verbose logging
if [[ $VERBOSE -eq 1 ]]; then
    mkdir -p logs
    log_verbose "Created logs directory for verbose output"
fi

log_verbose "Starting MATLAB wrapper test script"

echo "Creating minimal wrapper test..."
log_verbose "Initializing minimal wrapper test creation"

# Create temp dir
TEMP_DIR="./temp_test_$$"
log_verbose "Creating temporary directory: $TEMP_DIR"
mkdir -p "$TEMP_DIR"

if [[ $? -eq 0 ]]; then
    log_verbose "Successfully created temporary directory"
else
    echo "Error: Failed to create temporary directory"
    exit 1
fi

# Create simple MATLAB script
log_verbose "Creating MATLAB test script: $TEMP_DIR/test_startup.m"
cat > "$TEMP_DIR/test_startup.m" << 'MATLAB_EOF'
fprintf('\n=== MATLAB STARTED at %s ===\n', datestr(now));
fprintf('Working directory: %s\n', pwd);
fprintf('MATLAB version: %s\n', version);
fprintf('Pausing for 2 seconds to simulate work...\n');
pause(2);
fprintf('Done! Exiting cleanly.\n');
exit(0);
MATLAB_EOF

if [[ $? -eq 0 ]]; then
    log_verbose "Successfully created MATLAB test script"
else
    echo "Error: Failed to create MATLAB test script"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Create wrapper
log_verbose "Creating wrapper script: $TEMP_DIR/run_test.sh"
cat > "$TEMP_DIR/run_test.sh" << 'WRAPPER_EOF'
#!/bin/bash
echo "Wrapper: Starting MATLAB..."
matlab -nodisplay -nosplash -r "run('$1')" 2>&1
MATLAB_EXIT=$?
echo "Wrapper: MATLAB exited with code $MATLAB_EXIT"
exit $MATLAB_EXIT
WRAPPER_EOF

if [[ $? -eq 0 ]]; then
    log_verbose "Successfully created wrapper script"
else
    echo "Error: Failed to create wrapper script"
    rm -rf "$TEMP_DIR"
    exit 1
fi

log_verbose "Making wrapper script executable"
chmod +x "$TEMP_DIR/run_test.sh"

if [[ $? -eq 0 ]]; then
    log_verbose "Successfully made wrapper script executable"
else
    echo "Error: Failed to make wrapper script executable"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Run it
echo "Running wrapper test..."
log_verbose "Executing wrapper test with MATLAB script: $TEMP_DIR/test_startup.m"

"$TEMP_DIR/run_test.sh" "$TEMP_DIR/test_startup.m"
TEST_EXIT_CODE=$?

if [[ $TEST_EXIT_CODE -eq 0 ]]; then
    log_verbose "Wrapper test completed successfully (exit code: $TEST_EXIT_CODE)"
    echo "Wrapper test completed successfully!"
else
    log_verbose "Wrapper test failed with exit code: $TEST_EXIT_CODE"
    echo "Warning: Wrapper test failed with exit code: $TEST_EXIT_CODE"
fi

# Cleanup
log_verbose "Cleaning up temporary directory: $TEMP_DIR"
rm -rf "$TEMP_DIR"

if [[ $? -eq 0 ]]; then
    log_verbose "Successfully cleaned up temporary directory"
else
    echo "Warning: Failed to clean up temporary directory: $TEMP_DIR"
fi

log_verbose "MATLAB wrapper test script completed"

# Exit with the same code as the wrapper test
exit $TEST_EXIT_CODE