#!/bin/bash

# fix_wrapper_syntax.sh - Fix syntax errors in wrapper scripts
# Relocated to scripts/ directory with enhanced verbose logging support
# Usage: ./scripts/fix_wrapper_syntax.sh [-v|--verbose] [target_file]

# Initialize variables
VERBOSE=0
TARGET_FILE="setup_smoke_plume_config.sh"
SCRIPT_NAME="$(basename "$0")"

# Function to display usage information
usage() {
    echo "Usage: $SCRIPT_NAME [-v|--verbose] [target_file]"
    echo ""
    echo "Fix syntax errors in wrapper scripts by replacing malformed wrapper sections"
    echo "with clean, properly formatted versions."
    echo ""
    echo "Options:"
    echo "  -v, --verbose     Enable verbose logging with detailed trace output"
    echo "  -h, --help        Display this help message"
    echo ""
    echo "Arguments:"
    echo "  target_file       Optional target file to fix (default: setup_smoke_plume_config.sh)"
    echo ""
    echo "Examples:"
    echo "  $SCRIPT_NAME"
    echo "  $SCRIPT_NAME --verbose"
    echo "  $SCRIPT_NAME -v setup_smoke_plume_config.sh"
    echo ""
    echo "Output:"
    echo "  When verbose mode is enabled, detailed execution steps are written to stdout"
    echo "  and can be redirected to logs/ directory for persistent debugging records."
}

# Function for verbose logging
log_verbose() {
    if [[ $VERBOSE -eq 1 ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $SCRIPT_NAME: $1"
    fi
}

# Function for error logging (always displayed)
log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR $SCRIPT_NAME: $1" >&2
}

# Function for info logging (always displayed)
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $SCRIPT_NAME: $1"
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
            usage
            exit 0
            ;;
        -*)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
        *)
            # Assume it's the target file
            TARGET_FILE="$1"
            log_verbose "Target file specified: $TARGET_FILE"
            shift
            ;;
    esac
done

# Validate target file exists
log_verbose "Checking if target file exists: $TARGET_FILE"
if [[ ! -f "$TARGET_FILE" ]]; then
    log_error "Target file not found: $TARGET_FILE"
    log_error "Please ensure the file exists in the current directory"
    exit 1
fi

log_verbose "Target file found: $TARGET_FILE"
log_verbose "Starting wrapper syntax fix process"

# Create backup of original file
BACKUP_FILE="${TARGET_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
log_verbose "Creating backup: $BACKUP_FILE"
if ! cp "$TARGET_FILE" "$BACKUP_FILE"; then
    log_error "Failed to create backup file: $BACKUP_FILE"
    exit 1
fi

log_verbose "Backup created successfully"

# Apply the awk script to fix wrapper syntax
log_verbose "Applying awk script to fix wrapper syntax"
log_verbose "Processing malformed wrapper sections in $TARGET_FILE"

# The awk script that fixes the wrapper syntax
AWK_SCRIPT='
/# Create a wrapper script to handle timeouts and errors/ {
    print "    # Create a wrapper script to handle timeouts and errors"
    print "    cat > \"$TEMP_DIR/run_analysis.sh\" << '\''WRAPPER_EOF'\''"
    print "#!/bin/bash"
    print "MATLAB_SCRIPT=\"$1\""
    print ""
    print "echo \"Running MATLAB analysis with ${TIMEOUT_SECONDS}s timeout...\""
    print "echo \"  Mode: ${N_SAMPLE_FRAMES} frames\""
    print ""
    print "# Run MATLAB with timeout"
    print "if command -v timeout &> /dev/null; then"
    print "    timeout ${TIMEOUT_SECONDS}s matlab -nodisplay -nosplash -r \"run('\''${MATLAB_SCRIPT}'\'')\" 2>&1"
    print "    MATLAB_EXIT=$?"
    print "else"
    print "    perl -e \"alarm ${TIMEOUT_SECONDS}; exec @ARGV\" matlab -nodisplay -nosplash -r \"run('\''${MATLAB_SCRIPT}'\'')\" 2>&1"
    print "    MATLAB_EXIT=$?"
    print "fi"
    print ""
    print "if [ $MATLAB_EXIT -eq 124 ] || [ $MATLAB_EXIT -eq 142 ]; then"
    print "    echo \"\""
    print "    echo \"ERROR: MATLAB analysis timed out after ${TIMEOUT_SECONDS} seconds!\""
    print "    echo \"Possible causes:\""
    print "    echo \"  - HDF5 file is too large or on slow storage\""
    print "    echo \"  - Try running with '\''n'\'' to skip analysis\""
    print "    exit 1"
    print "elif [ $MATLAB_EXIT -ne 0 ]; then"
    print "    echo \"\""
    print "    echo \"ERROR: MATLAB exited with code $MATLAB_EXIT\""
    print "    exit $MATLAB_EXIT"
    print "fi"
    print "WRAPPER_EOF"
    # Skip to the next chmod line
    while (getline && $0 !~ /chmod \+x/) {}
}
{ print }
'

# Create temporary file for processing
TEMP_FILE="${TARGET_FILE}.tmp"
log_verbose "Creating temporary file: $TEMP_FILE"

# Execute the awk script
log_verbose "Executing awk script to process wrapper syntax"
if ! awk "$AWK_SCRIPT" "$TARGET_FILE" > "$TEMP_FILE"; then
    log_error "Failed to execute awk script"
    log_verbose "Cleaning up temporary file: $TEMP_FILE"
    rm -f "$TEMP_FILE"
    exit 1
fi

log_verbose "Awk script executed successfully"

# Verify the temporary file was created and is not empty
if [[ ! -s "$TEMP_FILE" ]]; then
    log_error "Temporary file is empty or was not created properly"
    log_verbose "Cleaning up temporary file: $TEMP_FILE"
    rm -f "$TEMP_FILE"
    exit 1
fi

log_verbose "Temporary file created and verified"

# Replace the original file with the fixed version
log_verbose "Replacing original file with fixed version"
if ! mv "$TEMP_FILE" "$TARGET_FILE"; then
    log_error "Failed to replace original file with fixed version"
    log_verbose "Cleaning up temporary file: $TEMP_FILE"
    rm -f "$TEMP_FILE"
    exit 1
fi

log_verbose "File replacement successful"

# Make sure the target file is executable
log_verbose "Setting executable permissions on $TARGET_FILE"
if ! chmod +x "$TARGET_FILE"; then
    log_error "Failed to set executable permissions on $TARGET_FILE"
    exit 1
fi

log_verbose "Executable permissions set successfully"

# Verify the file was modified
log_verbose "Verifying file modification"
if [[ ! -f "$TARGET_FILE" ]]; then
    log_error "Target file no longer exists after processing"
    exit 1
fi

# Check if the backup is different from the current file
if cmp -s "$TARGET_FILE" "$BACKUP_FILE"; then
    log_verbose "No changes were made to the file (files are identical)"
    log_verbose "Removing unnecessary backup: $BACKUP_FILE"
    rm -f "$BACKUP_FILE"
else
    log_verbose "File was successfully modified"
    log_verbose "Backup preserved at: $BACKUP_FILE"
fi

# Final success message
log_info "Wrapper syntax fixed successfully!"
log_verbose "Process completed successfully"

# If verbose mode is enabled, provide additional information
if [[ $VERBOSE -eq 1 ]]; then
    echo ""
    echo "Summary of actions performed:"
    echo "  • Validated target file: $TARGET_FILE"
    echo "  • Created backup file: $BACKUP_FILE"
    echo "  • Applied awk script to fix wrapper syntax"
    echo "  • Replaced malformed wrapper sections with clean versions"
    echo "  • Set executable permissions"
    echo "  • Verified successful completion"
    echo ""
    echo "The fixed wrapper script now includes proper:"
    echo "  • Timeout handling with both 'timeout' and 'perl' fallback"
    echo "  • MATLAB exit code checking"
    echo "  • Error message formatting"
    echo "  • Proper heredoc syntax with quoted delimiter"
    echo ""
    echo "Log output can be redirected to logs/ directory for persistent records:"
    echo "  $SCRIPT_NAME --verbose > logs/fix_wrapper_syntax_\$(date +%Y%m%d_%H%M%S).log 2>&1"
fi

exit 0