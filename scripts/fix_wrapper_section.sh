#!/bin/bash

# fix_wrapper_section.sh - Enhanced wrapper script section fixer with verbose logging
# 
# This script finds and replaces the wrapper script creation section in setup_smoke_plume_config.sh
# with an improved version that includes better error handling and timeout management.
#
# Usage: fix_wrapper_section.sh [OPTIONS]
# Options:
#   -v, --verbose    Enable verbose logging with detailed trace output
#   -h, --help       Show this help message
#
# The script uses AWK to locate and replace a specific section that creates wrapper scripts,
# improving the MATLAB execution timeout handling and error reporting.

# Initialize variables
VERBOSE=0
SCRIPT_NAME="$(basename "$0")"
TIMESTAMP="$(date '+%Y-%m-%d %H:%M:%S')"

# Function to display help
show_help() {
    cat << EOF
Usage: $SCRIPT_NAME [OPTIONS]

Enhanced wrapper script section fixer with verbose logging support.

This script finds and replaces the wrapper script creation section in 
setup_smoke_plume_config.sh with an improved version that includes better 
error handling and timeout management.

OPTIONS:
    -v, --verbose    Enable verbose logging with detailed trace output
    -h, --help       Show this help message

EXAMPLES:
    $SCRIPT_NAME                    # Run with default (quiet) mode
    $SCRIPT_NAME --verbose          # Run with verbose logging
    $SCRIPT_NAME -v > logs/fix_wrapper_\$(date +%Y%m%d_%H%M%S).log 2>&1

The script operates on setup_smoke_plume_config.sh and creates a backup before modification.
EOF
}

# Function for verbose logging
verbose_log() {
    if [[ $VERBOSE -eq 1 ]]; then
        echo "[$TIMESTAMP] $SCRIPT_NAME: $1" >&2
    fi
}

# Function for standard logging
log_info() {
    echo "[$TIMESTAMP] $SCRIPT_NAME: $1"
}

# Function for error logging
log_error() {
    echo "[$TIMESTAMP] $SCRIPT_NAME ERROR: $1" >&2
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=1
            verbose_log "Verbose logging enabled"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main script execution starts here
verbose_log "Starting wrapper section fix process"
verbose_log "Script location: $(pwd)/$SCRIPT_NAME"

# Check if target file exists
TARGET_FILE="setup_smoke_plume_config.sh"
if [[ ! -f "$TARGET_FILE" ]]; then
    log_error "Target file '$TARGET_FILE' not found in current directory"
    verbose_log "Current directory contents:"
    if [[ $VERBOSE -eq 1 ]]; then
        ls -la >&2
    fi
    exit 1
fi

verbose_log "Found target file: $TARGET_FILE"
verbose_log "File size: $(stat -c%s "$TARGET_FILE" 2>/dev/null || stat -f%z "$TARGET_FILE" 2>/dev/null || echo "unknown") bytes"

# Create backup of original file
BACKUP_FILE="${TARGET_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
verbose_log "Creating backup: $BACKUP_FILE"
cp "$TARGET_FILE" "$BACKUP_FILE"
if [[ $? -ne 0 ]]; then
    log_error "Failed to create backup file"
    exit 1
fi

verbose_log "Backup created successfully"

# Main AWK processing
verbose_log "Starting AWK processing to replace wrapper section"
verbose_log "Looking for pattern: '^    # Create a wrapper script to handle timeouts and errors$'"

awk '
BEGIN { 
    in_wrapper = 0;
    if (ENVIRON["VERBOSE"] == "1") {
        print "[" strftime("%Y-%m-%d %H:%M:%S") "] AWK: Starting wrapper section replacement" > "/dev/stderr"
    }
}
/^    # Create a wrapper script to handle timeouts and errors$/ {
    if (ENVIRON["VERBOSE"] == "1") {
        print "[" strftime("%Y-%m-%d %H:%M:%S") "] AWK: Found wrapper section start marker" > "/dev/stderr"
    }
    print
    print "    cat > \"$TEMP_DIR/run_analysis.sh\" << '\''WRAPPER_EOF'\''"
    print "#!/bin/bash"
    print "MATLAB_SCRIPT=\"$1\""
    print "TIMEOUT_SECONDS='$TIMEOUT_SECONDS'"
    print ""
    print "echo \"Running MATLAB analysis with ${TIMEOUT_SECONDS}s timeout...\""
    print "echo \"  Mode: '$( if [ \"$RUN_ANALYSIS\" == \"quick\" ]; then echo \"QUICK (10 frames)\"; else echo \"FULL (100 frames)\"; fi )'\""
    print ""
    print "# Run MATLAB with timeout"
    print "if command -v timeout &> /dev/null; then"
    print "    # GNU coreutils timeout available"
    print "    timeout ${TIMEOUT_SECONDS}s matlab -nodisplay -nosplash -r \"run('\''${MATLAB_SCRIPT}'\'')\" 2>&1"
    print "    MATLAB_EXIT=$?"
    print "else"
    print "    # Use perl for timeout if GNU timeout not available"
    print "    perl -e \"alarm ${TIMEOUT_SECONDS}; exec @ARGV\" matlab -nodisplay -nosplash -r \"run('\''${MATLAB_SCRIPT}'\'')\" 2>&1"
    print "    MATLAB_EXIT=$?"
    print "fi"
    print ""
    print "if [ $MATLAB_EXIT -eq 124 ] || [ $MATLAB_EXIT -eq 142 ]; then"
    print "    echo \"\""
    print "    echo \"ERROR: MATLAB analysis timed out after ${TIMEOUT_SECONDS} seconds!\""
    print "    echo \"Possible causes:\""
    print "    echo \"  - HDF5 file is too large or on slow storage\""
    print "    echo \"  - MATLAB is waiting for user input\""
    print "    echo \"  - Code directory not found\""
    print "    exit 1"
    print "elif [ $MATLAB_EXIT -ne 0 ]; then"
    print "    echo \"\""
    print "    echo \"ERROR: MATLAB exited with code $MATLAB_EXIT\""
    print "    exit $MATLAB_EXIT"
    print "fi"
    print "WRAPPER_EOF"
    in_wrapper = 1
    if (ENVIRON["VERBOSE"] == "1") {
        print "[" strftime("%Y-%m-%d %H:%M:%S") "] AWK: Inserted replacement wrapper section" > "/dev/stderr"
    }
    next
}
/^WRAPPER_EOF$/ && in_wrapper {
    if (ENVIRON["VERBOSE"] == "1") {
        print "[" strftime("%Y-%m-%d %H:%M:%S") "] AWK: Found wrapper section end marker, skipping original" > "/dev/stderr"
    }
    in_wrapper = 0
    next
}
!in_wrapper { print }
END {
    if (ENVIRON["VERBOSE"] == "1") {
        print "[" strftime("%Y-%m-%d %H:%M:%S") "] AWK: Processing complete" > "/dev/stderr"
    }
}
' ENVIRON=VERBOSE="$VERBOSE" "$TARGET_FILE" > "${TARGET_FILE}.tmp"

# Check if AWK processing was successful
if [[ $? -ne 0 ]]; then
    log_error "AWK processing failed"
    verbose_log "Restoring from backup: $BACKUP_FILE"
    cp "$BACKUP_FILE" "$TARGET_FILE"
    exit 1
fi

verbose_log "AWK processing completed successfully"

# Replace original file with processed version
verbose_log "Replacing original file with processed version"
mv "${TARGET_FILE}.tmp" "$TARGET_FILE"
if [[ $? -ne 0 ]]; then
    log_error "Failed to replace original file"
    verbose_log "Restoring from backup: $BACKUP_FILE"
    cp "$BACKUP_FILE" "$TARGET_FILE"
    exit 1
fi

# Make the target file executable
verbose_log "Setting executable permissions on $TARGET_FILE"
chmod +x "$TARGET_FILE"
if [[ $? -ne 0 ]]; then
    log_error "Failed to set executable permissions"
    exit 1
fi

# Verify the changes were applied
verbose_log "Verifying wrapper section replacement"
if grep -q "GNU coreutils timeout available" "$TARGET_FILE"; then
    verbose_log "Verification successful: Enhanced timeout handling found in file"
    log_info "Wrapper section replaced successfully!"
    verbose_log "Backup preserved as: $BACKUP_FILE"
else
    log_error "Verification failed: Enhanced timeout handling not found"
    verbose_log "Restoring from backup: $BACKUP_FILE"
    cp "$BACKUP_FILE" "$TARGET_FILE"
    exit 1
fi

verbose_log "Script execution completed successfully"