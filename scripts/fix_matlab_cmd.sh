#!/bin/bash

# Script for fixing MATLAB command line execution issues
# Relocated to scripts/ directory with enhanced verbose logging support for debugging MATLAB integration
# 
# Usage: ./scripts/fix_matlab_cmd.sh [-v|--verbose]
#   -v, --verbose    Enable detailed trace output for debugging

# Initialize variables
VERBOSE=0
SCRIPT_NAME="$(basename "$0")"
TIMESTAMP="$(date '+%Y-%m-%d %H:%M:%S')"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [-v|--verbose] [-h|--help]"
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
        echo "[$TIMESTAMP] $SCRIPT_NAME: $1"
    fi
}

# Create logs directory if it doesn't exist
if [[ ! -d "logs" ]]; then
    mkdir -p logs
    log_verbose "Created logs/ directory for centralized logging"
fi

log_verbose "Starting MATLAB command line execution fix"
log_verbose "Working directory: $(pwd)"

# Check if target file exists
TARGET_FILE="setup_smoke_plume_config.sh"
if [[ ! -f "$TARGET_FILE" ]]; then
    echo "Error: Target file '$TARGET_FILE' not found in current directory"
    log_verbose "Failed to locate target file: $TARGET_FILE"
    exit 1
fi

log_verbose "Target file found: $TARGET_FILE"

# Backup original file
BACKUP_FILE="${TARGET_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
cp "$TARGET_FILE" "$BACKUP_FILE"
log_verbose "Created backup: $BACKUP_FILE"

# Fix MATLAB_CMD usage with sed
log_verbose "Applying sed fix for MATLAB_CMD variable usage"
sed -i 's/\$MATLAB_CMD -nodisplay/matlab -nodisplay/g' "$TARGET_FILE"

if [[ $? -eq 0 ]]; then
    log_verbose "Successfully applied sed transformation for MATLAB_CMD"
else
    echo "Error: sed transformation failed"
    log_verbose "sed transformation failed, restoring from backup"
    cp "$BACKUP_FILE" "$TARGET_FILE"
    exit 1
fi

# Apply perl fix for proper MATLAB command setup
log_verbose "Applying perl fix for MATLAB command initialization"
perl -i -pe '
# Remove the previous stdbuf check if it exists
if (/# Check if we can unbuffer output/) {
    # Skip this section
    while (<>) {
        last if /^fi$/;
    }
    next;
}

# Add proper MATLAB command setup at the beginning of the wrapper
if (/^#!/ && !$done) {
    $_ .= "MATLAB_SCRIPT=\"\$1\"\n";
    $_ .= "MATLAB_CMD=\"matlab\"\n";
    $_ .= "\n";
    $done = 1;
    next;
}

# Remove the MATLAB_SCRIPT line if it appears later
next if /^MATLAB_SCRIPT="\$1"$/;
' "$TARGET_FILE"

if [[ $? -eq 0 ]]; then
    log_verbose "Successfully applied perl transformation for MATLAB initialization"
else
    echo "Error: perl transformation failed"
    log_verbose "perl transformation failed, restoring from backup"
    cp "$BACKUP_FILE" "$TARGET_FILE"
    exit 1
fi

# Verify the changes were applied
log_verbose "Verifying applied changes"
if grep -q "MATLAB_CMD=\"matlab\"" "$TARGET_FILE"; then
    log_verbose "Verification passed: MATLAB_CMD initialization found"
else
    echo "Warning: MATLAB_CMD initialization not found in modified file"
    log_verbose "Verification warning: MATLAB_CMD initialization not detected"
fi

# Check if sed replacement was effective
if grep -q "\$MATLAB_CMD -nodisplay" "$TARGET_FILE"; then
    echo "Warning: Some \$MATLAB_CMD references may still exist"
    log_verbose "Verification warning: Remaining \$MATLAB_CMD references detected"
else
    log_verbose "Verification passed: No remaining \$MATLAB_CMD references found"
fi

# Success message
echo "MATLAB_CMD issue fixed!"
log_verbose "MATLAB command line execution fix completed successfully"
log_verbose "Backup file preserved at: $BACKUP_FILE"

# Log completion to centralized logs if verbose mode is enabled
if [[ $VERBOSE -eq 1 ]]; then
    LOG_FILE="logs/fix_matlab_cmd_$(date +%Y%m%d_%H%M%S).log"
    {
        echo "[$TIMESTAMP] MATLAB_CMD Fix Execution Report"
        echo "[$TIMESTAMP] Script: $SCRIPT_NAME"
        echo "[$TIMESTAMP] Target File: $TARGET_FILE"
        echo "[$TIMESTAMP] Backup File: $BACKUP_FILE"
        echo "[$TIMESTAMP] Status: SUCCESS"
        echo "[$TIMESTAMP] Execution completed"
    } >> "$LOG_FILE"
    log_verbose "Detailed execution log written to: $LOG_FILE"
fi

exit 0