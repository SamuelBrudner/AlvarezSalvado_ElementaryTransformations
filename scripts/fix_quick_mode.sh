#!/bin/bash

# Fix Quick Mode Logic Script
# Fixes quick mode detection logic in setup_smoke_plume_config.sh by correcting variable references
# Relocated from repository root to scripts/ directory with enhanced verbose logging support

# Initialize verbose mode (default: disabled)
VERBOSE=0

# Function to display usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Fix quick mode execution issues in the simulation pipeline by correcting variable references
in setup_smoke_plume_config.sh.

OPTIONS:
    -v, --verbose    Enable verbose logging with detailed trace output
    -h, --help       Display this help message and exit

DESCRIPTION:
    This script fixes a logic issue where QUICK_ANALYSIS is set but RUN_ANALYSIS is being
    checked. It updates the conditional statement in setup_smoke_plume_config.sh to use
    the correct variable name.

EXAMPLES:
    $0                    # Run with standard output
    $0 --verbose          # Run with detailed logging
    $0 -v > logs/fix_quick_mode_\$(date +%Y%m%d_%H%M%S).log 2>&1  # Log to file

EOF
}

# Function for verbose logging
log_verbose() {
    [[ $VERBOSE -eq 1 ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] fix_quick_mode.sh: $1"
}

# Function for error logging
log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR fix_quick_mode.sh: $1" >&2
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=1
            log_verbose "Verbose mode enabled"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main execution starts here
log_verbose "Starting quick mode fix script"
log_verbose "Current working directory: $(pwd)"

# Check if target file exists
TARGET_FILE="setup_smoke_plume_config.sh"
if [[ ! -f "$TARGET_FILE" ]]; then
    log_error "Target file '$TARGET_FILE' not found in current directory"
    log_verbose "Available files in current directory:"
    [[ $VERBOSE -eq 1 ]] && ls -la
    exit 1
fi

log_verbose "Target file '$TARGET_FILE' found"
log_verbose "File size: $(stat -c%s "$TARGET_FILE" 2>/dev/null || stat -f%z "$TARGET_FILE" 2>/dev/null || echo "unknown") bytes"

# Create backup of original file if verbose mode is enabled
if [[ $VERBOSE -eq 1 ]]; then
    BACKUP_FILE="${TARGET_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
    cp "$TARGET_FILE" "$BACKUP_FILE"
    log_verbose "Created backup: $BACKUP_FILE"
fi

# Check if perl is available
if ! command -v perl &> /dev/null; then
    log_error "perl command not found. Please install perl to run this script."
    exit 1
fi

log_verbose "perl command found: $(which perl)"

# Show what we're looking for before making changes
if [[ $VERBOSE -eq 1 ]]; then
    log_verbose "Searching for current pattern in $TARGET_FILE:"
    if grep -n 'if \[\[ "\$RUN_ANALYSIS" == "quick" \]\]' "$TARGET_FILE" 2>/dev/null; then
        log_verbose "Found pattern to be replaced"
    else
        log_verbose "Pattern not found - checking if already fixed"
        if grep -n 'if \[\[ "\$QUICK_ANALYSIS" == "1" \]\]' "$TARGET_FILE" 2>/dev/null; then
            log_verbose "Pattern already appears to be fixed"
        else
            log_verbose "Neither old nor new pattern found - this may be expected"
        fi
    fi
fi

log_verbose "Applying perl substitution to fix quick mode detection logic"

# The main fix: update the analysis mode detection in Step 4
perl -i -pe '
# Fix the analysis mode detection in Step 4
if (/if \[\[ "\$RUN_ANALYSIS" == "quick" \]\]; then/) {
    $_ = "    if [[ \"\$QUICK_ANALYSIS\" == \"1\" ]]; then\n";
}
' "$TARGET_FILE"

# Check if the perl command executed successfully  
if [[ $? -eq 0 ]]; then
    log_verbose "Perl substitution completed successfully"
else
    log_error "Perl substitution failed with exit code $?"
    exit 1
fi

# Verify the fix was applied if verbose mode is enabled
if [[ $VERBOSE -eq 1 ]]; then
    log_verbose "Verifying fix was applied:"
    if grep -n 'if \[\[ "\$QUICK_ANALYSIS" == "1" \]\]' "$TARGET_FILE" 2>/dev/null; then
        log_verbose "Fix verified: QUICK_ANALYSIS condition found"
    else
        log_verbose "Warning: Expected pattern not found after fix"
    fi
    
    # Show lines that were changed
    log_verbose "Lines containing 'QUICK_ANALYSIS' after fix:"
    grep -n "QUICK_ANALYSIS" "$TARGET_FILE" 2>/dev/null || log_verbose "No QUICK_ANALYSIS lines found"
fi

# Ensure logs directory exists for future use
if [[ ! -d "logs" ]]; then
    log_verbose "Creating logs directory for future logging"
    mkdir -p logs
fi

# Final success message
echo "Quick mode detection fixed!"
log_verbose "Script completed successfully"
log_verbose "End of execution at $(date '+%Y-%m-%d %H:%M:%S')"

exit 0