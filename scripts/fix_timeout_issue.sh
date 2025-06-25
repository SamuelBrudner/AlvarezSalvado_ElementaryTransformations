#!/bin/bash

# Script for diagnosing and fixing timeout-related issues in long-running simulations
# Relocated to scripts/ directory with enhanced verbose logging support

# Initialize verbose logging
VERBOSE=0
LOG_DIR="logs"

# Parse command line arguments for verbose flag
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [-v|--verbose] [-h|--help]"
            echo ""
            echo "Script for diagnosing and fixing timeout-related issues in long-running simulations."
            echo ""
            echo "Options:"
            echo "  -v, --verbose    Enable verbose logging output"
            echo "  -h, --help       Show this help message"
            echo ""
            echo "This script fixes timeout handling in setup_smoke_plume_config.sh by:"
            echo "  - Replacing wrapper script section with proper variable expansion"
            echo "  - Fixing n_samples line in MATLAB script"
            echo "  - Adding comprehensive timeout error handling"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Verbose logging function
log_verbose() {
    if [[ $VERBOSE -eq 1 ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] fix_timeout_issue: $1"
        # Also log to file if logs directory exists
        if [[ -d "$LOG_DIR" ]]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] fix_timeout_issue: $1" >> "$LOG_DIR/fix_timeout_issue.log"
        fi
    fi
}

# Standard logging function (always outputs)
log_info() {
    echo "$1"
    if [[ -d "$LOG_DIR" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] fix_timeout_issue: $1" >> "$LOG_DIR/fix_timeout_issue.log"
    fi
}

# Create logs directory if it doesn't exist and verbose mode is enabled
if [[ $VERBOSE -eq 1 ]] && [[ ! -d "$LOG_DIR" ]]; then
    mkdir -p "$LOG_DIR"
    log_verbose "Created logs directory: $LOG_DIR"
fi

log_verbose "Starting timeout issue fix process"
log_verbose "Verbose logging enabled"

# Check if target file exists
if [[ ! -f "setup_smoke_plume_config.sh" ]]; then
    log_info "ERROR: setup_smoke_plume_config.sh not found in current directory"
    log_verbose "Current directory: $(pwd)"
    log_verbose "Files in current directory:"
    if [[ $VERBOSE -eq 1 ]]; then
        ls -la
    fi
    exit 1
fi

log_verbose "Found target file: setup_smoke_plume_config.sh"

# Create backup of original file
BACKUP_FILE="setup_smoke_plume_config.sh.backup.$(date '+%Y%m%d_%H%M%S')"
log_verbose "Creating backup: $BACKUP_FILE"
cp setup_smoke_plume_config.sh "$BACKUP_FILE"

if [[ $? -eq 0 ]]; then
    log_verbose "Backup created successfully"
else
    log_info "ERROR: Failed to create backup file"
    exit 1
fi

log_verbose "Applying timeout fix using Perl substitution"

# Replace the entire wrapper script section with proper variable expansion
perl -i -0pe '
s/cat > "\$TEMP_DIR\/run_analysis.sh" << '\''WRAPPER_EOF'\''.*?WRAPPER_EOF/
cat > "\$TEMP_DIR\/run_analysis.sh" << WRAPPER_EOF
#!\/bin\/bash
MATLAB_SCRIPT="\\\$1"

echo "Running MATLAB analysis with ${TIMEOUT_SECONDS}s timeout..."
echo "  Mode: $(if [ "\$RUN_ANALYSIS" == "quick" ]; then echo "QUICK (10 frames)"; else echo "FULL (100 frames)"; fi)"

# Run MATLAB with timeout
if command -v timeout &> \/dev\/null; then
    # GNU coreutils timeout available
    timeout ${TIMEOUT_SECONDS}s matlab -nodisplay -nosplash -r "run('\''\\\${MATLAB_SCRIPT}'\'')" 2>&1
    MATLAB_EXIT=\\\$?
else
    # Use perl for timeout if GNU timeout not available
    perl -e "alarm ${TIMEOUT_SECONDS}; exec \@ARGV" matlab -nodisplay -nosplash -r "run('\''\\\${MATLAB_SCRIPT}'\'')" 2>&1
    MATLAB_EXIT=\\\$?
fi

if [ \\\$MATLAB_EXIT -eq 124 ] || [ \\\$MATLAB_EXIT -eq 142 ]; then
    echo ""
    echo "ERROR: MATLAB analysis timed out after ${TIMEOUT_SECONDS} seconds!"
    echo "Possible causes:"
    echo "  - HDF5 file is too large or on slow storage"
    echo "  - MATLAB is waiting for user input"
    echo "  - Code directory not found"
    exit 1
elif [ \\\$MATLAB_EXIT -ne 0 ]; then
    echo ""
    echo "ERROR: MATLAB exited with code \\\$MATLAB_EXIT"
    exit \\\$MATLAB_EXIT
fi
WRAPPER_EOF/gs' setup_smoke_plume_config.sh

# Check if Perl substitution was successful
if [[ $? -eq 0 ]]; then
    log_verbose "Perl substitution completed successfully"
else
    log_info "ERROR: Perl substitution failed"
    log_verbose "Restoring from backup: $BACKUP_FILE"
    cp "$BACKUP_FILE" setup_smoke_plume_config.sh
    exit 1
fi

log_verbose "Applying n_samples line fix using sed"

# Also fix the n_samples line in MATLAB script
sed -i 's/n_samples = min('\''$N_SAMPLE_FRAMES'\'', n_frames);/n_samples = min('"'"'${N_SAMPLE_FRAMES}'"'"', n_frames);/' setup_smoke_plume_config.sh

# Check if sed substitution was successful
if [[ $? -eq 0 ]]; then
    log_verbose "Sed substitution completed successfully"
else
    log_info "ERROR: Sed substitution failed"
    log_verbose "Restoring from backup: $BACKUP_FILE"
    cp "$BACKUP_FILE" setup_smoke_plume_config.sh
    exit 1
fi

# Verify the changes were applied correctly
log_verbose "Verifying applied changes"

# Check if the timeout wrapper was properly inserted
if grep -q "timeout ${TIMEOUT_SECONDS}s matlab" setup_smoke_plume_config.sh; then
    log_verbose "Timeout wrapper successfully applied"
else
    log_info "WARNING: Timeout wrapper may not have been applied correctly"
    log_verbose "Checking for alternative timeout patterns"
    if grep -q "perl -e \"alarm ${TIMEOUT_SECONDS}" setup_smoke_plume_config.sh; then
        log_verbose "Alternative Perl timeout pattern found"
    else
        log_info "ERROR: No timeout patterns found in modified file"
        exit 1
    fi
fi

# Check if n_samples fix was applied
if grep -q "n_samples = min('${N_SAMPLE_FRAMES}'" setup_smoke_plume_config.sh; then
    log_verbose "n_samples fix successfully applied"
else
    log_verbose "n_samples fix may not be applicable (variable not found)"
fi

log_info "Fix applied!"
log_verbose "Timeout issue fix process completed successfully"
log_verbose "Backup file saved as: $BACKUP_FILE"

if [[ $VERBOSE -eq 1 ]]; then
    log_verbose "Summary of changes:"
    log_verbose "  - Updated wrapper script with proper variable expansion"
    log_verbose "  - Added comprehensive timeout handling for both GNU timeout and Perl fallback"
    log_verbose "  - Enhanced error messages for timeout scenarios"
    log_verbose "  - Fixed n_samples variable quoting in MATLAB script"
fi

exit 0