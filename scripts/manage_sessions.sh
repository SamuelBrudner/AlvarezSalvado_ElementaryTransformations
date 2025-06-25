#!/bin/bash
# manage_sessions.sh - Manage validation sessions and track job progress
# Relocated to scripts/ directory with enhanced verbose logging support

# Initialize verbose logging
VERBOSE=0
SCRIPT_NAME="manage_sessions"
LOG_DIR="../logs"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR" 2>/dev/null

# Verbose logging function
log_verbose() {
    local message="$1"
    local timestamp="[$(date '+%Y-%m-%d %H:%M:%S')]"
    if [[ $VERBOSE -eq 1 ]]; then
        echo "${timestamp} ${SCRIPT_NAME}: ${message}"
        echo "${timestamp} ${SCRIPT_NAME}: ${message}" >> "${LOG_DIR}/manage_sessions.log" 2>/dev/null
    fi
}

# Parse verbose flags before processing main arguments
ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=1
            log_verbose "Verbose logging enabled"
            shift
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

# Restore positional parameters
set -- "${ARGS[@]}"

log_verbose "Starting session management with command: ${1:-help}"

VALIDATION_DIR="validation_sessions"
log_verbose "Using validation directory: $VALIDATION_DIR"

case "$1" in
    list)
        log_verbose "Executing list command"
        echo "=== Validation Sessions ==="
        echo ""
        if [ -d "$VALIDATION_DIR" ] && [ "$(ls -A $VALIDATION_DIR 2>/dev/null)" ]; then
            log_verbose "Found validation directory with contents"
            # List sessions with details
            session_count=0
            for session in $VALIDATION_DIR/session_*.txt; do
                if [ -f "$session" ]; then
                    ((session_count++))
                    log_verbose "Processing session file: $(basename $session)"
                    echo "Session: $(basename $session)"
                    grep -E "Date:|Job ID:|Total agents:|Approved by:" "$session" | sed 's/^/  /'
                    echo ""
                fi
            done
            log_verbose "Listed $session_count session files"
        else
            log_verbose "No validation sessions found in $VALIDATION_DIR"
            echo "No validation sessions found."
        fi
        ;;
        
    latest)
        log_verbose "Executing latest command"
        echo "=== Latest Session ==="
        echo ""
        LATEST=$(ls -t $VALIDATION_DIR/session_*.txt 2>/dev/null | head -1)
        log_verbose "Latest session file: ${LATEST:-none}"
        if [ -f "$LATEST" ]; then
            log_verbose "Displaying contents of latest session: $LATEST"
            cat "$LATEST"
            
            # Extract job ID and check status
            JOB_ID=$(grep "Job ID:" "$LATEST" | awk '{print $3}')
            log_verbose "Extracted job ID: ${JOB_ID:-none}"
            if [ ! -z "$JOB_ID" ]; then
                echo ""
                echo "Current Status:"
                log_verbose "Querying SLURM status for job $JOB_ID"
                sacct -j "$JOB_ID" --format=JobID,State,ExitCode,Elapsed,NNodes --noheader | head -5
                log_verbose "SLURM status query completed"
            fi
        else
            log_verbose "No session files found"
            echo "No sessions found."
        fi
        ;;
        
    status)
        log_verbose "Executing status command with job ID: ${2:-auto-detect}"
        JOB_ID="$2"
        if [ -z "$JOB_ID" ]; then
            log_verbose "No job ID provided, searching for latest session"
            # Get latest job
            LATEST=$(ls -t $VALIDATION_DIR/session_*.txt 2>/dev/null | head -1)
            if [ -f "$LATEST" ]; then
                JOB_ID=$(grep "Job ID:" "$LATEST" | awk '{print $3}')
                log_verbose "Found job ID from latest session: ${JOB_ID:-none}"
            fi
        fi
        
        if [ ! -z "$JOB_ID" ]; then
            log_verbose "Checking status for job ID: $JOB_ID"
            echo "=== Job Status for $JOB_ID ==="
            echo ""
            
            # Show job status
            echo "SLURM Status:"
            log_verbose "Querying SLURM account information for job $JOB_ID"
            sacct -j "$JOB_ID" --format=JobID,State,ExitCode,Elapsed --noheader | head -10
            
            # Count completed tasks
            echo ""
            echo "Completed Results:"
            log_verbose "Counting completed result files"
            COMPLETED=$(ls -1 results/nav_results_*.mat 2>/dev/null | wc -l)
            echo "  Files in results/: $COMPLETED"
            log_verbose "Found $COMPLETED completed result files"
            
            # Check for errors
            echo ""
            echo "Recent Errors:"
            log_verbose "Checking for error files in logs/nav-${JOB_ID}_*.err"
            FAILED_COUNT=$(grep -l "FAILED" logs/nav-${JOB_ID}_*.err 2>/dev/null | wc -l)
            echo "  Failed tasks: $FAILED_COUNT"
            log_verbose "Found $FAILED_COUNT failed tasks"
            
        else
            log_verbose "No job ID could be determined"
            echo "No job ID specified and no recent sessions found."
        fi
        ;;
        
    figures)
        log_verbose "Executing figures command"
        echo "=== Validation Figures ==="
        echo ""
        log_verbose "Listing PNG files in $VALIDATION_DIR (most recent 10)"
        ls -lht $VALIDATION_DIR/*.png 2>/dev/null | head -10
        figure_count=$(ls -1 $VALIDATION_DIR/*.png 2>/dev/null | wc -l)
        log_verbose "Found $figure_count figure files total"
        ;;
        
    clean)
        log_verbose "Executing clean command"
        # Clean old sessions (optional, with confirmation)
        echo "This will remove validation sessions older than 7 days."
        log_verbose "Prompting user for confirmation to clean old files"
        read -p "Continue? (yes/no): " CONFIRM
        log_verbose "User response: $CONFIRM"
        if [ "$CONFIRM" = "yes" ]; then
            log_verbose "Starting cleanup of files older than 7 days"
            
            # Count files before deletion for verbose logging
            txt_count=$(find $VALIDATION_DIR -name "*.txt" -mtime +7 2>/dev/null | wc -l)
            png_count=$(find $VALIDATION_DIR -name "*.png" -mtime +7 2>/dev/null | wc -l)
            json_count=$(find $VALIDATION_DIR -name "*.json" -mtime +7 2>/dev/null | wc -l)
            
            log_verbose "Found $txt_count .txt files, $png_count .png files, $json_count .json files to delete"
            
            find $VALIDATION_DIR -name "*.txt" -mtime +7 -delete
            find $VALIDATION_DIR -name "*.png" -mtime +7 -delete
            find $VALIDATION_DIR -name "*.json" -mtime +7 -delete
            
            echo "✓ Cleaned old validation files"
            log_verbose "Cleanup completed successfully"
        else
            log_verbose "Cleanup cancelled by user"
        fi
        ;;
        
    *)
        log_verbose "Invalid or no command provided, showing help"
        echo "Usage: $0 [OPTIONS] {list|latest|status [JOB_ID]|figures|clean}"
        echo ""
        echo "Options:"
        echo "  -v, --verbose   Enable verbose logging output"
        echo ""
        echo "Commands:"
        echo "  list    - List all validation sessions"
        echo "  latest  - Show latest session details and status"
        echo "  status  - Check job status (latest or specific JOB_ID)"
        echo "  figures - List validation figures"
        echo "  clean   - Remove old validation files (>7 days)"
        echo ""
        echo "Examples:"
        echo "  $0 list"
        echo "  $0 -v status"
        echo "  $0 --verbose status 35525225"
        ;;
esac

log_verbose "Session management script completed"