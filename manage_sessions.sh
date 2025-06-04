#!/bin/bash
# manage_sessions.sh - Manage validation sessions and track job progress

VALIDATION_DIR="validation_sessions"

case "$1" in
    list)
        echo "=== Validation Sessions ==="
        echo ""
        if [ -d "$VALIDATION_DIR" ] && [ "$(ls -A $VALIDATION_DIR 2>/dev/null)" ]; then
            # List sessions with details
            for session in $VALIDATION_DIR/session_*.txt; do
                if [ -f "$session" ]; then
                    echo "Session: $(basename $session)"
                    grep -E "Date:|Job ID:|Total agents:|Approved by:" "$session" | sed 's/^/  /'
                    echo ""
                fi
            done
        else
            echo "No validation sessions found."
        fi
        ;;
        
    latest)
        echo "=== Latest Session ==="
        echo ""
        LATEST=$(ls -t $VALIDATION_DIR/session_*.txt 2>/dev/null | head -1)
        if [ -f "$LATEST" ]; then
            cat "$LATEST"
            
            # Extract job ID and check status
            JOB_ID=$(grep "Job ID:" "$LATEST" | awk '{print $3}')
            if [ ! -z "$JOB_ID" ]; then
                echo ""
                echo "Current Status:"
                sacct -j "$JOB_ID" --format=JobID,State,ExitCode,Elapsed,NNodes --noheader | head -5
            fi
        else
            echo "No sessions found."
        fi
        ;;
        
    status)
        JOB_ID="$2"
        if [ -z "$JOB_ID" ]; then
            # Get latest job
            LATEST=$(ls -t $VALIDATION_DIR/session_*.txt 2>/dev/null | head -1)
            if [ -f "$LATEST" ]; then
                JOB_ID=$(grep "Job ID:" "$LATEST" | awk '{print $3}')
            fi
        fi
        
        if [ ! -z "$JOB_ID" ]; then
            echo "=== Job Status for $JOB_ID ==="
            echo ""
            
            # Show job status
            echo "SLURM Status:"
            sacct -j "$JOB_ID" --format=JobID,State,ExitCode,Elapsed --noheader | head -10
            
            # Count completed tasks
            echo ""
            echo "Completed Results:"
            COMPLETED=$(ls -1 results/nav_results_*.mat 2>/dev/null | wc -l)
            echo "  Files in results/: $COMPLETED"
            
            # Check for errors
            echo ""
            echo "Recent Errors:"
            grep -l "FAILED" logs/nav-${JOB_ID}_*.err 2>/dev/null | wc -l | xargs echo "  Failed tasks:"
            
        else
            echo "No job ID specified and no recent sessions found."
        fi
        ;;
        
    figures)
        echo "=== Validation Figures ==="
        echo ""
        ls -lht $VALIDATION_DIR/*.png 2>/dev/null | head -10
        ;;
        
    clean)
        # Clean old sessions (optional, with confirmation)
        echo "This will remove validation sessions older than 7 days."
        read -p "Continue? (yes/no): " CONFIRM
        if [ "$CONFIRM" = "yes" ]; then
            find $VALIDATION_DIR -name "*.txt" -mtime +7 -delete
            find $VALIDATION_DIR -name "*.png" -mtime +7 -delete
            find $VALIDATION_DIR -name "*.json" -mtime +7 -delete
            echo "✓ Cleaned old validation files"
        fi
        ;;
        
    *)
        echo "Usage: $0 {list|latest|status [JOB_ID]|figures|clean}"
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
        echo "  $0 status"
        echo "  $0 status 35525225"
        ;;
esac