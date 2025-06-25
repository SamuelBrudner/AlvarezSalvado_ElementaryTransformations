#!/bin/bash
# setup_validation_workflow.sh - Set up the complete validation workflow

# Verbose logging support
VERBOSE=0

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
            echo "Set up the complete validation workflow"
            echo ""
            echo "Options:"
            echo "  -v, --verbose    Enable verbose logging output"
            echo "  -h, --help       Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Initialize logging
[[ $VERBOSE -eq 1 ]] && echo "[$(date)] setup_validation_workflow.sh: Starting validation workflow setup"

echo "=== Setting Up Validation Workflow ==="
echo ""

# Make all scripts executable
echo "1. Making scripts executable..."
[[ $VERBOSE -eq 1 ]] && echo "[$(date)] setup_validation_workflow.sh: Making validation scripts executable"
chmod +x scripts/validate_and_submit.sh scripts/quick_validate.sh scripts/manage_sessions.sh scripts/submit_range.sh 2>/dev/null
[[ $VERBOSE -eq 1 ]] && echo "[$(date)] setup_validation_workflow.sh: Scripts made executable"

# Create validation sessions directory
echo "2. Creating validation_sessions directory..."
[[ $VERBOSE -eq 1 ]] && echo "[$(date)] setup_validation_workflow.sh: Creating validation_sessions directory"
mkdir -p validation_sessions
[[ $VERBOSE -eq 1 ]] && echo "[$(date)] setup_validation_workflow.sh: Directory created: validation_sessions"

# Test if we have validate_plume_setup_simple
echo "3. Checking validation function..."
[[ $VERBOSE -eq 1 ]] && echo "[$(date)] setup_validation_workflow.sh: Running quick validation check"
./scripts/quick_validate.sh
[[ $VERBOSE -eq 1 ]] && echo "[$(date)] setup_validation_workflow.sh: Validation check completed"

echo ""
echo "=== Validation Workflow Ready ==="
echo ""
echo "Main workflows:"
echo ""
echo "1. Full validation and submission (recommended):"
echo "   ./scripts/validate_and_submit.sh [NUM_TASKS] [START_IDX]"
echo "   Examples:"
echo "     ./scripts/validate_and_submit.sh           # Submit all 100 tasks"
echo "     ./scripts/validate_and_submit.sh 50 0      # Submit tasks 0-49"
echo "     ./scripts/validate_and_submit.sh 50 50     # Submit tasks 50-99"
echo ""
echo "2. Quick submission for specific ranges:"
echo "   ./scripts/submit_range.sh                   # Interactive mode"
echo "   ./scripts/submit_range.sh 0 49              # Submit tasks 0-49"
echo "   ./scripts/submit_range.sh missing           # Find and submit missing tasks"
echo ""
echo "3. Session management:"
echo "   ./scripts/manage_sessions.sh list           # List all sessions"
echo "   ./scripts/manage_sessions.sh latest         # Show latest session"
echo "   ./scripts/manage_sessions.sh status         # Check job status"
echo "   ./scripts/manage_sessions.sh figures        # List validation figures"
echo ""
echo "=== Best Practices ==="
echo ""
echo "1. Always use validation for first submission:"
echo "   ./scripts/validate_and_submit.sh"
echo ""
echo "2. Check validation figure before approving"
echo ""
echo "3. Monitor your jobs:"
echo "   squeue -u $USER"
echo "   ./scripts/manage_sessions.sh status"
echo ""
echo "4. For reruns, use submit_range.sh:"
echo "   ./scripts/submit_range.sh missing"
echo ""

# Check current status
[[ $VERBOSE -eq 1 ]] && echo "[$(date)] setup_validation_workflow.sh: Checking current completion status"
COMPLETED=$(ls -1 results/nav_results_*.mat 2>/dev/null | wc -l)
[[ $VERBOSE -eq 1 ]] && echo "[$(date)] setup_validation_workflow.sh: Found $COMPLETED completed results"

if [ $COMPLETED -gt 0 ]; then
    echo "=== Current Status ==="
    echo "You have $COMPLETED/100 results completed"
    if [ $COMPLETED -lt 100 ]; then
        echo ""
        echo "To submit remaining tasks:"
        echo "  ./scripts/submit_range.sh $COMPLETED 99"
        [[ $VERBOSE -eq 1 ]] && echo "[$(date)] setup_validation_workflow.sh: Suggested remaining task range: $COMPLETED to 99"
    fi
fi

[[ $VERBOSE -eq 1 ]] && echo "[$(date)] setup_validation_workflow.sh: Validation workflow setup completed successfully"