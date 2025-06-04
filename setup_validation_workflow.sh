#!/bin/bash
# setup_validation_workflow.sh - Set up the complete validation workflow

echo "=== Setting Up Validation Workflow ==="
echo ""

# Make all scripts executable
echo "1. Making scripts executable..."
chmod +x validate_and_submit.sh quick_validate.sh manage_sessions.sh submit_range.sh 2>/dev/null

# Create validation sessions directory
echo "2. Creating validation_sessions directory..."
mkdir -p validation_sessions

# Test if we have validate_plume_setup_simple
echo "3. Checking validation function..."
./quick_validate.sh

echo ""
echo "=== Validation Workflow Ready ==="
echo ""
echo "Main workflows:"
echo ""
echo "1. Full validation and submission (recommended):"
echo "   ./validate_and_submit.sh [NUM_TASKS] [START_IDX]"
echo "   Examples:"
echo "     ./validate_and_submit.sh           # Submit all 100 tasks"
echo "     ./validate_and_submit.sh 50 0      # Submit tasks 0-49"
echo "     ./validate_and_submit.sh 50 50     # Submit tasks 50-99"
echo ""
echo "2. Quick submission for specific ranges:"
echo "   ./submit_range.sh                   # Interactive mode"
echo "   ./submit_range.sh 0 49              # Submit tasks 0-49"
echo "   ./submit_range.sh missing           # Find and submit missing tasks"
echo ""
echo "3. Session management:"
echo "   ./manage_sessions.sh list           # List all sessions"
echo "   ./manage_sessions.sh latest         # Show latest session"
echo "   ./manage_sessions.sh status         # Check job status"
echo "   ./manage_sessions.sh figures        # List validation figures"
echo ""
echo "=== Best Practices ==="
echo ""
echo "1. Always use validation for first submission:"
echo "   ./validate_and_submit.sh"
echo ""
echo "2. Check validation figure before approving"
echo ""
echo "3. Monitor your jobs:"
echo "   squeue -u $USER"
echo "   ./manage_sessions.sh status"
echo ""
echo "4. For reruns, use submit_range.sh:"
echo "   ./submit_range.sh missing"
echo ""

# Check current status
COMPLETED=$(ls -1 results/nav_results_*.mat 2>/dev/null | wc -l)
if [ $COMPLETED -gt 0 ]; then
    echo "=== Current Status ==="
    echo "You have $COMPLETED/100 results completed"
    if [ $COMPLETED -lt 100 ]; then
        echo ""
        echo "To submit remaining tasks:"
        echo "  ./submit_range.sh $COMPLETED 99"
    fi
fi