#!/bin/bash
# setup_complete_workflow.sh - Complete setup for the validation workflow

echo "=== Complete Validation Workflow Setup ==="
echo ""
echo "This script will set up the complete validation workflow for"
echo "navigation model simulations with visual confirmation."
echo ""

# Step 1: Create all necessary scripts
echo "Step 1: Creating workflow scripts..."

# Check if scripts already exist
SCRIPTS=(
    "validate_and_submit.sh"
    "quick_validate.sh" 
    "manage_sessions.sh"
    "submit_range.sh"
    "setup_validation_workflow.sh"
)

MISSING_SCRIPTS=()
for script in "${SCRIPTS[@]}"; do
    if [ ! -f "$script" ]; then
        MISSING_SCRIPTS+=("$script")
    fi
done

if [ ${#MISSING_SCRIPTS[@]} -gt 0 ]; then
    echo "  Missing scripts: ${MISSING_SCRIPTS[*]}"
    echo "  Please run the artifact creation commands first."
    exit 1
else
    echo "  ✓ All workflow scripts present"
fi

# Step 2: Create the simple validation function if needed
echo ""
echo "Step 2: Checking MATLAB validation function..."

if [ ! -f "Code/validate_plume_setup_simple.m" ]; then
    echo "  Creating validate_plume_setup_simple.m..."
    # Would create it here if not present
    echo "  Please ensure Code/validate_plume_setup_simple.m exists"
else
    echo "  ✓ Validation function exists"
fi

# Step 3: Make everything executable
echo ""
echo "Step 3: Setting permissions..."
chmod +x validate_and_submit.sh quick_validate.sh manage_sessions.sh submit_range.sh setup_validation_workflow.sh
echo "  ✓ Scripts are executable"

# Step 4: Create necessary directories
echo ""
echo "Step 4: Creating directories..."
mkdir -p validation_sessions results logs
echo "  ✓ Directories created"

# Step 5: Run the setup
echo ""
echo "Step 5: Running validation workflow setup..."
./setup_validation_workflow.sh

# Step 6: Show current status
echo ""
echo "=== Current Status ==="
COMPLETED=$(ls -1 results/nav_results_*.mat 2>/dev/null | wc -l)
echo "Completed simulations: $COMPLETED/100"

if [ $COMPLETED -eq 0 ]; then
    echo ""
    echo "=== Ready to Start ==="
    echo ""
    echo "To run your first simulation with validation:"
    echo "  ./validate_and_submit.sh"
    echo ""
    echo "This will:"
    echo "  1. Create a validation figure"
    echo "  2. Show simulation parameters"
    echo "  3. Ask for your approval"
    echo "  4. Submit the SLURM job if approved"
    
elif [ $COMPLETED -lt 100 ]; then
    echo ""
    echo "=== Continue Simulations ==="
    echo ""
    echo "You have partial results. Options:"
    echo ""
    echo "1. Submit remaining with validation:"
    echo "   ./validate_and_submit.sh $((100-COMPLETED)) $COMPLETED"
    echo ""
    echo "2. Quick submit remaining:"
    echo "   ./submit_range.sh $COMPLETED 99"
    echo ""
    echo "3. Check for missing tasks:"
    echo "   ./submit_range.sh missing"
    
else
    echo ""
    echo "✓ All simulations complete!"
    echo ""
    echo "To analyze results:"
    echo "  python view_results.py results/nav_results_0000.mat"
    echo "  ./create_results_report.sh"
fi

echo ""
echo "=== Documentation ==="
echo "See VALIDATION_WORKFLOW.md for detailed documentation"
echo ""
echo "Quick reference:"
echo "  Validate & submit:  ./validate_and_submit.sh [tasks] [start]"
echo "  Quick submit:       ./submit_range.sh [start] [end]"
echo "  Check status:       ./manage_sessions.sh status"
echo "  List sessions:      ./manage_sessions.sh list"