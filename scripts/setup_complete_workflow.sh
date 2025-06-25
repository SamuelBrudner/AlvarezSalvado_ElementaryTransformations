#!/bin/bash
# setup_complete_workflow.sh - Complete setup for the validation workflow
# Relocated to scripts/ directory with enhanced verbose logging support

# Initialize verbose logging
VERBOSE=0
LOG_PREFIX="[$(date '+%Y-%m-%d %H:%M:%S')] setup_complete_workflow.sh:"

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
            echo "Complete setup for the validation workflow"
            echo ""
            echo "Options:"
            echo "  -v, --verbose    Enable verbose logging output"
            echo "  -h, --help       Show this help message"
            echo ""
            echo "This script will set up the complete validation workflow for"
            echo "navigation model simulations with visual confirmation."
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
    [[ $VERBOSE -eq 1 ]] && echo "$LOG_PREFIX $1"
}

# Ensure logs directory exists
mkdir -p logs
[[ $VERBOSE -eq 1 ]] && exec > >(tee -a "logs/setup_complete_workflow_$(date +%Y%m%d_%H%M%S).log") 2>&1

log_verbose "Starting complete validation workflow setup"
log_verbose "Verbose logging enabled, output will be captured in logs/ directory"

echo "=== Complete Validation Workflow Setup ==="
echo ""
echo "This script will set up the complete validation workflow for"
echo "navigation model simulations with visual confirmation."
echo ""

# Step 1: Create all necessary scripts
echo "Step 1: Creating workflow scripts..."
log_verbose "Checking for required workflow scripts in scripts/ directory"

# Check if scripts already exist (updated paths to scripts/ directory)
SCRIPTS=(
    "scripts/validate_and_submit.sh"
    "scripts/quick_validate.sh" 
    "scripts/manage_sessions.sh"
    "scripts/submit_range.sh"
    "scripts/setup_validation_workflow.sh"
)

log_verbose "Required scripts: ${SCRIPTS[*]}"

MISSING_SCRIPTS=()
for script in "${SCRIPTS[@]}"; do
    log_verbose "Checking for script: $script"
    if [ ! -f "$script" ]; then
        MISSING_SCRIPTS+=("$script")
        log_verbose "Missing script: $script"
    else
        log_verbose "Found script: $script"
    fi
done

if [ ${#MISSING_SCRIPTS[@]} -gt 0 ]; then
    echo "  Missing scripts: ${MISSING_SCRIPTS[*]}"
    echo "  Please run the artifact creation commands first."
    log_verbose "Exiting due to missing scripts: ${MISSING_SCRIPTS[*]}"
    exit 1
else
    echo "  ✓ All workflow scripts present"
    log_verbose "All required workflow scripts are present"
fi

# Step 2: Create the simple validation function if needed
echo ""
echo "Step 2: Checking MATLAB validation function..."
log_verbose "Checking for MATLAB validation function in Code/ directory"

if [ ! -f "Code/validate_plume_setup_simple.m" ]; then
    echo "  Creating validate_plume_setup_simple.m..."
    log_verbose "MATLAB validation function not found at Code/validate_plume_setup_simple.m"
    # Would create it here if not present
    echo "  Please ensure Code/validate_plume_setup_simple.m exists"
else
    echo "  ✓ Validation function exists"
    log_verbose "MATLAB validation function found at Code/validate_plume_setup_simple.m"
fi

# Step 3: Make everything executable
echo ""
echo "Step 3: Setting permissions..."
log_verbose "Setting executable permissions for workflow scripts"

# Update script paths to use scripts/ directory
chmod +x scripts/validate_and_submit.sh scripts/quick_validate.sh scripts/manage_sessions.sh scripts/submit_range.sh scripts/setup_validation_workflow.sh 2>/dev/null || {
    log_verbose "Some scripts may not exist yet for permission setting"
}
echo "  ✓ Scripts are executable"
log_verbose "Executable permissions set for all workflow scripts"

# Step 4: Create necessary directories
echo ""
echo "Step 4: Creating directories..."
log_verbose "Creating required directories for workflow execution"

mkdir -p validation_sessions results logs slurm_logs
log_verbose "Created directories: validation_sessions, results, logs, slurm_logs"
echo "  ✓ Directories created"

# Step 5: Run the setup
echo ""
echo "Step 5: Running validation workflow setup..."
log_verbose "Executing setup_validation_workflow.sh script"

# Use updated path for setup script
if [ -f "scripts/setup_validation_workflow.sh" ]; then
    if [[ $VERBOSE -eq 1 ]]; then
        ./scripts/setup_validation_workflow.sh --verbose
    else
        ./scripts/setup_validation_workflow.sh
    fi
    log_verbose "Setup validation workflow completed"
else
    echo "  Warning: scripts/setup_validation_workflow.sh not found"
    log_verbose "Warning: setup_validation_workflow.sh not found in scripts/ directory"
fi

# Step 6: Show current status
echo ""
echo "Step 6: Analyzing current status..."
log_verbose "Checking for existing simulation results"

echo ""
echo "=== Current Status ==="
COMPLETED=$(ls -1 results/nav_results_*.mat 2>/dev/null | wc -l)
echo "Completed simulations: $COMPLETED/100"
log_verbose "Found $COMPLETED completed simulations out of 100 total"

if [ $COMPLETED -eq 0 ]; then
    echo ""
    echo "=== Ready to Start ==="
    echo ""
    echo "To run your first simulation with validation:"
    echo "  ./scripts/validate_and_submit.sh"
    echo ""
    echo "This will:"
    echo "  1. Create a validation figure"
    echo "  2. Show simulation parameters"
    echo "  3. Ask for your approval"
    echo "  4. Submit the SLURM job if approved"
    log_verbose "No simulations completed yet - showing startup instructions"
    
elif [ $COMPLETED -lt 100 ]; then
    echo ""
    echo "=== Continue Simulations ==="
    echo ""
    echo "You have partial results. Options:"
    echo ""
    echo "1. Submit remaining with validation:"
    echo "   ./scripts/validate_and_submit.sh $((100-COMPLETED)) $COMPLETED"
    echo ""
    echo "2. Quick submit remaining:"
    echo "   ./scripts/submit_range.sh $COMPLETED 99"
    echo ""
    echo "3. Check for missing tasks:"
    echo "   ./scripts/submit_range.sh missing"
    log_verbose "Partial completion detected ($COMPLETED/100) - showing continuation options"
    
else
    echo ""
    echo "✓ All simulations complete!"
    echo ""
    echo "To analyze results:"
    echo "  python scripts/view_results.py results/nav_results_0000.mat"
    echo "  ./scripts/create_results_report.sh"
    log_verbose "All simulations completed - showing analysis options"
fi

echo ""
echo "=== Documentation ==="
echo "See VALIDATION_WORKFLOW.md for detailed documentation"
echo ""
echo "Quick reference:"
echo "  Validate & submit:  ./scripts/validate_and_submit.sh [tasks] [start]"
echo "  Quick submit:       ./scripts/submit_range.sh [start] [end]"
echo "  Check status:       ./scripts/manage_sessions.sh status"
echo "  List sessions:      ./scripts/manage_sessions.sh list"

log_verbose "Complete validation workflow setup finished successfully"

if [[ $VERBOSE -eq 1 ]]; then
    echo ""
    echo "=== Verbose Logging Summary ==="
    echo "Log file: logs/setup_complete_workflow_$(date +%Y%m%d_%H%M%S).log"
    echo "All verbose output has been captured for debugging purposes"
fi