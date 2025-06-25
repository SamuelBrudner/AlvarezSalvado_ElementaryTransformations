#!/bin/bash
# deploy_hpc_tools.sh - Deploy all HPC plume simulation tools
#
# This script ensures all HPC tools are properly set up and ready to use
#
# Usage: deploy_hpc_tools.sh [-v|--verbose]
#   -v, --verbose    Enable verbose logging with detailed trace output

# Initialize verbose mode
VERBOSE=0
SCRIPT_NAME="deploy_hpc_tools.sh"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [-v|--verbose]"
            echo "Deploy all HPC plume simulation tools"
            echo ""
            echo "Options:"
            echo "  -v, --verbose    Enable verbose logging with detailed trace output"
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

# Setup logging
LOG_DIR="../logs"
mkdir -p "$LOG_DIR" 2>/dev/null
LOG_FILE="$LOG_DIR/deploy_hpc_tools_$(date +%Y%m%d_%H%M%S).log"

# Verbose logging function
log_verbose() {
    local message="$1"
    local timestamp="[$(date '+%Y-%m-%d %H:%M:%S')]"
    if [[ $VERBOSE -eq 1 ]]; then
        echo "$timestamp $SCRIPT_NAME: $message" | tee -a "$LOG_FILE"
    fi
}

# Regular logging function (always output)
log_info() {
    local message="$1"
    echo "$message"
    if [[ $VERBOSE -eq 1 ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $SCRIPT_NAME: $message" >> "$LOG_FILE"
    fi
}

# Start deployment
log_verbose "Starting HPC tools deployment with verbose logging enabled"
log_verbose "Log file: $LOG_FILE"

echo "=== Deploying HPC Plume Simulation Tools ==="
echo ""

# Check if we're in the right directory
log_verbose "Checking current directory structure"
if [ ! -d "Code" ] || [ ! -d "configs" ]; then
    log_info "Error: Must run from project root directory"
    log_info "Expected: /vast/palmer/home.grace/snb6/Documents/AlvarezSalvado_ElementaryTransformations"
    log_verbose "Current directory: $(pwd)"
    log_verbose "Directory contents: $(ls -la)"
    exit 1
fi
log_verbose "✓ Confirmed running from project root directory: $(pwd)"

# Create required directories
log_info "Creating directories..."
log_verbose "Creating required directories: results, logs, validation_sessions, comparative_studies"
mkdir -p results logs validation_sessions comparative_studies
if [[ $? -eq 0 ]]; then
    log_info "✓ Directories created"
    log_verbose "✓ Successfully created all required directories"
else
    log_verbose "✗ Failed to create some directories"
fi

# Make all scripts executable
echo ""
log_info "Setting permissions..."
log_verbose "Setting executable permissions for scripts in scripts/ directory"

# Update path reference - scripts are now in scripts/ directory
if [ -d "scripts" ]; then
    chmod +x scripts/*.sh 2>/dev/null
    chmod +x scripts/*.py 2>/dev/null
    log_verbose "✓ Set permissions for scripts in scripts/ directory"
else
    log_verbose "⚠️ scripts/ directory not found, trying root directory"
    chmod +x *.sh 2>/dev/null
fi

log_info "✓ Scripts are executable"

# Check for required plume files
echo ""
log_info "Checking plume data files..."
log_verbose "Verifying existence and size of required plume data files"
MISSING_FILES=0

# Check Crimaldi
CRIM_FILE="/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations/data/plumes/10302017_10cms_bounded.hdf5"
log_verbose "Checking Crimaldi plume file: $CRIM_FILE"
if [ -f "$CRIM_FILE" ]; then
    FILE_SIZE=$(ls -lh "$CRIM_FILE" | awk '{print $5}')
    log_info "✓ Crimaldi plume data found ($FILE_SIZE)"
    log_verbose "✓ Crimaldi plume file verified: size $FILE_SIZE"
else
    log_info "✗ Crimaldi plume data missing"
    log_verbose "✗ Crimaldi plume file not found at: $CRIM_FILE"
    MISSING_FILES=1
fi

# Check Smoke
SMOKE_FILE="/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations/data/plumes/smoke_1a_rotated_3d_fixed.h5"
log_verbose "Checking Smoke plume file: $SMOKE_FILE"
if [ -f "$SMOKE_FILE" ]; then
    FILE_SIZE=$(ls -lh "$SMOKE_FILE" | awk '{print $5}')
    log_info "✓ Smoke plume data found ($FILE_SIZE)"
    log_verbose "✓ Smoke plume file verified: size $FILE_SIZE"
else
    log_info "✗ Smoke plume data missing"
    log_info "  Run: matlab -batch 'fix_smoke_dimensions_batch'"
    log_verbose "✗ Smoke plume file not found at: $SMOKE_FILE"
    log_verbose "  Suggested fix: matlab -batch 'fix_smoke_dimensions_batch'"
    MISSING_FILES=1
fi

# Check SLURM scripts - now in slurm/ directory
echo ""
log_info "Checking SLURM job scripts..."
log_verbose "Verifying SLURM job templates in slurm/ directory"

# Look for SLURM scripts in both old and new locations for compatibility
SLURM_SCRIPTS=("nav_job_smoke.slurm" "nav_job_crimaldi.slurm" "nav_job_flexible.slurm")
for script in "${SLURM_SCRIPTS[@]}"; do
    log_verbose "Checking for SLURM script: $script"
    if [ -f "slurm/$script" ]; then
        log_info "✓ $script (in slurm/)"
        log_verbose "✓ Found SLURM script at: slurm/$script"
    elif [ -f "$script" ]; then
        log_info "✓ $script (in root - consider moving to slurm/)"
        log_verbose "✓ Found SLURM script at root: $script (should be moved to slurm/)"
    else
        log_info "✗ $script missing"
        log_verbose "✗ SLURM script not found: $script (checked both slurm/ and root)"
        MISSING_FILES=1
    fi
done

# Test MATLAB
echo ""
log_info "Testing MATLAB availability..."
log_verbose "Checking MATLAB installation and version"
if command -v matlab >/dev/null 2>&1; then
    log_verbose "MATLAB command found, getting version information"
    MATLAB_VERSION=$(matlab -batch "disp(version)" 2>&1 | grep -E '^[0-9]' | head -1)
    log_info "✓ MATLAB found: $MATLAB_VERSION"
    log_verbose "✓ MATLAB version verified: $MATLAB_VERSION"
else
    log_info "✗ MATLAB not found - load module with: module load MATLAB/2023b"
    log_verbose "✗ MATLAB command not available in PATH"
    log_verbose "  Environment PATH: $PATH"
fi

# Quick functionality test
echo ""
log_info "Running quick functionality test..."
log_verbose "Testing MATLAB functionality with plume configuration"

MATLAB_TEST_OUTPUT=$(matlab -batch "
    try
        cd('$(pwd)');
        addpath(genpath('Code'));
        [pf, pc] = get_plume_file();
        fprintf('✓ Current plume: %s\n', pc.plume_id);
        fprintf('✓ MATLAB path configuration working\n');
    catch ME
        fprintf('✗ Error: %s\n', ME.message);
    end
" 2>/dev/null | grep -E '^✓|^✗')

echo "$MATLAB_TEST_OUTPUT"
log_verbose "MATLAB test results: $MATLAB_TEST_OUTPUT"

# Summary
echo ""
echo "=== Deployment Summary ==="
log_verbose "Generating deployment summary"

if [ $MISSING_FILES -eq 0 ]; then
    log_info "✅ All files present and ready"
    echo ""
    echo "Quick start:"
    # Updated script paths to reflect new scripts/ directory structure
    echo "  ./scripts/run_plume_sim.sh test     # Test both plumes"
    echo "  ./scripts/run_plume_sim.sh help     # Show all options"
    echo ""
    echo "Full guide: cat HPC_PLUME_GUIDE.md"
    log_verbose "✅ All required files found and verified"
    log_verbose "Quick start commands updated to use scripts/ directory paths"
else
    log_info "⚠️  Some files are missing - see above for details"
    log_verbose "⚠️  Deployment incomplete due to missing files"
fi

# Create quick reference card with updated paths
log_verbose "Creating quick reference card with updated script paths"
cat > QUICK_REFERENCE.txt << 'EOF'
HPC PLUME SIMULATION QUICK REFERENCE
====================================

QUICK COMMANDS:
  ./scripts/run_plume_sim.sh test              # Test (10 agents each)
  ./scripts/run_plume_sim.sh both 1000         # 1000 agents on BOTH plumes
  ./scripts/run_plume_sim.sh smoke 500         # 500 agents smoke only
  ./scripts/run_plume_sim.sh crimaldi 500      # 500 agents Crimaldi only

MONITOR:
  ./scripts/hpc_monitor_results.sh watch       # Live monitoring
  ./scripts/hpc_monitor_results.sh compare     # Compare results
  squeue -u $USER                              # Check jobs

MANAGE RESULTS:
  ./scripts/cleanup_results.sh summary         # Show summary
  ./scripts/cleanup_results.sh archive         # Archive results
  
FULL VALIDATION:
  ./scripts/validate_and_submit_plume.sh       # Interactive with figures

HELP:
  ./scripts/run_plume_sim.sh help
  cat HPC_PLUME_GUIDE.md

VERBOSE MODE:
  All scripts now support -v or --verbose flag for detailed logging
  Example: ./scripts/deploy_hpc_tools.sh -v
EOF

log_info ""
log_info "✓ Created QUICK_REFERENCE.txt"
log_verbose "✓ Quick reference card created with updated script paths"

if [[ $VERBOSE -eq 1 ]]; then
    echo ""
    echo "Verbose logging was enabled. Log file saved to: $LOG_FILE"
    log_verbose "Deployment script completed successfully"
fi