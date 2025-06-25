#!/bin/bash
# setup_hpc_scripts.sh - Make all HPC scripts executable and verify setup
# Relocated to scripts/ directory with enhanced verbose logging support

# Initialize verbose flag
VERBOSE=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -v, --verbose    Enable verbose logging output"
            echo "  -h, --help       Show this help message"
            echo ""
            echo "Sets up HPC scripts by making them executable and verifying required files."
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
    [[ $VERBOSE -eq 1 ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] setup_hpc_scripts: $1"
}

# Create logs directory if it doesn't exist
mkdir -p ../logs

# Log to file if verbose mode is enabled
if [[ $VERBOSE -eq 1 ]]; then
    exec > >(tee -a "../logs/setup_hpc_scripts_$(date '+%Y%m%d_%H%M%S').log")
fi

log_verbose "Starting HPC scripts setup process"

echo "=== HPC Scripts Setup ==="
echo ""

log_verbose "Making HPC scripts executable"

# Make all scripts executable (updated paths to reflect scripts/ directory)
chmod +x validate_and_submit_plume.sh
log_verbose "Made validate_and_submit_plume.sh executable"

chmod +x hpc_batch_submit.sh
log_verbose "Made hpc_batch_submit.sh executable"

chmod +x hpc_comparative_study.sh
log_verbose "Made hpc_comparative_study.sh executable"

chmod +x hpc_monitor_results.sh
log_verbose "Made hpc_monitor_results.sh executable"

chmod +x run_plume_sim.sh
log_verbose "Made run_plume_sim.sh executable"

chmod +x setup_crimaldi_plume.sh
log_verbose "Made setup_crimaldi_plume.sh executable"

chmod +x setup_smoke_plume_config.sh
log_verbose "Made setup_smoke_plume_config.sh executable"

chmod +x check_plume_status.sh
log_verbose "Made check_plume_status.sh executable"

chmod +x test_both_plumes.sh
log_verbose "Made test_both_plumes.sh executable"

chmod +x quick_submit_plume.sh
log_verbose "Made quick_submit_plume.sh executable"

echo "✓ Made all scripts executable"
echo ""

# Check for required files
echo "Checking required files..."
log_verbose "Starting file verification process"

# Check SLURM scripts (updated paths to reflect slurm/ directory)
log_verbose "Checking SLURM template files"
for script in ../slurm/nav_job_crimaldi.slurm ../slurm/nav_job_smoke.slurm ../slurm/nav_job_flexible.slurm; do
    if [ -f "$script" ]; then
        echo "  ✓ $(basename $script)"
        log_verbose "Found SLURM template: $(basename $script)"
    else
        echo "  ✗ $(basename $script) (missing)"
        log_verbose "Missing SLURM template: $(basename $script)"
    fi
done

# Check plume configs
echo ""
echo "Checking plume configurations..."
log_verbose "Checking plume configuration files"
for config in ../configs/plumes/crimaldi_10cms_bounded.json ../configs/plumes/smoke_1a_backgroundsubtracted.json; do
    if [ -f "$config" ]; then
        echo "  ✓ $(basename $config)"
        log_verbose "Found plume config: $(basename $config)"
    else
        echo "  ✗ $(basename $config) (missing)"
        log_verbose "Missing plume config: $(basename $config)"
    fi
done

# Check HDF5 files
echo ""
echo "Checking plume data files..."
log_verbose "Checking HDF5 data files"
CRIM_HDF5="/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations/data/plumes/10302017_10cms_bounded.hdf5"
SMOKE_HDF5="/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations/data/plumes/smoke_1a_rotated_3d_fixed.h5"

for hdf5 in "$CRIM_HDF5" "$SMOKE_HDF5"; do
    if [ -f "$hdf5" ]; then
        SIZE=$(ls -lh "$hdf5" | awk '{print $5}')
        echo "  ✓ $(basename $hdf5) ($SIZE)"
        log_verbose "Found HDF5 file: $(basename $hdf5) with size $SIZE"
    else
        echo "  ✗ $(basename $hdf5) (missing)"
        log_verbose "Missing HDF5 file: $(basename $hdf5)"
    fi
done

# Create required directories
echo ""
echo "Creating required directories..."
log_verbose "Creating required directory structure"
mkdir -p ../results ../logs ../validation_sessions ../comparative_studies ../slurm_logs
echo "  ✓ results/"
log_verbose "Created results/ directory"
echo "  ✓ logs/"
log_verbose "Created logs/ directory"
echo "  ✓ validation_sessions/"
log_verbose "Created validation_sessions/ directory"
echo "  ✓ comparative_studies/"
log_verbose "Created comparative_studies/ directory"
echo "  ✓ slurm_logs/"
log_verbose "Created slurm_logs/ directory for SLURM job logs"

# Check current plume
echo ""
echo "Checking current active plume..."
log_verbose "Checking current active plume configuration"
./check_plume_status.sh 2>/dev/null | grep "ACTIVE PLUME:" || echo "  Could not determine active plume"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Quick start commands:"
echo "  ./scripts/run_plume_sim.sh test           # Test both plumes (10 agents each)"
echo "  ./scripts/run_plume_sim.sh both 1000      # Run 1000 agents on each plume"
echo "  ./scripts/hpc_monitor_results.sh watch    # Monitor progress"
echo ""
echo "For help:"
echo "  ./scripts/run_plume_sim.sh help"
echo "  cat HPC_PLUME_GUIDE.md"

log_verbose "HPC scripts setup process completed successfully"

# Exit with appropriate code
if [[ $VERBOSE -eq 1 ]]; then
    echo ""
    echo "Verbose logging was enabled. Check ../logs/setup_hpc_scripts_$(date '+%Y%m%d')*.log for detailed output."
fi