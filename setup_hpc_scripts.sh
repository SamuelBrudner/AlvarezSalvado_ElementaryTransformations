#!/bin/bash
# setup_hpc_scripts.sh - Make all HPC scripts executable and verify setup

echo "=== HPC Scripts Setup ==="
echo ""

# Make all scripts executable
chmod +x validate_and_submit_plume.sh
chmod +x hpc_batch_submit.sh
chmod +x hpc_comparative_study.sh
chmod +x hpc_monitor_results.sh
chmod +x run_plume_sim.sh
chmod +x setup_crimaldi_plume.sh
chmod +x setup_smoke_plume_config.sh
chmod +x check_plume_status.sh
chmod +x test_both_plumes.sh
chmod +x quick_submit_plume.sh

echo "✓ Made all scripts executable"
echo ""

# Check for required files
echo "Checking required files..."

# Check SLURM scripts
for script in nav_job_crimaldi.slurm nav_job_smoke.slurm nav_job_flexible.slurm; do
    if [ -f "$script" ]; then
        echo "  ✓ $script"
    else
        echo "  ✗ $script (missing)"
    fi
done

# Check plume configs
echo ""
echo "Checking plume configurations..."
for config in configs/plumes/crimaldi_10cms_bounded.json configs/plumes/smoke_1a_backgroundsubtracted.json; do
    if [ -f "$config" ]; then
        echo "  ✓ $config"
    else
        echo "  ✗ $config (missing)"
    fi
done

# Check HDF5 files
echo ""
echo "Checking plume data files..."
CRIM_HDF5="/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations/data/plumes/10302017_10cms_bounded.hdf5"
SMOKE_HDF5="/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations/data/plumes/smoke_1a_rotated_3d_fixed.h5"

for hdf5 in "$CRIM_HDF5" "$SMOKE_HDF5"; do
    if [ -f "$hdf5" ]; then
        SIZE=$(ls -lh "$hdf5" | awk '{print $5}')
        echo "  ✓ $(basename $hdf5) ($SIZE)"
    else
        echo "  ✗ $(basename $hdf5) (missing)"
    fi
done

# Create required directories
echo ""
echo "Creating required directories..."
mkdir -p results logs validation_sessions comparative_studies
echo "  ✓ results/"
echo "  ✓ logs/"
echo "  ✓ validation_sessions/"
echo "  ✓ comparative_studies/"

# Check current plume
echo ""
echo "Checking current active plume..."
./check_plume_status.sh 2>/dev/null | grep "ACTIVE PLUME:" || echo "  Could not determine active plume"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Quick start commands:"
echo "  ./run_plume_sim.sh test           # Test both plumes (10 agents each)"
echo "  ./run_plume_sim.sh both 1000      # Run 1000 agents on each plume"
echo "  ./hpc_monitor_results.sh watch    # Monitor progress"
echo ""
echo "For help:"
echo "  ./run_plume_sim.sh help"
echo "  cat HPC_PLUME_GUIDE.md"