#!/bin/bash
# deploy_hpc_tools.sh - Deploy all HPC plume simulation tools
#
# This script ensures all HPC tools are properly set up and ready to use

echo "=== Deploying HPC Plume Simulation Tools ==="
echo ""

# Check if we're in the right directory
if [ ! -d "Code" ] || [ ! -d "configs" ]; then
    echo "Error: Must run from project root directory"
    echo "Expected: /vast/palmer/home.grace/snb6/Documents/AlvarezSalvado_ElementaryTransformations"
    exit 1
fi

# Create required directories
echo "Creating directories..."
mkdir -p results logs validation_sessions comparative_studies
echo "✓ Directories created"

# Make all scripts executable
echo ""
echo "Setting permissions..."
chmod +x *.sh 2>/dev/null
echo "✓ Scripts are executable"

# Check for required plume files
echo ""
echo "Checking plume data files..."
MISSING_FILES=0

# Check Crimaldi
CRIM_FILE="/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations/data/plumes/10302017_10cms_bounded.hdf5"
if [ -f "$CRIM_FILE" ]; then
    echo "✓ Crimaldi plume data found ($(ls -lh "$CRIM_FILE" | awk '{print $5}'))"
else
    echo "✗ Crimaldi plume data missing"
    MISSING_FILES=1
fi

# Check Smoke
SMOKE_FILE="/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations/data/plumes/smoke_1a_rotated_3d_fixed.h5"
if [ -f "$SMOKE_FILE" ]; then
    echo "✓ Smoke plume data found ($(ls -lh "$SMOKE_FILE" | awk '{print $5}'))"
else
    echo "✗ Smoke plume data missing"
    echo "  Run: matlab -batch 'fix_smoke_dimensions_batch'"
    MISSING_FILES=1
fi

# Check SLURM scripts
echo ""
echo "Checking SLURM job scripts..."
for script in nav_job_smoke.slurm nav_job_crimaldi.slurm nav_job_flexible.slurm; do
    if [ -f "$script" ]; then
        echo "✓ $script"
    else
        echo "✗ $script missing"
        MISSING_FILES=1
    fi
done

# Test MATLAB
echo ""
echo "Testing MATLAB availability..."
if command -v matlab >/dev/null 2>&1; then
    MATLAB_VERSION=$(matlab -batch "disp(version)" 2>&1 | grep -E '^[0-9]' | head -1)
    echo "✓ MATLAB found: $MATLAB_VERSION"
else
    echo "✗ MATLAB not found - load module with: module load MATLAB/2023b"
fi

# Quick functionality test
echo ""
echo "Running quick functionality test..."
matlab -batch "
    try
        cd('$(pwd)');
        addpath(genpath('Code'));
        [pf, pc] = get_plume_file();
        fprintf('✓ Current plume: %s\n', pc.plume_id);
        fprintf('✓ MATLAB path configuration working\n');
    catch ME
        fprintf('✗ Error: %s\n', ME.message);
    end
" 2>/dev/null | grep -E '^✓|^✗'

# Summary
echo ""
echo "=== Deployment Summary ==="
if [ $MISSING_FILES -eq 0 ]; then
    echo "✅ All files present and ready"
    echo ""
    echo "Quick start:"
    echo "  ./run_plume_sim.sh test     # Test both plumes"
    echo "  ./run_plume_sim.sh help     # Show all options"
    echo ""
    echo "Full guide: cat HPC_PLUME_GUIDE.md"
else
    echo "⚠️  Some files are missing - see above for details"
fi

# Create quick reference card
cat > QUICK_REFERENCE.txt << 'EOF'
HPC PLUME SIMULATION QUICK REFERENCE
====================================

QUICK COMMANDS:
  ./run_plume_sim.sh test              # Test (10 agents each)
  ./run_plume_sim.sh both 1000         # 1000 agents on BOTH plumes
  ./run_plume_sim.sh smoke 500         # 500 agents smoke only
  ./run_plume_sim.sh crimaldi 500      # 500 agents Crimaldi only

MONITOR:
  ./hpc_monitor_results.sh watch       # Live monitoring
  ./hpc_monitor_results.sh compare     # Compare results
  squeue -u $USER                      # Check jobs

MANAGE RESULTS:
  ./cleanup_results.sh summary         # Show summary
  ./cleanup_results.sh archive         # Archive results
  
FULL VALIDATION:
  ./validate_and_submit_plume.sh       # Interactive with figures

HELP:
  ./run_plume_sim.sh help
  cat HPC_PLUME_GUIDE.md
EOF

echo ""
echo "✓ Created QUICK_REFERENCE.txt"