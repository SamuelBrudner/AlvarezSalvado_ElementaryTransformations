#!/bin/bash
# Script to check prerequisites and submit test_batch_v3
echo "=== Pre-submission checks for test_batch_v3 ==="

# Check if we're on a login node (can submit jobs)
if ! command -v sbatch &> /dev/null; then
    echo "ERROR: sbatch command not found. Are you on a login node?"
    exit 1
fi

# Check for required files
echo "Checking required files..."

# Check for HDF5 file
if [ ! -f "data/10302017_10cms_bounded.hdf5" ]; then
    echo "ERROR: Crimaldi HDF5 file not found at data/10302017_10cms_bounded.hdf5"
    exit 1
fi
echo "✓ Crimaldi HDF5 file found"

# Check for smoke HDF5 (our new 3D file)
if [ -f "data/smoke_1a_rotated_3d.h5" ]; then
    echo "✓ Smoke 3D HDF5 file found"
    SMOKE_TYPE="hdf5_3d"
elif [ -f "data/smoke_1a_orig_backgroundsubtracted.avi" ]; then
    echo "✓ Smoke AVI file found"
    SMOKE_TYPE="avi"
else
    echo "ERROR: Neither smoke HDF5 nor AVI file found"
    echo "Expected: data/smoke_1a_rotated_3d.h5 or data/smoke_1a_orig_backgroundsubtracted.avi"
    exit 1
fi

# Check smoke metadata file
if [ -f "data/smoke_hdf5_meta.yaml" ]; then
    echo "✓ Smoke metadata file found"
else
    echo "ERROR: Smoke metadata file not found at data/smoke_hdf5_meta.yaml"
    exit 1
fi

# Check MATLAB module availability
echo "Checking MATLAB module..."
if module avail 2>&1 | grep -q "MATLAB/2023b"; then
    echo "✓ MATLAB/2023b module available"
else
    echo "WARNING: MATLAB/2023b module may not be available"
    echo "Available MATLAB modules:"
    module avail 2>&1 | grep -i matlab
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p logs configs data/raw/test_batch_v3 data/processed/test_batch_v3 slurm_out slurm_err

# Check and report config files
echo -e "\nConfiguration files:"
if [ -f "configs/batch_crimaldi.yaml" ]; then
    echo "✓ configs/batch_crimaldi.yaml exists"
else
    echo "- configs/batch_crimaldi.yaml will be created"
fi

if [ -f "configs/batch_smoke_hdf5.yaml" ]; then
    echo "✓ configs/batch_smoke_hdf5.yaml exists"
else
    echo "- configs/batch_smoke_hdf5.yaml will be created"
fi

# Check test_batch_v3.sh exists
if [ ! -f "test_batch_v3.sh" ]; then
    echo "ERROR: test_batch_v3.sh not found!"
    exit 1
fi

# Make script executable
chmod +x test_batch_v3.sh

# DIAGNOSTIC: Check plume data
echo -e "\n=== Plume Data Diagnostics ==="
echo "Creating diagnostic plots for both plumes..."

# Create a Python script to check the plumes
cat > check_plumes_diagnostic.py << 'PYEOF'
#!/usr/bin/env python3
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

def check_plume(filepath, dataset_name, plume_name):
    print(f"\nChecking {plume_name} Plume ({filepath})...")
    try:
        with h5py.File(filepath, 'r') as f:
            data = f[dataset_name]
            shape = data.shape
            print(f"  Dataset: {dataset_name}")
            print(f"  Shape: {shape}")
            
            # Get middle frame
            if len(shape) == 3:
                middle_idx = shape[2] // 2
                middle_frame = data[:, :, middle_idx]
                print(f"  Middle frame index: {middle_idx}")
                print(f"  Range: [{middle_frame.min():.6f}, {middle_frame.max():.6f}]")
                print(f"  Mean: {middle_frame.mean():.6f}")
                
                # Create diagnostic plot
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Raw data plot
                im1 = ax1.imshow(middle_frame, cmap='hot', aspect='auto')
                ax1.set_title(f'{plume_name} - Frame {middle_idx} (Raw)')
                ax1.set_xlabel(f'Width: {shape[1]}')
                ax1.set_ylabel(f'Height: {shape[0]}')
                plt.colorbar(im1, ax=ax1)
                
                # Log scale plot (helpful for seeing low values)
                # Add small epsilon to avoid log(0)
                log_frame = np.log10(middle_frame + 1e-10)
                im2 = ax2.imshow(log_frame, cmap='hot', aspect='auto')
                ax2.set_title(f'{plume_name} - Frame {middle_idx} (Log10 scale)')
                ax2.set_xlabel(f'Width: {shape[1]}')
                ax2.set_ylabel(f'Height: {shape[0]}')
                plt.colorbar(im2, ax=ax2)
                
                plt.suptitle(f'{plume_name} Plume Diagnostic - {datetime.now().strftime("%Y-%m-%d %H:%M")}')
                plt.tight_layout()
                
                # Save the plot
                output_file = f'plume_diagnostic_{plume_name.lower()}.png'
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Saved diagnostic plot to {output_file}")
                
                return True
            else:
                print(f"  ERROR: Expected 3D data, got shape {shape}")
                return False
                
    except Exception as e:
        print(f"  ERROR reading file: {e}")
        return False

# Check both plumes
success1 = check_plume("data/10302017_10cms_bounded.hdf5", "/dataset2", "Crimaldi")
success2 = check_plume("data/smoke_1a_rotated_3d.h5", "/dataset2", "Smoke")

if success1 and success2:
    print("\n✓ Plume diagnostics complete - check plume_diagnostic_*.png files")
    exit(0)
else:
    print("\n✗ Plume diagnostic failed")
    exit(1)
PYEOF

# Run the diagnostic
if [ -f "./dev_env/bin/python" ]; then
    if conda run --prefix ./dev_env python check_plumes_diagnostic.py; then
        echo "✓ Diagnostic plots created successfully"
        echo "  View: plume_diagnostic_crimaldi.png and plume_diagnostic_smoke.png"
    else
        echo "ERROR: Diagnostic script failed"
        rm -f check_plumes_diagnostic.py
        exit 1
    fi
else
    echo "WARNING: Could not run Python diagnostic (conda environment not found)"
fi

# Clean up diagnostic script
rm -f check_plumes_diagnostic.py

# Show what will be submitted
echo -e "\n=== Job parameters ==="
echo "Job name: test_batch_v3"
echo "Array size: 0-9 (10 jobs total)"
echo "Plumes: crimaldi, smoke_hdf5 (3D)"
echo "Agents per condition: 10"
echo "Agents per job: 2"
echo "Time limit: 30 minutes"
echo "Memory: 80GB per job"
echo "Smoke data: $(readlink -f data/smoke_1a_rotated_3d.h5)"

# Ask for confirmation
echo -e "\nReady to submit test_batch_v3?"
echo "(Check the diagnostic plots first if needed)"
read -p "Continue? (y/n) " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Submission cancelled"
    exit 0
fi

# Submit the job
echo -e "\nSubmitting job..."
JOB_ID=$(sbatch test_batch_v3.sh | awk '{print $4}')

if [ -n "$JOB_ID" ]; then
    echo "✓ Job submitted successfully with ID: $JOB_ID"
    echo ""
    echo "Monitor with:"
    echo "  squeue -u $USER"
    echo "  tail -f slurm_out/test_batch_v3_*.out"
    echo ""
    echo "Results will be in:"
    echo "  data/raw/test_batch_v3/"
    echo "  data/processed/test_batch_v3/"
else
    echo "ERROR: Job submission failed"
    exit 1
fi
