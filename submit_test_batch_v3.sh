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

# Check for smoke video or HDF5
if [ -f "data/smoke_1a_orig_backgroundsubtracted.h5" ]; then
    echo "✓ Smoke HDF5 file found"
    SMOKE_TYPE="hdf5"
elif [ -f "data/smoke_1a_orig_backgroundsubtracted.avi" ]; then
    echo "✓ Smoke AVI file found"
    SMOKE_TYPE="avi"
else
    echo "ERROR: Neither smoke HDF5 nor AVI file found"
    echo "Expected: data/smoke_1a_orig_backgroundsubtracted.h5 or .avi"
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
mkdir -p logs configs data/raw/test_batch_v3 data/processed/test_batch_v3

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

# Create test_batch_v3.sh if it doesn't exist
if [ ! -f "test_batch_v3.sh" ]; then
    echo -e "\ntest_batch_v3.sh not found. Creating it now..."
    # Copy the content from the artifact here
    # (In practice, you'd have saved the artifact content to this file)
    echo "ERROR: Please save the test_batch_v3.sh script first"
    exit 1
fi

# Make script executable
chmod +x test_batch_v3.sh

# Show what will be submitted
echo -e "\n=== Job parameters ==="
echo "Job name: test_batch_v3"
echo "Array size: 0-9 (10 jobs total)"
echo "Plumes: crimaldi, smoke_hdf5"
echo "Agents per condition: 10"
echo "Agents per job: 2"
echo "Time limit: 30 minutes"
echo "Memory: 80GB per job"

# Ask for confirmation
echo -e "\nReady to submit test_batch_v3?"
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
    echo "  tail -f logs/test_batch_v3_${JOB_ID}_*.out"
    echo ""
    echo "Results will be in:"
    echo "  data/raw/test_batch_v3/"
    echo "  data/processed/test_batch_v3/"
else
    echo "ERROR: Job submission failed"
    exit 1
fi