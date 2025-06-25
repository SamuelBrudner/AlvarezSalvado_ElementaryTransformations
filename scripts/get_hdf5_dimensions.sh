#!/bin/bash

# HDF5 Dimensions Utility Script
# Relocated from repository root to scripts/ directory per organizational restructure
# Enhanced with verbose logging support

# Default configuration
HDF5_FILE="/vast/palmer/scratch/emonet/snb6/plume/smoke_1a_orig_backgroundsubtracted_rotated.h5"
MM_PER_PIXEL=0.15299877600979192
VERBOSE=0

# Function to display usage information
usage() {
    echo "Usage: $0 [-v|--verbose] [-h|--help]"
    echo "Extract HDF5 file dimensions and metadata"
    echo ""
    echo "Options:"
    echo "  -v, --verbose    Enable verbose logging with detailed trace output"
    echo "  -h, --help       Display this help message"
    echo ""
    echo "Example:"
    echo "  $0                # Run with standard output"
    echo "  $0 -v            # Run with verbose logging"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option '$1'"
            usage
            exit 1
            ;;
    esac
done

# Verbose logging function
verbose_log() {
    [[ $VERBOSE -eq 1 ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] get_hdf5_dimensions.sh: $1"
}

# Create logs directory if verbose mode is enabled and directory doesn't exist
if [[ $VERBOSE -eq 1 ]]; then
    if [[ ! -d "logs" ]]; then
        mkdir -p logs
        verbose_log "Created logs directory"
    fi
fi

# Start main execution
verbose_log "Starting HDF5 dimensions extraction"
verbose_log "Target file: $HDF5_FILE"
verbose_log "MM per pixel conversion: $MM_PER_PIXEL"

echo "Getting dimensions from HDF5 file..."

# Verbose logging for file validation
if [[ $VERBOSE -eq 1 ]]; then
    if [[ -f "$HDF5_FILE" ]]; then
        verbose_log "HDF5 file exists and is accessible"
        file_size=$(stat -c%s "$HDF5_FILE" 2>/dev/null || echo "unknown")
        verbose_log "File size: $file_size bytes"
    else
        verbose_log "WARNING: HDF5 file not found at $HDF5_FILE"
    fi
fi

verbose_log "Preparing MATLAB script for dimension extraction"

# Create minimal MATLAB script that ONLY gets dimensions
verbose_log "Executing MATLAB batch command"
matlab_output=$(matlab -batch "
try
    fprintf('Reading file info...\n');
    tic;
    info = h5info('$HDF5_FILE');
    fprintf('Got info in %.1f seconds\n', toc);
    
    % Get first dataset
    if isempty(info.Datasets)
        error('No datasets found');
    end
    
    dims = info.Datasets(1).Dataspace.Size;
    width_px = dims(1);
    height_px = dims(2);
    n_frames = dims(3);
    
    % Calculate arena size
    width_cm = width_px * $MM_PER_PIXEL / 10;
    height_cm = height_px * $MM_PER_PIXEL / 10;
    
    fprintf('\nDimensions found:\n');
    fprintf('  Pixels: %d x %d\n', width_px, height_px);
    fprintf('  Arena: %.1f x %.1f cm\n', width_cm, height_cm);
    fprintf('  Frames: %d (%.1f minutes at 60 Hz)\n', n_frames, n_frames/60/60);
    
catch ME
    fprintf('ERROR: %s\n', ME.message);
    exit(1);
end
" 2>&1)

# Check MATLAB execution status
matlab_exit_code=$?
verbose_log "MATLAB execution completed with exit code: $matlab_exit_code"

if [[ $matlab_exit_code -ne 0 ]]; then
    verbose_log "ERROR: MATLAB execution failed"
    if [[ $VERBOSE -eq 1 ]]; then
        echo "Full MATLAB output:"
        echo "$matlab_output"
    fi
    exit 1
fi

# Filter and display output
filtered_output=$(echo "$matlab_output" | grep -v "Warning:")
echo "$filtered_output"

verbose_log "HDF5 dimensions extraction completed successfully"

# Log output to file if verbose mode is enabled  
if [[ $VERBOSE -eq 1 ]]; then
    log_file="logs/get_hdf5_dimensions_$(date '+%Y%m%d_%H%M%S').log"
    {
        echo "=== HDF5 Dimensions Extraction Log ==="
        echo "Timestamp: $(date)"
        echo "Script: get_hdf5_dimensions.sh"
        echo "Target file: $HDF5_FILE"
        echo "MM per pixel: $MM_PER_PIXEL"
        echo ""
        echo "=== MATLAB Output ==="
        echo "$filtered_output"
        echo ""
        echo "=== Execution Summary ==="
        echo "Exit code: $matlab_exit_code"
        echo "Completed at: $(date)"
    } > "$log_file"
    verbose_log "Detailed log saved to: $log_file"
fi

exit 0