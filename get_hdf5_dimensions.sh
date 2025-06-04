#!/bin/bash

HDF5_FILE="/vast/palmer/scratch/emonet/snb6/plume/smoke_1a_orig_backgroundsubtracted_rotated.h5"
MM_PER_PIXEL=0.15299877600979192

echo "Getting dimensions from HDF5 file..."

# Create minimal MATLAB script that ONLY gets dimensions
matlab -batch "
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
" 2>&1 | grep -v "Warning:"

