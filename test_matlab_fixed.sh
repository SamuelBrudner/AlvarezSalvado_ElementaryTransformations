#!/bin/bash

echo "Testing MATLAB functionality..."

# Test 1: Basic MATLAB
echo -n "Test 1 - MATLAB startup: "
matlab -batch "disp('MATLAB OK')" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "PASSED"
else
    echo "FAILED"
fi

# Test 2: HDF5 access (this might take longer)
echo "Test 2 - HDF5 file access (this may take a minute)..."
matlab -batch "
try
    fprintf('[%s] Testing HDF5 access\\n', datestr(now, 'HH:MM:SS'));
    tic;
    info = h5info('/vast/palmer/scratch/emonet/snb6/plume/smoke_1a_orig_backgroundsubtracted_rotated.h5');
    fprintf('[%s] Success! h5info took %.1f seconds\\n', datestr(now, 'HH:MM:SS'), toc);
    fprintf('File has %d datasets\\n', length(info.Datasets));
catch ME
    fprintf('[%s] ERROR: %s\\n', datestr(now, 'HH:MM:SS'), ME.message);
    rethrow(ME);
end
" 2>&1 | tee matlab_hdf5_test.log

echo "Check matlab_hdf5_test.log for details"
