#!/bin/bash

echo "Testing basic MATLAB functionality..."

# Test 1: Can MATLAB start and exit?
echo -n "Test 1 - MATLAB startup: "
timeout 30s matlab -nodisplay -nosplash -r "disp('MATLAB OK'); exit(0)" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "PASSED"
else
    echo "FAILED - MATLAB cannot start properly"
    exit 1
fi

# Test 2: Can MATLAB access the HDF5 file?
echo -n "Test 2 - HDF5 file access: "
timeout 60s matlab -nodisplay -nosplash -r "
try
    fprintf('[%s] Testing HDF5 access\\n', datestr(now, 'HH:MM:SS'));
    tic;
    info = h5info('/vast/palmer/scratch/emonet/snb6/plume/smoke_1a_orig_backgroundsubtracted_rotated.h5');
    fprintf('[%s] Success! h5info took %.1f seconds\\n', datestr(now, 'HH:MM:SS'), toc);
    fprintf('File has %d datasets\\n', length(info.Datasets));
    exit(0);
catch ME
    fprintf('[%s] ERROR: %s\\n', datestr(now, 'HH:MM:SS'), ME.message);
    exit(1);
end
" 2>&1 | tee matlab_test.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "PASSED"
    echo "Check matlab_test.log for timing details"
else
    echo "FAILED - Cannot read HDF5 file"
    echo "Check matlab_test.log for error details"
fi

