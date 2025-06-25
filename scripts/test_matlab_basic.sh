#!/bin/bash

# Basic MATLAB functionality test script
# Relocated to scripts/ directory with enhanced verbose logging support
# Maintains same CLI interface while adding optional -v/--verbose flag parsing

# Initialize verbose mode flag
VERBOSE=0

# Parse command line arguments for verbose flag
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        *)
            # Unknown option, ignore to maintain CLI compatibility
            shift
            ;;
    esac
done

# Verbose logging function with timestamp
log_verbose() {
    if [[ $VERBOSE -eq 1 ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    fi
}

# Create logs directory if it doesn't exist and verbose mode is enabled
if [[ $VERBOSE -eq 1 ]]; then
    if [[ ! -d "logs" ]]; then
        mkdir -p logs
        log_verbose "Created logs directory"
    fi
fi

log_verbose "Starting MATLAB basic functionality test"
log_verbose "Script location: scripts/test_matlab_basic.sh"
log_verbose "Verbose logging: ENABLED"

echo "Testing basic MATLAB functionality..."

# Test 1: Can MATLAB start and exit?
log_verbose "Preparing Test 1: MATLAB startup test"
echo -n "Test 1 - MATLAB startup: "

log_verbose "Executing MATLAB startup test with 30-second timeout"
timeout 30s matlab -nodisplay -nosplash -r "disp('MATLAB OK'); exit(0)" > /dev/null 2>&1
test1_result=$?

if [ $test1_result -eq 0 ]; then
    echo "PASSED"
    log_verbose "Test 1 PASSED: MATLAB started and exited successfully"
else
    echo "FAILED - MATLAB cannot start properly"
    log_verbose "Test 1 FAILED: MATLAB startup failed with exit code $test1_result"
    log_verbose "Script terminating due to critical failure"
    exit 1
fi

# Test 2: Can MATLAB access the HDF5 file?
log_verbose "Preparing Test 2: HDF5 file access test"
echo -n "Test 2 - HDF5 file access: "

log_verbose "Executing HDF5 access test with 60-second timeout"
log_verbose "Target HDF5 file: /vast/palmer/scratch/emonet/snb6/plume/smoke_1a_orig_backgroundsubtracted_rotated.h5"

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

test2_result=${PIPESTATUS[0]}

if [ $test2_result -eq 0 ]; then
    echo "PASSED"
    echo "Check matlab_test.log for timing details"
    log_verbose "Test 2 PASSED: HDF5 file access successful"
    log_verbose "MATLAB test log written to matlab_test.log"
else
    echo "FAILED - Cannot read HDF5 file"
    echo "Check matlab_test.log for error details"
    log_verbose "Test 2 FAILED: HDF5 file access failed with exit code $test2_result"
    log_verbose "Error details written to matlab_test.log"
fi

log_verbose "MATLAB basic functionality test completed"
log_verbose "Test summary: Test 1 (MATLAB startup): PASSED, Test 2 (HDF5 access): $([ $test2_result -eq 0 ] && echo 'PASSED' || echo 'FAILED')"

# Exit with the result of Test 2 (most critical test)
exit $test2_result