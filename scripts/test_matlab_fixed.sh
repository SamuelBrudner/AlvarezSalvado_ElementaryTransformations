#!/bin/bash

# test_matlab_fixed.sh - Enhanced MATLAB testing script with verbose logging support
# Moved from repository root to scripts/ directory per organizational restructure
# Added verbose logging capabilities per Section 7.2 CLI requirements

# Initialize verbose logging flag
VERBOSE=0

# Function to output verbose messages with timestamp
verbose_log() {
    if [[ $VERBOSE -eq 1 ]]; then
        echo "[$(date '+%H:%M:%S')] test_matlab_fixed.sh: $1"
    fi
}

# Function to display usage information
show_usage() {
    echo "Usage: $0 [-v|--verbose] [-h|--help]"
    echo
    echo "Test MATLAB functionality including basic startup and HDF5 file access."
    echo
    echo "Options:"
    echo "  -v, --verbose    Enable detailed trace output for debugging"
    echo "  -h, --help       Show this help message"
    echo
    echo "Examples:"
    echo "  $0                    # Run tests with standard output"
    echo "  $0 --verbose          # Run tests with detailed trace output"
    echo "  $0 -v > logs/matlab_test_\$(date +%Y%m%d_%H%M%S).log 2>&1"
    echo "                        # Run with verbose logging to file"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=1
            verbose_log "Verbose logging enabled"
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option '$1'"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Ensure logs directory exists if verbose output is redirected
if [[ $VERBOSE -eq 1 ]]; then
    verbose_log "Initializing MATLAB testing workflow"
    verbose_log "Creating logs directory if it doesn't exist"
    mkdir -p logs
fi

echo "Testing MATLAB functionality..."
verbose_log "Starting MATLAB functionality validation"

# Test 1: Basic MATLAB startup
verbose_log "Initiating Test 1: Basic MATLAB startup validation"
echo -n "Test 1 - MATLAB startup: "

verbose_log "Executing MATLAB batch command for startup test"
matlab -batch "disp('MATLAB OK')" > /dev/null 2>&1
test1_result=$?

if [ $test1_result -eq 0 ]; then
    echo "PASSED"
    verbose_log "Test 1 PASSED: MATLAB startup successful"
else
    echo "FAILED"
    verbose_log "Test 1 FAILED: MATLAB startup failed with exit code $test1_result"
fi

# Test 2: HDF5 access (this might take longer)
verbose_log "Initiating Test 2: HDF5 file access validation"
echo "Test 2 - HDF5 file access (this may take a minute)..."

verbose_log "Preparing MATLAB batch command for HDF5 access test"
verbose_log "Target file: /vast/palmer/scratch/emonet/snb6/plume/smoke_1a_orig_backgroundsubtracted_rotated.h5"

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

test2_result=${PIPESTATUS[0]}

if [ $test2_result -eq 0 ]; then
    verbose_log "Test 2 PASSED: HDF5 file access successful"
else
    verbose_log "Test 2 FAILED: HDF5 file access failed with exit code $test2_result"
fi

verbose_log "Test execution completed, detailed results saved to matlab_hdf5_test.log"
echo "Check matlab_hdf5_test.log for details"

# Final summary with verbose logging
overall_result=0
if [ $test1_result -ne 0 ] || [ $test2_result -ne 0 ]; then
    overall_result=1
fi

if [[ $VERBOSE -eq 1 ]]; then
    verbose_log "=== Test Summary ==="
    verbose_log "Test 1 (MATLAB startup): $([ $test1_result -eq 0 ] && echo 'PASSED' || echo 'FAILED')"
    verbose_log "Test 2 (HDF5 access): $([ $test2_result -eq 0 ] && echo 'PASSED' || echo 'FAILED')"
    verbose_log "Overall result: $([ $overall_result -eq 0 ] && echo 'SUCCESS' || echo 'FAILURE')"
    verbose_log "MATLAB testing workflow completed"
fi

exit $overall_result