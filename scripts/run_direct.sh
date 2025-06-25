#!/bin/bash
# run_direct.sh - Run MATLAB with explicit path setup
# Relocated to scripts/ directory with enhanced verbose logging support

# Initialize verbose flag
VERBOSE=0

# Parse command line arguments for verbose flag
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        *)
            # Pass through any other arguments (for future extensibility)
            shift
            ;;
    esac
done

# Verbose logging function
log_verbose() {
    [[ $VERBOSE -eq 1 ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_direct.sh: $1"
}

# Start execution with verbose logging
log_verbose "Starting direct MATLAB execution script"
log_verbose "Verbose logging enabled"

# Change to the working directory
WORK_DIR="/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations"
log_verbose "Changing to working directory: $WORK_DIR"

cd "$WORK_DIR" || {
    echo "Error: Cannot change to directory $WORK_DIR" >&2
    exit 1
}

log_verbose "Successfully changed to working directory"
log_verbose "Setting up MATLAB execution environment"

# Run MATLAB with verbose logging
log_verbose "Executing MATLAB with nodisplay and nosplash options"
log_verbose "Adding Code directory to MATLAB path"
log_verbose "Running test_simple.m (located in matlab/ directory)"

matlab -nodisplay -nosplash -r "
% Add Code directory first
addpath(genpath('Code'));

% Log MATLAB execution start
fprintf('MATLAB execution started at %s\n', datestr(now));

% Now run the test
try
    fprintf('Running test_simple.m...\n');
    run('matlab/test_simple.m');
    fprintf('test_simple.m completed successfully\n');
catch ME
    fprintf('Error running test_simple.m: %s\n', ME.message);
    fprintf('Error details: %s\n', getReport(ME));
    exit(1);
end

fprintf('MATLAB execution completed at %s\n', datestr(now));
exit;
"

# Capture MATLAB exit status
MATLAB_EXIT_STATUS=$?

if [[ $MATLAB_EXIT_STATUS -eq 0 ]]; then
    log_verbose "MATLAB execution completed successfully"
else
    log_verbose "MATLAB execution failed with exit status: $MATLAB_EXIT_STATUS"
    echo "Error: MATLAB execution failed" >&2
    exit $MATLAB_EXIT_STATUS
fi

log_verbose "run_direct.sh execution completed successfully"