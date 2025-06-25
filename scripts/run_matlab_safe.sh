#!/bin/bash
# run_matlab_safe.sh - Safely run MATLAB code using temporary files
#
# Usage: 
#   run_matlab_safe.sh [-v|--verbose] script.m
#   echo "matlab code" | run_matlab_safe.sh [-v|--verbose]
#   run_matlab_safe.sh [-v|--verbose] << 'EOF'
#     matlab code here
#   EOF
#
# Options:
#   -v, --verbose    Enable verbose logging output

set -euo pipefail

# Initialize verbose flag
VERBOSE=0

# Function for verbose logging
log_verbose() {
    if [[ $VERBOSE -eq 1 ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_matlab_safe: $*" >&2
        # Also log to file if logs directory exists
        if [[ -d "logs" ]]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_matlab_safe: $*" >> "logs/run_matlab_safe.log"
        fi
    fi
}

# Parse command line arguments
MATLAB_SCRIPT=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=1
            log_verbose "Verbose mode enabled"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [-v|--verbose] [script.m]"
            echo "Options:"
            echo "  -v, --verbose    Enable verbose logging output"
            echo "  -h, --help       Show this help message"
            exit 0
            ;;
        -*)
            echo "Unknown option: $1" >&2
            echo "Use -h or --help for usage information" >&2
            exit 1
            ;;
        *)
            if [[ -z "$MATLAB_SCRIPT" ]]; then
                MATLAB_SCRIPT="$1"
            else
                echo "Too many arguments. Use -h or --help for usage information" >&2
                exit 1
            fi
            shift
            ;;
    esac
done

log_verbose "Starting MATLAB safe execution wrapper"

# Get project directory
PROJECT_DIR="/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations"
log_verbose "Project directory: $PROJECT_DIR"

# Create temp file for MATLAB code
TEMP_SCRIPT=$(mktemp /tmp/matlab_safe_XXXXXX.m)
log_verbose "Created temporary MATLAB script: $TEMP_SCRIPT"
trap "log_verbose 'Cleaning up temporary file: $TEMP_SCRIPT'; rm -f $TEMP_SCRIPT" EXIT

# Prepare MATLAB code with proper setup
log_verbose "Writing MATLAB header with error handling"
cat > "$TEMP_SCRIPT" << 'MATLAB_HEADER'
% Auto-generated safe MATLAB script
try
    % Ensure we're in the project directory
    project_dir = '/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations';
    if exist(project_dir, 'dir')
        cd(project_dir);
    end
    
    % Add Code to path if it exists
    code_dir = fullfile(pwd, 'Code');
    if exist(code_dir, 'dir')
        addpath(genpath(code_dir));
    end
    
    % User code begins here
    % ---------------------
MATLAB_HEADER

# Add user code
if [[ -n "$MATLAB_SCRIPT" ]] && [[ -f "$MATLAB_SCRIPT" ]]; then
    # File provided
    log_verbose "Reading MATLAB code from file: $MATLAB_SCRIPT"
    cat "$MATLAB_SCRIPT" >> "$TEMP_SCRIPT"
elif [[ -z "$MATLAB_SCRIPT" ]]; then
    # Read from stdin
    log_verbose "Reading MATLAB code from stdin"
    cat >> "$TEMP_SCRIPT"
elif [[ -n "$MATLAB_SCRIPT" ]]; then
    echo "Error: File '$MATLAB_SCRIPT' not found" >&2
    exit 1
else
    echo "Usage: $0 [-v|--verbose] [script.m]" >&2
    exit 1
fi

# Add error handling footer
log_verbose "Adding MATLAB error handling footer"
cat >> "$TEMP_SCRIPT" << 'MATLAB_FOOTER'
    
    % ---------------------
    % User code ends here
    
catch ME
    fprintf('\n!!! MATLAB ERROR !!!\n');
    fprintf('Identifier: %s\n', ME.identifier);
    fprintf('Message: %s\n', ME.message);
    fprintf('\nStack trace:\n');
    for i = 1:length(ME.stack)
        fprintf('  In %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
    exit(1);
end

% Successful completion
exit(0);
MATLAB_FOOTER

# Run MATLAB
log_verbose "Executing MATLAB with temporary script"
if [[ $VERBOSE -eq 1 ]]; then
    echo "Running MATLAB code..."
else
    echo "Running MATLAB code..."
fi

# Execute MATLAB and capture exit code properly
log_verbose "Starting MATLAB with options: -nodisplay -nosplash"
matlab -nodisplay -nosplash < "$TEMP_SCRIPT" 2>&1 | grep -v "^>>" | grep -v "^$"
MATLAB_EXIT_CODE=${PIPESTATUS[0]}

log_verbose "MATLAB execution completed with exit code: $MATLAB_EXIT_CODE"

# Exit with MATLAB's exit code
if [[ $MATLAB_EXIT_CODE -eq 0 ]]; then
    log_verbose "MATLAB script executed successfully"
else
    log_verbose "MATLAB script failed with exit code: $MATLAB_EXIT_CODE"
fi

exit $MATLAB_EXIT_CODE