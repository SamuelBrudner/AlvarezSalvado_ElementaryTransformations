#!/bin/bash
# run_matlab_safe.sh - Safely run MATLAB code using temporary files
#
# Usage: 
#   run_matlab_safe.sh script.m
#   echo "matlab code" | run_matlab_safe.sh
#   run_matlab_safe.sh << 'EOF'
#     matlab code here
#   EOF

set -euo pipefail

# Get project directory
PROJECT_DIR="/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations"

# Create temp file for MATLAB code
TEMP_SCRIPT=$(mktemp /tmp/matlab_safe_XXXXXX.m)
trap "rm -f $TEMP_SCRIPT" EXIT

# Prepare MATLAB code with proper setup
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
if [ $# -eq 1 ] && [ -f "$1" ]; then
    # File provided
    cat "$1" >> "$TEMP_SCRIPT"
elif [ $# -eq 0 ]; then
    # Read from stdin
    cat >> "$TEMP_SCRIPT"
else
    echo "Usage: $0 [script.m]" >&2
    exit 1
fi

# Add error handling footer
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
echo "Running MATLAB code..."
matlab -nodisplay -nosplash < "$TEMP_SCRIPT" 2>&1 | grep -v "^>>" | grep -v "^$"

# Exit with MATLAB's exit code
exit ${PIPESTATUS[0]}