#!/bin/bash
MATLAB_SCRIPT="$1"

echo "Running MATLAB analysis with 600s timeout..."
echo "  Mode: 100 frames"

# Run MATLAB with timeout
if command -v timeout &> /dev/null; then
    timeout 600s matlab -nodisplay -nosplash -r "run('${MATLAB_SCRIPT}')" 2>&1
    MATLAB_EXIT=$?
else
    perl -e "alarm 600; exec @ARGV" matlab -nodisplay -nosplash -r "run('${MATLAB_SCRIPT}')" 2>&1
    MATLAB_EXIT=$?
fi

if [ $MATLAB_EXIT -eq 124 ] || [ $MATLAB_EXIT -eq 142 ]; then
    echo ""
    echo "ERROR: MATLAB analysis timed out after 600 seconds!"
    echo "Try using 'quick' mode or 'n' to skip analysis"
    exit 1
elif [ $MATLAB_EXIT -ne 0 ]; then
    echo ""
    echo "ERROR: MATLAB exited with code $MATLAB_EXIT"
    exit $MATLAB_EXIT
fi
