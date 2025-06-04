#!/bin/bash

# Replace the entire wrapper script section with proper variable expansion
perl -i -0pe '
s/cat > "\$TEMP_DIR\/run_analysis.sh" << '\''WRAPPER_EOF'\''.*?WRAPPER_EOF/
cat > "\$TEMP_DIR\/run_analysis.sh" << WRAPPER_EOF
#!\/bin\/bash
MATLAB_SCRIPT="\\\$1"

echo "Running MATLAB analysis with ${TIMEOUT_SECONDS}s timeout..."
echo "  Mode: $(if [ "\$RUN_ANALYSIS" == "quick" ]; then echo "QUICK (10 frames)"; else echo "FULL (100 frames)"; fi)"

# Run MATLAB with timeout
if command -v timeout &> \/dev\/null; then
    # GNU coreutils timeout available
    timeout ${TIMEOUT_SECONDS}s matlab -nodisplay -nosplash -r "run('\''\\\${MATLAB_SCRIPT}'\'')" 2>&1
    MATLAB_EXIT=\\\$?
else
    # Use perl for timeout if GNU timeout not available
    perl -e "alarm ${TIMEOUT_SECONDS}; exec \@ARGV" matlab -nodisplay -nosplash -r "run('\''\\\${MATLAB_SCRIPT}'\'')" 2>&1
    MATLAB_EXIT=\\\$?
fi

if [ \\\$MATLAB_EXIT -eq 124 ] || [ \\\$MATLAB_EXIT -eq 142 ]; then
    echo ""
    echo "ERROR: MATLAB analysis timed out after ${TIMEOUT_SECONDS} seconds!"
    echo "Possible causes:"
    echo "  - HDF5 file is too large or on slow storage"
    echo "  - MATLAB is waiting for user input"
    echo "  - Code directory not found"
    exit 1
elif [ \\\$MATLAB_EXIT -ne 0 ]; then
    echo ""
    echo "ERROR: MATLAB exited with code \\\$MATLAB_EXIT"
    exit \\\$MATLAB_EXIT
fi
WRAPPER_EOF/gs' setup_smoke_plume_config.sh

# Also fix the n_samples line in MATLAB script
sed -i 's/n_samples = min('\''$N_SAMPLE_FRAMES'\'', n_frames);/n_samples = min('"'"'${N_SAMPLE_FRAMES}'"'"', n_frames);/' setup_smoke_plume_config.sh

echo "Fix applied!"
