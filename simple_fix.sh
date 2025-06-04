#!/bin/bash

# Find the line that creates the wrapper script and ensure variables are expanded
sed -i "s/cat > \"\$TEMP_DIR\/run_analysis.sh\" << 'WRAPPER_EOF'/cat > \"\$TEMP_DIR\/run_analysis.sh\" << WRAPPER_EOF/" setup_smoke_plume_config.sh

# Fix the timeout line to ensure TIMEOUT_SECONDS is expanded
sed -i 's/timeout \${TIMEOUT_SECONDS}s/timeout ${TIMEOUT_SECONDS}s/g' setup_smoke_plume_config.sh

# Fix the mode display to use the correct variable check
sed -i 's/Mode: \$(if \[ \"\$RUN_ANALYSIS\" == \"quick\" \]/Mode: $(if [ "$RUN_ANALYSIS" == "quick" ]/g' setup_smoke_plume_config.sh

# Ensure N_SAMPLE_FRAMES is properly set in the MATLAB script
sed -i "s/n_samples = min('\$N_SAMPLE_FRAMES', n_frames);/n_samples = min(${N_SAMPLE_FRAMES:-100}, n_frames);/" setup_smoke_plume_config.sh

echo "Simple fix applied!"
