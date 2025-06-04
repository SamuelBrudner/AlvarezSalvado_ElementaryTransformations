#!/bin/bash

# Fix the heredoc issue and clean up Step 4
perl -i -pe '
# Fix the wrapper heredoc back to quoted to prevent variable expansion
s/cat > "\$TEMP_DIR\/run_analysis.sh" << WRAPPER_EOF/cat > "\$TEMP_DIR\/run_analysis.sh" << '\''WRAPPER_EOF'\''/;

# Remove duplicate "Step 4: Running comprehensive plume analysis..." message
s/^echo "Step 4: Running comprehensive plume analysis\.\.\."$//;

# Remove the duplicate "This will sample 100 random frames" message
s/^echo "  This will sample 100 random frames and calculate statistics\.\.\."$//;
' setup_smoke_plume_config.sh

# Also fix the MATLAB script to use the variable correctly
sed -i 's/n_samples = min($N_SAMPLE_FRAMES, n_frames);/n_samples = min('$N_SAMPLE_FRAMES', n_frames);/' setup_smoke_plume_config.sh

# Fix the timeout line in wrapper to properly escape
sed -i '/timeout \${TIMEOUT_SECONDS}s matlab/ s/\${TIMEOUT_SECONDS}s/\$TIMEOUT_SECONDS"s"/' setup_smoke_plume_config.sh

echo "Fixes applied!"
