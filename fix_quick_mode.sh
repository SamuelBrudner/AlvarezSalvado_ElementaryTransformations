#!/bin/bash

# The issue is that QUICK_ANALYSIS is set but RUN_ANALYSIS is being checked
# Let's fix the logic in Step 4
perl -i -pe '
# Fix the analysis mode detection in Step 4
if (/if \[\[ "\$RUN_ANALYSIS" == "quick" \]\]; then/) {
    $_ = "    if [[ \"\$QUICK_ANALYSIS\" == \"1\" ]]; then\n";
}
' setup_smoke_plume_config.sh

echo "Quick mode detection fixed!"
