#!/bin/bash

# Find where MATLAB_CMD is used and fix it
sed -i 's/\$MATLAB_CMD -nodisplay/matlab -nodisplay/g' setup_smoke_plume_config.sh

# Or better, let's properly initialize MATLAB_CMD
perl -i -pe '
# Remove the previous stdbuf check if it exists
if (/# Check if we can unbuffer output/) {
    # Skip this section
    while (<>) {
        last if /^fi$/;
    }
    next;
}

# Add proper MATLAB command setup at the beginning of the wrapper
if (/^#!/ && !$done) {
    $_ .= "MATLAB_SCRIPT=\"\$1\"\n";
    $_ .= "MATLAB_CMD=\"matlab\"\n";
    $_ .= "\n";
    $done = 1;
    next;
}

# Remove the MATLAB_SCRIPT line if it appears later
next if /^MATLAB_SCRIPT="\$1"$/;
' setup_smoke_plume_config.sh

echo "MATLAB_CMD issue fixed!"
