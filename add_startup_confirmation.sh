#!/bin/bash

# Add a startup message right at the beginning of the MATLAB script
perl -i -pe '
# Find the beginning of the MATLAB script and add startup confirmation
if (/^% Comprehensive analysis script with error handling$/) {
    $_ .= "fprintf('"'"'\\n=== MATLAB STARTED SUCCESSFULLY at %s ===\\n'"'"', datestr(now));\n";
    $_ .= "fprintf('"'"'Running from: %s\\n'"'"', pwd);\n";
    $_ .= "fprintf('"'"'MATLAB version: %s\\n\\n'"'"', version);\n";
}

# Also add a message right before the try block
if (/^try$/ && !$done_try) {
    print "fprintf('"'"'Starting analysis try block...\\n'"'"');\n";
    $done_try = 1;
}
' setup_smoke_plume_config.sh

echo "Startup confirmation added!"
