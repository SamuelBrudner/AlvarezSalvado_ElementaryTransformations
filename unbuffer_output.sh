#!/bin/bash

# Modify the wrapper to use stdbuf for unbuffered output
sed -i 's/matlab -nodisplay -nosplash/stdbuf -o0 -e0 matlab -nodisplay -nosplash/g' setup_smoke_plume_config.sh

# If stdbuf is not available, use script command as alternative
perl -i -pe '
# Add check for stdbuf
if (/if command -v timeout &> \/dev\/null; then/) {
    print "# Check if we can unbuffer output\n";
    print "if command -v stdbuf &> /dev/null; then\n";
    print "    MATLAB_CMD=\"stdbuf -o0 -e0 matlab\"\n";
    print "else\n";
    print "    MATLAB_CMD=\"matlab\"\n";
    print "fi\n\n";
}

# Use MATLAB_CMD variable
s/matlab -nodisplay/\$MATLAB_CMD -nodisplay/g;
' setup_smoke_plume_config.sh

echo "Output unbuffering added!"
