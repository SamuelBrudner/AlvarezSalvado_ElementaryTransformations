#!/bin/bash

# Find where the wrapper script is created and replace it entirely
awk '
/# Create a wrapper script to handle timeouts and errors/ {
    print "    # Create a wrapper script to handle timeouts and errors"
    print "    cat > \"$TEMP_DIR/run_analysis.sh\" << WRAPPER_EOF"
    print "#!/bin/bash"
    print "MATLAB_SCRIPT=\"\\$1\""
    print ""
    print "echo \"Running MATLAB analysis with ${TIMEOUT_SECONDS}s timeout...\""
    print "if [ \"${RUN_ANALYSIS}\" == \"quick\" ]; then"
    print "    echo \"  Mode: QUICK (10 frames)\""
    print "else"
    print "    echo \"  Mode: FULL (100 frames)\""
    print "fi"
    print ""
    print "# Run MATLAB with timeout"
    print "if command -v timeout &> /dev/null; then"
    print "    timeout ${TIMEOUT_SECONDS}s matlab -nodisplay -nosplash -r \"run('"'"'\\${MATLAB_SCRIPT}'"'"')\" 2>&1"
    print "    MATLAB_EXIT=\\$?"
    print "else"
    print "    perl -e \"alarm ${TIMEOUT_SECONDS}; exec @ARGV\" matlab -nodisplay -nosplash -r \"run('"'"'\\${MATLAB_SCRIPT}'"'"')\" 2>&1"
    print "    MATLAB_EXIT=\\$?"
    print "fi"
    print ""
    print "if [ \\$MATLAB_EXIT -eq 124 ] || [ \\$MATLAB_EXIT -eq 142 ]; then"
    print "    echo \"\""
    print "    echo \"ERROR: MATLAB analysis timed out after ${TIMEOUT_SECONDS} seconds!\""
    print "    echo \"Possible causes:\""
    print "    echo \"  - HDF5 file is too large or on slow storage\""
    print "    echo \"  - Try running with '"'"'quick'"'"' mode or '"'"'n'"'"' to skip\""
    print "    exit 1"
    print "elif [ \\$MATLAB_EXIT -ne 0 ]; then"
    print "    echo \"\""
    print "    echo \"ERROR: MATLAB exited with code \\$MATLAB_EXIT\""
    print "    exit \\$MATLAB_EXIT"
    print "fi"
    print "WRAPPER_EOF"
    # Skip until we find WRAPPER_EOF
    while (getline && $0 !~ /^WRAPPER_EOF/) {}
    next
}
{ print }
' setup_smoke_plume_config.sh > setup_smoke_plume_config.new && mv setup_smoke_plume_config.new setup_smoke_plume_config.sh

chmod +x setup_smoke_plume_config.sh
echo "Wrapper creation fixed!"
