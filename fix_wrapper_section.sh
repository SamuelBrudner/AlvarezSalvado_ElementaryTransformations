#!/bin/bash

# Find and replace the wrapper script creation section
awk '
BEGIN { in_wrapper = 0; }
/^    # Create a wrapper script to handle timeouts and errors$/ {
    print
    print "    cat > \"$TEMP_DIR/run_analysis.sh\" << '\''WRAPPER_EOF'\''"
    print "#!/bin/bash"
    print "MATLAB_SCRIPT=\"$1\""
    print "TIMEOUT_SECONDS='$TIMEOUT_SECONDS'"
    print ""
    print "echo \"Running MATLAB analysis with ${TIMEOUT_SECONDS}s timeout...\""
    print "echo \"  Mode: '$( if [ \"$RUN_ANALYSIS\" == \"quick\" ]; then echo \"QUICK (10 frames)\"; else echo \"FULL (100 frames)\"; fi )'\""
    print ""
    print "# Run MATLAB with timeout"
    print "if command -v timeout &> /dev/null; then"
    print "    # GNU coreutils timeout available"
    print "    timeout ${TIMEOUT_SECONDS}s matlab -nodisplay -nosplash -r \"run('\''${MATLAB_SCRIPT}'\'')\" 2>&1"
    print "    MATLAB_EXIT=$?"
    print "else"
    print "    # Use perl for timeout if GNU timeout not available"
    print "    perl -e \"alarm ${TIMEOUT_SECONDS}; exec @ARGV\" matlab -nodisplay -nosplash -r \"run('\''${MATLAB_SCRIPT}'\'')\" 2>&1"
    print "    MATLAB_EXIT=$?"
    print "fi"
    print ""
    print "if [ $MATLAB_EXIT -eq 124 ] || [ $MATLAB_EXIT -eq 142 ]; then"
    print "    echo \"\""
    print "    echo \"ERROR: MATLAB analysis timed out after ${TIMEOUT_SECONDS} seconds!\""
    print "    echo \"Possible causes:\""
    print "    echo \"  - HDF5 file is too large or on slow storage\""
    print "    echo \"  - MATLAB is waiting for user input\""
    print "    echo \"  - Code directory not found\""
    print "    exit 1"
    print "elif [ $MATLAB_EXIT -ne 0 ]; then"
    print "    echo \"\""
    print "    echo \"ERROR: MATLAB exited with code $MATLAB_EXIT\""
    print "    exit $MATLAB_EXIT"
    print "fi"
    print "WRAPPER_EOF"
    in_wrapper = 1
    next
}
/^WRAPPER_EOF$/ && in_wrapper {
    in_wrapper = 0
    next
}
!in_wrapper { print }
' setup_smoke_plume_config.sh > setup_smoke_plume_config.tmp && mv setup_smoke_plume_config.tmp setup_smoke_plume_config.sh

chmod +x setup_smoke_plume_config.sh
echo "Wrapper section replaced!"
