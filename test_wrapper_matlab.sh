#!/bin/bash

echo "Creating minimal wrapper test..."

# Create temp dir
TEMP_DIR="./temp_test_$$"
mkdir -p "$TEMP_DIR"

# Create simple MATLAB script
cat > "$TEMP_DIR/test_startup.m" << 'MATLAB_EOF'
fprintf('\n=== MATLAB STARTED at %s ===\n', datestr(now));
fprintf('Working directory: %s\n', pwd);
fprintf('MATLAB version: %s\n', version);
fprintf('Pausing for 2 seconds to simulate work...\n');
pause(2);
fprintf('Done! Exiting cleanly.\n');
exit(0);
MATLAB_EOF

# Create wrapper
cat > "$TEMP_DIR/run_test.sh" << 'WRAPPER_EOF'
#!/bin/bash
echo "Wrapper: Starting MATLAB..."
matlab -nodisplay -nosplash -r "run('$1')" 2>&1
MATLAB_EXIT=$?
echo "Wrapper: MATLAB exited with code $MATLAB_EXIT"
exit $MATLAB_EXIT
WRAPPER_EOF

chmod +x "$TEMP_DIR/run_test.sh"

# Run it
echo "Running wrapper test..."
"$TEMP_DIR/run_test.sh" "$TEMP_DIR/test_startup.m"

# Cleanup
rm -rf "$TEMP_DIR"
