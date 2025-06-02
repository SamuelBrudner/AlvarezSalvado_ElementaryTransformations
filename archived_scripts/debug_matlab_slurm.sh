#!/bin/bash
#SBATCH --job-name=debug_matlab
#SBATCH --partition=day
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --output=debug_matlab_%j.out
#SBATCH --error=debug_matlab_%j.err

echo "=== MATLAB Command Debug Script ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"
echo ""

# Test 1: Check MATLAB availability
echo "Test 1: Checking MATLAB installation"
which matlab || echo "MATLAB not found in PATH"
matlab -nodisplay -r "version; exit" || echo "Failed to run MATLAB"
echo ""

# Test 2: Test simple command
echo "Test 2: Testing simple MATLAB command"
matlab -nodisplay -r "disp('Hello from MATLAB'); exit"
echo ""

# Test 3: Test with quotes
echo "Test 3: Testing command with quotes"
matlab -nodisplay -r "addpath('Code'); exit"
echo ""

# Test 4: Test with full command using variable
echo "Test 4: Testing full command with variable"
CMD="addpath(genpath('Code')); disp('Path added successfully'); exit"
echo "Command string: $CMD"
matlab -nodisplay -r "$CMD"
echo ""

# Test 5: Test using stdin
echo "Test 5: Testing with stdin"
matlab -nodisplay << 'EOF'
disp('Testing stdin method');
addpath(genpath('Code'));
disp('Path added via stdin');
exit
EOF
echo ""

# Test 6: Test with script file
echo "Test 6: Testing with script file"
cat > test_matlab.m << 'EOF'
disp('Testing script file method');
addpath(genpath('Code'));
disp('Path added via script');
exit
EOF
matlab -nodisplay < test_matlab.m
rm -f test_matlab.m
echo ""

echo "=== Debug script completed ==="