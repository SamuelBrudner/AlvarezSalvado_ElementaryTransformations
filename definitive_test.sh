#!/bin/bash
# definitive_test.sh - Find out exactly what works

echo "=== DEFINITIVE TEST - What Works? ==="
echo ""
echo "This will test all options and tell you exactly what to use."
echo ""

# Ensure we're in the right place
cd /home/snb6/Documents/AlvarezSalvado_ElementaryTransformations 2>/dev/null || {
    echo "ERROR: Not in the right directory!"
    echo "Run: cd /home/snb6/Documents/AlvarezSalvado_ElementaryTransformations"
    exit 1
}

# Test each option
cat > definitive_test.m << 'EOF'
% Definitive test of all options
addpath(genpath('Code'));

fprintf('\n=== TESTING ALL OPTIONS ===\n\n');

results = struct();

% TEST 1: Gaussian with fixed duration
fprintf('1. Gaussian (15000 samples = 300s):\n');
try
    tic;
    out = navigation_model_vec(15000, 'gaussian', 0, 2);
    t = toc;
    fprintf('   ✓ WORKS! %d agents in %.1f seconds\n', size(out.x,2), t);
    results.gaussian_fixed = true;
catch ME
    fprintf('   ✗ FAILED: %s\n', ME.message);
    results.gaussian_fixed = false;
end

% TEST 2: Gaussian with 'config'
fprintf('\n2. Gaussian with config duration:\n');
try
    tic;
    out = navigation_model_vec('config', 'gaussian', 0, 2);
    t = toc;
    fprintf('   ✓ WORKS! %d samples = %.1f seconds duration\n', size(out.x,1), size(out.x,1)/50);
    results.gaussian_config = true;
catch ME
    fprintf('   ✗ FAILED: %s\n', ME.message);
    results.gaussian_config = false;
end

% TEST 3: Config loading
fprintf('\n3. Config file loading:\n');
try
    [pf, pc] = get_plume_file();
    fprintf('   ✓ Config loads\n');
    fprintf('   Plume path: %s\n', pf);
    fprintf('   File exists: %s\n', iif(exist(pf,'file'), 'YES', 'NO'));
    results.config_loads = true;
    results.plume_exists = exist(pf,'file');
catch ME
    fprintf('   ✗ Config fails: %s\n', ME.message);
    results.config_loads = false;
    results.plume_exists = false;
end

% TEST 4: Crimaldi if possible
if results.plume_exists
    fprintf('\n4. Crimaldi simulation:\n');
    try
        tic;
        out = navigation_model_vec(4500, 'Crimaldi', 0, 2);
        t = toc;
        fprintf('   ✓ WORKS! %d agents in %.1f seconds\n', size(out.x,2), t);
        results.crimaldi_works = true;
    catch ME
        fprintf('   ✗ FAILED: %s\n', ME.message);
        results.crimaldi_works = false;
    end
else
    fprintf('\n4. Skipping Crimaldi (no plume file)\n');
    results.crimaldi_works = false;
end

% RECOMMENDATIONS
fprintf('\n\n=== RECOMMENDATIONS ===\n');

if results.gaussian_config
    fprintf('\n✓ USE THIS:\n');
    fprintf('  out = navigation_model_vec(''config'', ''gaussian'', 0, 10);\n');
    fprintf('  (Will use 300s duration from config)\n');
elseif results.gaussian_fixed
    fprintf('\n✓ USE THIS:\n');
    fprintf('  out = navigation_model_vec(15000, ''gaussian'', 0, 10);\n');
    fprintf('  (300 seconds at 50 Hz)\n');
else
    fprintf('\n✗ PROBLEM: Nothing works! Check your Code/ directory.\n');
end

if results.crimaldi_works
    fprintf('\n✓ ALSO AVAILABLE:\n');
    fprintf('  out = navigation_model_vec(4500, ''Crimaldi'', 0, 10);\n');
    fprintf('  (300 seconds at 15 Hz)\n');
end

fprintf('\n=== END OF TEST ===\n');

function r = iif(c,t,f)
    if c, r=t; else, r=f; end
end
EOF

# Run the test
matlab -nodisplay -nosplash -r "definitive_test; exit" 2>&1 | tail -50

# Clean up
rm -f definitive_test.m

echo ""
echo "========================================="
echo "LOOK ABOVE FOR '✓ USE THIS:'"
echo "That's the command you should use!"
echo "========================================="