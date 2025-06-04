#!/bin/bash
# fix_validate_plume_colorbar.sh - Fix the ColorBar label issue

echo "Fixing ColorBar issue in validate_plume_setup.m..."

# Fix the colorbar line in the file
sed -i.backup "s/cbar = colorbar('Label', 'Odor Concentration');/cbar = colorbar(); ylabel(cbar, 'Odor Concentration');/" Code/validate_plume_setup.m

echo "✓ Fixed ColorBar syntax"

# Also create a simpler test that doesn't use inline functions
cat > test_validation_simple.m << 'EOF'
% test_validation_simple.m - Simple test without inline functions

fprintf('\n=== Simple Validation Test ===\n');
addpath(genpath('Code'));

% Test creating validation figure
fprintf('Creating validation figure...\n');
try
    validate_plume_setup('test_validation_simple.png');
    fprintf('✓ Success! Figure saved to test_validation_simple.png\n');
catch ME
    fprintf('✗ Error: %s\n', ME.message);
    fprintf('  In: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
end

fprintf('\nTest complete.\n');
exit;
EOF

echo "✓ Created simple test script"
echo ""
echo "To test the fix:"
echo "  matlab -nodisplay -r \"run('test_validation_simple.m')\""