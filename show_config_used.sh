#!/bin/bash
# show_config_used.sh - Show which config is being used

echo "=== Current Config Status ==="

# Check PLUME_CONFIG env var
if [ -z "$PLUME_CONFIG" ]; then
    echo "PLUME_CONFIG: not set (will use default)"
    DEFAULT_CONFIG="configs/plumes/crimaldi_10cms_bounded.json"
    echo "Default: $DEFAULT_CONFIG"
    if [ -f "$DEFAULT_CONFIG" ]; then
        echo "✓ Default config exists"
    else
        echo "✗ Default config missing!"
    fi
else
    echo "PLUME_CONFIG: $PLUME_CONFIG"
    if [ -f "$PLUME_CONFIG" ]; then
        echo "✓ Config file exists"
    else
        echo "✗ Config file missing!"
    fi
fi

# Test loading
echo -e "\nTesting config load:"
matlab -nodisplay -nosplash -nojvm -r "
addpath('Code');
try
    [pf, pc] = get_plume_file();
    fprintf('✓ Config loaded successfully\n');
    fprintf('  Frame rate: %.1f Hz\n', pc.frame_rate);
    fprintf('  Scale: %.3f mm/px\n', pc.mm_per_pixel);
catch ME
    fprintf('✗ Failed to load config\n');
    fprintf('  Error: %s\n', ME.message);
end
exit
" 2>&1 | grep -v ">>" | tail -10