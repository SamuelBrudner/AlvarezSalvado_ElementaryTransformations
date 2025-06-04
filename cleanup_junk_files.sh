#!/bin/bash
# cleanup_junk_files.sh - Remove temporary files from debugging

echo "=== Cleaning Up Temporary Files ==="

# Remove temporary MATLAB scripts
echo "Removing temporary scripts..."
rm -f check_all_dimensions.m
rm -f check_dimensions.m
rm -f check_arena_calc.m
rm -f check_json_raw.m
rm -f fix_configs_direct.m
rm -f fix_smoke_coordinates.m
rm -f fix_smoke_like_crimaldi.m
rm -f generate_minimal_configs.m
rm -f fix_existing_configs.m
rm -f fix_smoke_*.m
rm -f smoke_test*.m
rm -f test_smoke*.m
rm -f visualize_smoke*.m
rm -f run_smoke_*.m
rm -f check_smoke_*.m

# Remove backup and temporary configs
echo "Removing temporary configs..."
rm -f configs/plumes/*_FIXED.json
rm -f configs/plumes/*_CORRECT.json
rm -f configs/plumes/*_minimal.json
rm -f configs/plumes/*.backup*
rm -f smoke_*.sh

# Remove temporary result files
echo "Removing temporary results..."
rm -f results/smoke_nav_results_TEST.mat
rm -f results/smoke_nav_results_PROPER.mat
rm -f results/smoke_nav_results_CORRECTED.mat
rm -f results/smoke_nav_results_CRIMALDI_STYLE.mat
rm -f results/nav_results_0001.mat  # Keep only 0000

# Remove broken adaptive model
rm -f Code/Elifenavmodel_bilateral_adaptive.m
rm -f Code/Elifenavmodel_bilateral_smoke_temp.m

# Remove temporary plots
rm -f smoke_*.png
rm -f both_plumes_comparison.png  # Keep if you want

echo ""
echo "✓ Cleanup complete!"
echo ""
echo "Essential files remaining:"
echo "  configs/plumes/crimaldi_10cms_bounded.json"
echo "  configs/plumes/smoke_1a_backgroundsubtracted.json"
echo "  generate_clean_configs.m"
echo "  plot_both_plumes.m"
echo "  run_both_plumes_test.m"
echo "  results/nav_results_0000.mat"
echo "  results/smoke_nav_results_1000.mat (if exists)"