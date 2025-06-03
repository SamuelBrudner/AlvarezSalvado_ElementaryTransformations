#!/bin/bash
# show_problem.sh - Show exactly what's wrong

echo "=== Current State Analysis ==="

echo -e "\n1. Checking around 'tic' in navigation_model_vec.m:"
echo "----------------------------------------"
grep -A 15 "^tic$" Code/navigation_model_vec.m | grep -B 2 -A 13 "tic"

echo -e "\n\n2. Checking if tscale is defined elsewhere:"
echo "----------------------------------------"
grep -n "tscale =" Code/navigation_model_vec.m

echo -e "\n\n3. The problem is:"
echo "----------------------------------------"
if grep -q "LOAD PLUME CONFIGURATION" Code/navigation_model_vec.m; then
    echo "✗ Config loading was added but the 'else' clause doesn't set tscale/pxscale"
    echo "  for non-Crimaldi environments"
else
    echo "✗ The original tscale/pxscale definitions may have been removed"
fi

echo -e "\n\n4. Quick manual fix:"
echo "----------------------------------------"
echo "Edit Code/navigation_model_vec.m and ensure this structure exists after 'tic':"
echo ""
echo "tic"
echo "% Scaling factors"
echo "if strcmpi(environment, 'Crimaldi') || strcmpi(environment, 'crimaldi')"
echo "    [plume_filename, plume_config] = get_plume_file();"
echo "    tscale = plume_config.time_scale_50hz;"
echo "    pxscale = plume_config.pixel_scale;"
echo "    % ... other config assignments ..."
echo "else"
echo "    % DEFAULT VALUES FOR NON-CRIMALDI"
echo "    tscale = 15/50;"
echo "    pxscale = 0.74;"
echo "    plume_xlims = [1 216];"
echo "    plume_ylims = [1 406];"
echo "    dataset_name = '/dataset2';"
echo "end"