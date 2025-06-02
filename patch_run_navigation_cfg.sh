#!/bin/bash
# Update run_navigation_cfg.m to use load_plume_hdf5_info

# Backup first
cp Code/run_navigation_cfg.m Code/run_navigation_cfg.m.bak

# Replace load_plume_hdf5 with load_plume_hdf5_info
sed -i 's/load_plume_hdf5(/load_plume_hdf5_info(/g' Code/run_navigation_cfg.m

# Also update load_custom_plume to use the info loader
sed -i 's/plume = load_plume_hdf5(/plume = load_plume_hdf5_info(/g' Code/load_custom_plume.m

echo "Updated to use load_plume_hdf5_info"
