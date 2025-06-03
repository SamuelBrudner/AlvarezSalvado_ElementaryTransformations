#!/bin/bash
# check_hpc_status.sh - Quick check of current HPC path status

echo "=== HPC Path Status Check ==="
echo ""

# Quick test
matlab -nodisplay -nosplash -nojvm -r "
addpath(genpath('Code'));
[pf, ~] = get_plume_file();
fprintf('Current plume path: %s\n', pf);
if exist(pf, 'file')
    fprintf('Status: FILE EXISTS ✓\n');
else
    fprintf('Status: FILE NOT FOUND ✗\n');
    % Try the symlink version
    pf2 = strrep(pf, '/vast/palmer/home.grace/snb6/', '/home/snb6/');
    if exist(pf2, 'file')
        fprintf('But exists at: %s\n', pf2);
        fprintf('→ Need to fix path resolution!\n');
    end
end
exit;
" 2>&1 | grep -E "(plume path:|Status:|But exists|Need to fix)"

echo ""
echo "Quick fixes:"
echo "1. Set env var:  export MATLAB_PLUME_FILE=\"/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations/data/plumes/10302017_10cms_bounded.hdf5\""
echo "2. Update config: sed -i 's|data/plumes|/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations/data/plumes|' configs/plumes/crimaldi_10cms_bounded.json"
echo "3. Use Gaussian: Just change 'Crimaldi' to 'gaussian' (no plume file needed)"