#!/bin/bash
# Set plume file path explicitly
export MATLAB_PLUME_FILE="/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations/data/plumes/10302017_10cms_bounded.hdf5"

echo "Testing with MATLAB_PLUME_FILE=$MATLAB_PLUME_FILE"

matlab -nodisplay -nosplash -r "
addpath(genpath('Code'));

% Check for environment variable override
plume_override = getenv('MATLAB_PLUME_FILE');
if ~isempty(plume_override)
    fprintf('Using plume file override: %s\n', plume_override);
    % Temporarily modify get_plume_file
    plume_file = plume_override;
else
    [plume_file, ~] = get_plume_file();
end

if exist(plume_file, 'file')
    fprintf('✓ File exists!\n');
    out = navigation_model_vec(100, 'Crimaldi', 0, 1);
    fprintf('✓ Crimaldi works!\n');
else
    fprintf('✗ File not found\n');
end
exit;
" 2>&1 | tail -10
