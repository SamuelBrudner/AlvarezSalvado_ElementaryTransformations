#!/bin/bash
# run_direct.sh - Run MATLAB with explicit path setup

cd /home/snb6/Documents/AlvarezSalvado_ElementaryTransformations

matlab -nodisplay -nosplash -r "
% Add Code directory first
addpath(genpath('Code'));

% Now run the test
try
    run('test_simple.m');
catch ME
    fprintf('Error: %s\n', ME.message);
end
exit;
"
