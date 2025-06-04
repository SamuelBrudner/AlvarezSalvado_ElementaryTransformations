#!/bin/bash
#SBATCH --job-name=smoke_simple
#SBATCH --partition=day
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/smoke_simple-%j.out
#SBATCH --error=logs/smoke_simple-%j.err

module load MATLAB/2023b
cd /home/snb6/Documents/AlvarezSalvado_ElementaryTransformations

matlab -nodisplay -nosplash << 'MATLAB'
addpath(genpath('Code'));

% Direct smoke configuration
setenv('MATLAB_PLUME_FILE', '/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations/data/plumes/smoke_1a_rotated_3d_fixed.h5');

% Run with navigation_model_vec which handles configs better
fprintf('Running smoke simulation...\n');
out = navigation_model_vec(18000, 'Crimaldi', 0, 10);

% Save result
save('results/smoke_nav_results_1000.mat', 'out', '-v7.3');
fprintf('Success rate: %.1f%%\n', out.successrate * 100);
MATLAB
