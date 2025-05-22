% RUN_MY_SIMULATION Run the navigation model using the default configuration.
%
% This script adds the Code directory to the MATLAB path based on its own
% location so it works even when MATLAB is launched from another directory,
% for example within SLURM batch jobs.

thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);

cfg = load_config(fullfile(thisDir, '..', 'configs', 'my_complex_plume_config.yaml'));
run_navigation_cfg(cfg);
