% RUN_MY_SIMULATION Run the navigation model using the default configuration.
%
% This script relies on `startup.m` to add the Code directory to the MATLAB
% path.  It simply loads the default configuration and runs the navigation
% model.

thisDir = fileparts(mfilename('fullpath'));

cfg = load_config(fullfile(thisDir, '..', 'configs', 'my_complex_plume_config.yaml'));
run_navigation_cfg(cfg);
