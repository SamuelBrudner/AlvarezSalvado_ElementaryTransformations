addpath(fullfile(pwd, 'Code'));
cfg = load_config(fullfile('configs', 'my_complex_plume_config.yaml'));
run_navigation_cfg(cfg);
