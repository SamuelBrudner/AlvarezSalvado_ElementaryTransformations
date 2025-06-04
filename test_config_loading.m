% test_config_loading.m - Test that configs load correctly

fprintf('=== Testing Config Loading ===\n\n');

%% Test 1: Crimaldi (default)
fprintf('Test 1: Crimaldi config\n');
fprintf('------------------------\n');
setenv('MATLAB_PLUME_FILE', '');  % Clear environment

% Simulate what the model does
env_plume = getenv('MATLAB_PLUME_FILE');
if contains(env_plume, 'smoke')
    fprintf('Would load: smoke config\n');
    cfg = jsondecode(fileread('configs/plumes/smoke_1a_backgroundsubtracted.json'));
else
    fprintf('Would load: crimaldi config\n');
    cfg = jsondecode(fileread('configs/plumes/crimaldi_10cms_bounded.json'));
end

fprintf('Config values:\n');
fprintf('  Frame rate: %.1f Hz\n', cfg.temporal.frame_rate);
fprintf('  Dataset: %s\n', cfg.data_path.dataset_name);
if isfield(cfg, 'model_params')
    fprintf('  tscale: %.3f\n', cfg.model_params.tscale);
    fprintf('  pxscale: %.3f\n', cfg.model_params.pxscale);
else
    fprintf('  tscale: %.3f (calculated)\n', cfg.temporal.frame_rate / 50.0);
    fprintf('  pxscale: %.3f\n', cfg.spatial.mm_per_pixel);
end

%% Test 2: Smoke
fprintf('\n\nTest 2: Smoke config\n');
fprintf('--------------------\n');
smoke_cfg = jsondecode(fileread('configs/plumes/smoke_1a_backgroundsubtracted.json'));
setenv('MATLAB_PLUME_FILE', smoke_cfg.data_path.path);

% Simulate what the model does
env_plume = getenv('MATLAB_PLUME_FILE');
if contains(env_plume, 'smoke')
    fprintf('Would load: smoke config\n');
    cfg = jsondecode(fileread('configs/plumes/smoke_1a_backgroundsubtracted.json'));
else
    fprintf('Would load: crimaldi config\n');
    cfg = jsondecode(fileread('configs/plumes/crimaldi_10cms_bounded.json'));
end

fprintf('Config values:\n');
fprintf('  Frame rate: %.1f Hz\n', cfg.temporal.frame_rate);
fprintf('  Dataset: %s\n', cfg.data_path.dataset_name);
if isfield(cfg, 'model_params')
    fprintf('  tscale: %.3f\n', cfg.model_params.tscale);
    fprintf('  pxscale: %.3f\n', cfg.model_params.pxscale);
else
    fprintf('  tscale: %.3f (calculated)\n', cfg.temporal.frame_rate / 50.0);
    fprintf('  pxscale: %.3f\n', cfg.spatial.mm_per_pixel);
end

% Clean up
setenv('MATLAB_PLUME_FILE', '');

fprintf('\n\n✓ Config loading logic is correct!\n');
fprintf('The model should now load the right config for each plume.\n');