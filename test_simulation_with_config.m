addpath('Code');

fprintf('Testing navigation model with config file...\n\n');

% Load config
cfg = load_config('configs/test_inline_comments.yaml');

% Run a quick test simulation
fprintf('Running navigation model with:\n');
fprintf('  environment: %s\n', cfg.environment);
fprintf('  ntrials: %d\n', cfg.ntrials);
fprintf('  triallength: %d\n', cfg.triallength);

try
    % Run the navigation model
    result = run_navigation_cfg(cfg);
    fprintf('\nSimulation completed successfully!\n');
    fprintf('Result fields: %s\n', strjoin(fieldnames(result), ', '));
catch ME
    fprintf('\nError in simulation: %s\n', ME.message);
    fprintf('Error occurred in: %s\n', ME.stack(1).name);
end

exit;
