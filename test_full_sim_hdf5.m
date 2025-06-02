addpath('Code');

% Load config
cfg = load_config('configs/smoke_plume_hdf5.yaml');

fprintf('Running full simulation with HDF5 plume...\n');
fprintf('  environment: %s\n', cfg.environment);
fprintf('  plume_metadata: %s\n', cfg.plume_metadata);
fprintf('  ntrials: %d\n', cfg.ntrials);
fprintf('  triallength: %d (%.1f seconds at 60 fps)\n', cfg.triallength, cfg.triallength/60);

tic;
try
    % Run the navigation model
    result = run_navigation_cfg(cfg);
    elapsed = toc;
    
    fprintf('\nSimulation completed successfully in %.2f seconds!\n', elapsed);
    fprintf('Result fields: %s\n', strjoin(fieldnames(result), ', '));
    
    % Check the output
    if isfield(result, 'x') && isfield(result, 'y')
        fprintf('Generated %d timesteps for %d trials\n', size(result.x, 1), size(result.x, 2));
        fprintf('Average trajectory length: %.2f cm\n', mean(sum(sqrt(diff(result.x).^2 + diff(result.y).^2))));
    end
    
    % Show odor statistics
    fprintf('\nOdor statistics:\n');
    fprintf('  Min: %.4f, Max: %.4f\n', min(result.odor(:)), max(result.odor(:)));
    fprintf('  Mean: %.4f, Std: %.4f\n', mean(result.odor(:)), std(result.odor(:)));
    
catch ME
    fprintf('\nError in simulation: %s\n', ME.message);
end

exit;
