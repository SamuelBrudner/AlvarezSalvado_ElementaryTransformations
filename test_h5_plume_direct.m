addpath('Code');

% Create config using the HDF5 plume with absolute path
cfg = struct();
cfg.environment = 'video';
cfg.plume_metadata = '/home/snb6/palmer_scratch/plume/smoke_1a_orig_backgroundsubtracted_meta.yaml';
cfg.plotting = 0;
cfg.ntrials = 2;
cfg.triallength = 1000;  % shorter for testing

fprintf('Testing navigation model with HDF5 plume from palmer_scratch...\n');
fprintf('  environment: %s\n', cfg.environment);
fprintf('  plume_metadata: %s\n', cfg.plume_metadata);
fprintf('  ntrials: %d\n', cfg.ntrials);
fprintf('  triallength: %d\n', cfg.triallength);

try
    % Run the navigation model
    result = run_navigation_cfg(cfg);
    fprintf('\nSimulation completed successfully!\n');
    fprintf('Result fields: %s\n', strjoin(fieldnames(result), ', '));
    
    % Check the output
    if isfield(result, 'x') && isfield(result, 'y')
        fprintf('Generated %d timesteps for %d trials\n', size(result.x, 1), size(result.x, 2));
    end
    
catch ME
    fprintf('\nError in simulation: %s\n', ME.message);
    fprintf('Error stack:\n');
    for i = 1:min(3, length(ME.stack))
        fprintf('  %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
end

exit;
