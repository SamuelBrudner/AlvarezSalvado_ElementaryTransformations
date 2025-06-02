addpath('Code');

% Load the actual config file that exists
fprintf('Testing load_config with my_complex_plume_config.yaml...\n');
cfg = load_config('configs/my_complex_plume_config.yaml');

% Check the types and values
fprintf('\nChecking data types:\n');
fields_to_check = {'environment', 'plotting', 'ntrials', 'px_per_mm', 'frame_rate'};
for i = 1:length(fields_to_check)
    field = fields_to_check{i};
    if isfield(cfg, field)
        if isnumeric(cfg.(field))
            fprintf('%s: %s (value: %g)\n', field, class(cfg.(field)), cfg.(field));
        else
            fprintf('%s: %s (value: %s)\n', field, class(cfg.(field)), cfg.(field));
        end
    end
end

% Check if triallength exists
if isfield(cfg, 'triallength')
    fprintf('triallength: %s (value: %g)\n', class(cfg.triallength), cfg.triallength);
    try
        x = zeros(cfg.triallength, 1);
        fprintf('\nSuccess! triallength works correctly\n');
    catch ME
        fprintf('\nError with triallength: %s\n', ME.message);
    end
else
    fprintf('\ntriallength not set in config (commented out)\n');
end

exit;
