addpath('Code');

fprintf('Testing comparison_config.yaml...\n');
try
    cfg = load_config('configs/comparison_config.yaml');
    
    % Check if triallength exists and its type
    if isfield(cfg, 'triallength')
        fprintf('triallength: %g (type: %s)\n', cfg.triallength, class(cfg.triallength));
        
        % Test if it works in array creation
        x = zeros(cfg.triallength, 1);
        fprintf('Success! Can create array with triallength=%d\n', cfg.triallength);
    else
        fprintf('triallength not found in config\n');
    end
    
    % Show other fields
    fprintf('\nOther config fields:\n');
    if isfield(cfg, 'environment')
        fprintf('environment: %s\n', cfg.environment);
    end
    if isfield(cfg, 'ntrials')
        fprintf('ntrials: %d\n', cfg.ntrials);
    end
    
catch ME
    fprintf('Error loading config: %s\n', ME.message);
    if contains(ME.message, 'Size inputs must be numeric')
        fprintf('\nThis suggests triallength is still a string with comments\n');
    end
end

exit;
