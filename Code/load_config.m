function config = load_config(config_path, default_config)
% LOAD_CONFIG Load and validate a YAML configuration file
%   config = LOAD_CONFIG(config_path) loads the YAML file at config_path
%   config = LOAD_CONFIG(config_path, default_config) uses default_config for missing values
%
%   Throws an error if the config file is invalid or missing required fields

    % Check if YAML toolbox is available
    if ~exist('yaml.ReadYaml', 'file')
        error('YAML toolbox not found. Please run startup.m first');
    end

    % Load the YAML file
    try
        config = yaml.ReadYaml(config_path);
    catch ME
        error('Failed to load config file %s: %s', config_path, ME.message);
    end
    
    % Apply defaults if provided
    if nargin > 1 && isstruct(default_config)
        config = merge_structs(default_config, config);
    end
    
    % Validate required fields
    required_fields = {'experiment', 'plume_types', 'sensing_modes', 'agents_per_condition'};
    for i = 1:length(required_fields)
        if ~isfield(config, required_fields{i})
            error('Missing required field in config: %s', required_fields{i});
        end
    end
end

function out = merge_structs(defaults, overrides)
% MERGE_STRUCTS Recursively merge two structs
    out = defaults;
    if isstruct(overrides)
        fields = fieldnames(overrides);
        for i = 1:length(fields)
            field = fields{i};
            if isfield(defaults, field) && isstruct(defaults.(field)) && isstruct(overrides.(field))
                out.(field) = merge_structs(defaults.(field), overrides.(field));
            else
                out.(field) = overrides.(field);
            end
        end
    end
end
