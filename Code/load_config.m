function config = load_config(config_path, default_config)
% LOAD_CONFIG Load and validate simulation configuration
%   CONFIG = LOAD_CONFIG(CONFIG_PATH) loads configuration from YAML or key-value text file
%   CONFIG = LOAD_CONFIG(CONFIG_PATH, DEFAULT_CONFIG) merges with default values
%
%   Supports both YAML (if yaml toolbox is available) and simple key:value format
%   Automatically detects file format based on extension or content

    % Check if file exists and is readable
    if ~exist(config_path, 'file')
        error('Configuration file not found: %s', config_path);
    end
    
    % Try to load as YAML first if available
    [~, ~, ext] = fileparts(config_path);
    use_yaml = strcmpi(ext, '.yaml') || strcmpi(ext, '.yml');
    
    if use_yaml && exist('yaml.ReadYaml', 'file')
        try
            config = yaml.ReadYaml(config_path);
        catch ME
            warning('Failed to load as YAML, falling back to simple parser: %s', ME.message);
            config = load_simple_config(config_path);
        end
    else
        config = load_simple_config(config_path);
    end
    
    % Apply defaults if provided
    if nargin > 1 && isstruct(default_config)
        config = merge_structs(default_config, config);
    end
    
    % Validate required fields
    validate_config(config);
    
    % Convert numeric strings to numbers where appropriate
    config = convert_numeric_fields(config);
end

function config = load_simple_config(config_path)
% LOAD_SIMPLE_CONFIG Load simple key:value config file
    fid = fopen(config_path, 'r');
    if fid == -1
        error('Could not open configuration file: %s', config_path);
    end
    
    lines = textscan(fid, '%s', 'Delimiter', '\n', 'Whitespace', '');
    fclose(fid);
    lines = lines{1};
    
    config = struct();
    for i = 1:numel(lines)
        line = strtrim(lines{i});
        % Skip comments and empty lines
        if isempty(line) || line(1) == '#'
            continue;
        end
        
        % Split on first colon
        colon_pos = strfind(line, ':');
        if isempty(colon_pos)
            continue;
        end
        
        key = strtrim(line(1:colon_pos(1)-1));
        value = strtrim(line(colon_pos(1)+1:end));
        
        % Convert value to appropriate type
        if isempty(value)
            config.(key) = '';
        elseif strcmpi(value, 'true')
            config.(key) = true;
        elseif strcmpi(value, 'false')
            config.(key) = false;
        else
            % Try to convert to number
            num = str2double(value);
            if ~isnan(num) && ~isnan(str2double(value(1)))
                config.(key) = num;
            else
                config.(key) = value;
            end
        end
    end
end

function validate_config(config)
% VALIDATE_CONFIG Check for recommended experiment fields
%   Only warn about missing fields if any of them are present.
    recommended_fields = {'experiment_name', 'plume_type', 'sensing_mode'};

    has_any = any(isfield(config, recommended_fields));
    if has_any
        for i = 1:numel(recommended_fields)
            if ~isfield(config, recommended_fields{i})
                warning('Missing recommended field in config: %s', recommended_fields{i});
            end
        end
    end
end

function config = convert_numeric_fields(config)
% CONVERT_NUMERIC_FIELDS Convert string numbers to numeric
    fields = fieldnames(config);
    for i = 1:length(fields)
        if ischar(config.(fields{i})) || isstring(config.(fields{i}))
            % Handle cases where YAML values have inline comments
            % e.g., "3600  # Match Crimaldi duration"
            str_val = char(config.(fields{i}));
            
            % Extract just the numeric part before any comment
            numeric_part = regexp(str_val, '^\s*(\d+\.?\d*)\s*', 'tokens', 'once');
            if ~isempty(numeric_part)
                num = str2double(numeric_part{1});
            else
                num = str2double(str_val);
            end
            
            if ~isnan(num)
                config.(fields{i}) = num;
            end
        end
    end
end

function out = merge_structs(defaults, overrides)
% MERGE_STRUCTS Recursively merge two structs
    out = defaults;
    if ~isstruct(overrides)
        return;
    end
    
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
