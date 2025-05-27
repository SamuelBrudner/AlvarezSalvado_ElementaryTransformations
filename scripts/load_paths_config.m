function config = load_paths_config()
% LOAD_PATHS_CONFIG Load the project paths configuration
%   Loads the paths.yaml configuration file, expanding any environment variables.
%   Returns a struct with the configuration values.

    % Get the directory of the currently executing script
    scriptDir = fileparts(mfilename('fullpath'));
    projectRoot = fullfile(scriptDir, '..');
    
    % Path to the configuration file
    configFile = fullfile(projectRoot, 'configs', 'paths.yaml');
    
    % Check if the file exists
    if ~exist(configFile, 'file')
        error('Configuration file not found: %s\nRun setup_local_config.py first.', configFile);
    end
    
    % Read the YAML file
    config = yaml.loadFile(configFile, 'ConvertToArray', true);
    
    % Expand environment variables in paths
    config.project_root = expand_path(config.project_root);
    
    % Expand script paths
    scriptFields = fieldnames(config.scripts);
    for i = 1:numel(scriptFields)
        config.scripts.(scriptFields{i}) = expand_path(config.scripts.(scriptFields{i}));
    end
    
    % Expand data paths
    dataFields = fieldnames(config.data);
    for i = 1:numel(dataFields)
        config.data.(dataFields{i}) = expand_path(config.data.(dataFields{i}));
    end
    
    % Expand config paths
    configFields = fieldnames(config.configs);
    for i = 1:numel(configFields)
        config.configs.(configFields{i}) = expand_path(config.configs.(configFields{i}));
    end
    
    % Expand output paths
    outputFields = fieldnames(config.output);
    for i = 1:numel(outputFields)
        config.output.(outputFields{i}) = expand_path(config.output.(outputFields{i}));
    end
end

function path = expand_path(path)
% EXPAND_PATH Expand environment variables in a path
%   Replaces ${VAR} with the value of environment variable VAR
    if ischar(path)
        % Handle environment variables like ${VAR}
        tokens = regexp(path, '\${([^}]+)}', 'tokens');
        for i = 1:length(tokens)
            varName = tokens{i}{1};
            if strcmp(varName, 'PROJECT_DIR')
                % Special case for project directory
                scriptDir = fileparts(mfilename('fullpath'));
                value = fullfile(scriptDir, '..');
            else
                value = getenv(varName);
                if isempty(value)
                    warning('Environment variable %s not set', varName);
                    value = '';
                end
            end
            path = strrep(path, ['${' varName '}'], value);
        end
        
        % Convert to absolute path if it's a relative path
        if ~startsWith(path, '/') && ~contains(path, ':')
            path = fullfile(pwd, path);
        end
        
        % Normalize path separators
        path = strrep(path, '\', '/');
    end
end
