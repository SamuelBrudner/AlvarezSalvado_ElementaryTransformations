function paths = load_paths()
%LOAD_PATHS Load environment paths from configuration
%   This function loads the paths configuration created by setup_env_paths.sh
%   ensuring consistent path usage across all MATLAB scripts.

% Get the directory where this function lives
this_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(this_dir);

% Load paths config
paths_config_file = fullfile(project_root, 'configs', 'paths.json');

if ~exist(paths_config_file, 'file')
    error('Paths configuration not found. Run setup_env_paths.sh first!');
end

% Read and parse the JSON
try
    json_text = fileread(paths_config_file);
    paths = jsondecode(json_text);
catch ME
    error('Failed to load paths config: %s', ME.message);
end

% Set environment variables for compatibility
setenv('MATLAB_PROJECT_ROOT', paths.project_root);
setenv('MATLAB_PLUME_FILE', paths.plume_file);
setenv('PLUME_CONFIG', paths.plume_config);

% Add Code directory to path if not already there
if ~contains(path, paths.code_dir)
    addpath(genpath(paths.code_dir));
end

% Display loaded paths (optional)
if nargout == 0
    fprintf('Loaded paths from: %s\n', paths_config_file);
    fprintf('  Project root: %s\n', paths.project_root);
    fprintf('  Plume file: %s\n', paths.plume_file);
end

end
