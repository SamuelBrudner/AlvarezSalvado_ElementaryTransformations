% startup.m - Auto-load paths configuration
fprintf('Loading AlvarezSalvado environment paths...\n');

% First, add Code directory to path using the current directory
startup_dir = fileparts(mfilename('fullpath'));
if isempty(startup_dir)
    startup_dir = pwd;
end

% Add Code directory to path
code_dir = fullfile(startup_dir, 'Code');
if exist(code_dir, 'dir')
    addpath(genpath(code_dir));
    fprintf('Added Code directory to path: %s\n', code_dir);
else
    warning('Code directory not found: %s', code_dir);
end

% Now try to load paths
try
    % Load the stored paths
    paths = load_paths();
    
    % Change to project root
    cd(paths.project_root);
    
    fprintf('Environment ready:\n');
    fprintf('  Working directory: %s\n', pwd);
    fprintf('  Plume file: %s\n', paths.plume_file);
catch ME
    % If load_paths fails, we're still OK - Code is in the path
    fprintf('Note: Could not load paths config (%s)\n', ME.message);
    fprintf('Working directory: %s\n', pwd);
end
