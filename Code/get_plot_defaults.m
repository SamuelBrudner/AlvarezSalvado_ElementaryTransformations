function cfg = get_plot_defaults()
%GET_PLOT_DEFAULTS Loads and caches plot style defaults from JSON config.
%   cfg = get_plot_defaults() returns a struct with color, line width, and
%   scale bar length settings for arena plotting, loaded from
%   configs/plot_defaults.json. Uses persistent caching for efficiency.
%
%   To enable verbose debug logging, set debug_mode = true in the debug_log function

persistent cached_cfg cached_time

% Define a function to log diagnostics only in debug mode
function debug_log(varargin)
    % Only print diagnostic messages if in debug mode
    % To enable debug mode, set this to true
    debug_mode = false;
    
    if debug_mode
        fprintf(varargin{:});
    end
end

% Use absolute path by first getting the project root directory
% Either from current directory or relative to this script's location
try
    % Try multiple methods to find project root
    project_root = '';
    
    % Method 1: Look for the paths.json file which should be in the config directory
    debug_log('[get_plot_defaults] Looking for project root using paths.json...\n');
    
    % Try to load the paths.json which should have project_root defined
    try
        % First check in standard locations
        potential_paths = {
            fullfile(pwd, 'configs', 'paths.json'), 
            fullfile(fileparts(pwd), 'configs', 'paths.json'),
            fullfile(pwd, '..', 'configs', 'paths.json')
        };
        
        for i = 1:length(potential_paths)
            if exist(potential_paths{i}, 'file')
                debug_log('[get_plot_defaults] Found paths.json at: %s\n', potential_paths{i});
                fid = fopen(potential_paths{i}, 'r');
                if fid > 0
                    raw_json = fread(fid, inf, 'uint8=>char')';
                    fclose(fid);
                    paths_config = jsondecode(raw_json);
                    if isfield(paths_config, 'project_root') && ~isempty(paths_config.project_root)
                        project_root = paths_config.project_root;
                        debug_log('[get_plot_defaults] Found project_root from paths.json: %s\n', project_root);
                        break;
                    end
                end
            end
        end
    catch
        debug_log('[get_plot_defaults] Error reading paths.json, trying other methods\n');
    end
    
    % Method 2: If paths.json didn't work, use directory structure
    if isempty(project_root)
        % First try to find project root from current directory
        current_dir = pwd;
        debug_log('[get_plot_defaults] Starting with working directory: %s\n', current_dir);
        
        if contains(current_dir, 'AlvarezSalvado_ElementaryTransformations')
            % Navigate up to project root if in a subdirectory
            debug_log('[get_plot_defaults] In project directory, finding root...\n');
            while ~contains(fileparts(current_dir), 'AlvarezSalvado_ElementaryTransformations') && ~isempty(fileparts(current_dir))
                current_dir = fileparts(current_dir);
                debug_log('[get_plot_defaults] Moving up to: %s\n', current_dir);
            end
            project_root = current_dir;
        else
            % If not in project directory, use the location of this script
            debug_log('[get_plot_defaults] Not in project directory, using script location...\n');
            script_path = mfilename('fullpath');
            debug_log('[get_plot_defaults] Script path: %s\n', script_path);
            script_dir = fileparts(script_path);
            debug_log('[get_plot_defaults] Script directory: %s\n', script_dir);
            project_root = fileparts(script_dir); % Assuming Code dir is directly under project root
        end
    end
    
    debug_log('[get_plot_defaults] Project root identified as: %s\n', project_root);
    
    % Check if we found a valid project root
    if isempty(project_root) || ~exist(project_root, 'dir')
        warning('[get_plot_defaults] Valid project root not found, using default values');
        cfg = get_default_values();
        return;
    end
    
    if ~exist(fullfile(project_root, 'configs'), 'dir')
        warning('[get_plot_defaults] Configs directory not found in project root');
        % List contents of the supposed project root to help debugging
        dir_contents = dir(project_root);
        debug_log('[get_plot_defaults] Project root contents:\n');
        for i = 1:numel(dir_contents)
            if dir_contents(i).isdir
                item_type = 'dir';
            else
                item_type = 'file';
            end
            debug_log('  - %s (%s)\n', dir_contents(i).name, item_type);
        end
    end
    
    % Try multiple locations for plot_defaults.json
    potential_config_paths = {
        fullfile(project_root, 'configs', 'plot_defaults.json'),  % Standard location
        fullfile(pwd, 'configs', 'plot_defaults.json'),            % Current directory
        fullfile(fileparts(pwd), 'configs', 'plot_defaults.json'), % Parent of current directory
        fullfile(pwd, '..', 'configs', 'plot_defaults.json'),      % Another way to check parent
    };
    
    config_path = '';
    for i = 1:length(potential_config_paths)
        if exist(potential_config_paths{i}, 'file')
            config_path = potential_config_paths{i};
            debug_log('[get_plot_defaults] Found plot defaults at: %s\n', config_path);
            break;
        end
    end
    
    % Fall back to default values if we can't find the file
    if isempty(config_path) || ~exist(config_path, 'file')
        warning('[get_plot_defaults] Plot defaults file not found in any standard location. Using default values.');
        cfg = get_default_values();
        return;
    end
    
    % Only reload if file changed or not loaded yet
    if isempty(cached_cfg) || isempty(cached_time) || ...
            dir(config_path).datenum > cached_time
        
        try
            % Try to open the file with proper error checking
            fid = fopen(config_path, 'r');
            if fid < 0
                warning('[get_plot_defaults] Could not open plot defaults file: %s\nUsing default values.', config_path);
                cfg = get_default_values();
                return;
            end
            
            % Read the file contents
            raw = fread(fid, inf, 'uint8=>char')';
            fclose(fid);
            
            % Parse JSON with error checking
            if isempty(raw)
                warning('[get_plot_defaults] Empty plot defaults file: %s\nUsing default values.', config_path);
                cfg = get_default_values();
                return;
            end
            
            json_cfg = jsondecode(raw);
            
            % Validate essential fields
            if ~isstruct(json_cfg) || ~isfield(json_cfg, 'colors') || ~isfield(json_cfg, 'line_widths')
                warning('[get_plot_defaults] Invalid plot defaults structure in %s\nUsing default values.', config_path);
                cfg = get_default_values();
                return;
            end
            
            % Update cache
            cached_cfg = json_cfg;
            cached_time = dir(config_path).datenum;
            
            % Log success
            debug_log('[get_plot_defaults] Successfully loaded plot defaults from %s\n', config_path);
        catch err
            % Handle any errors during file reading or JSON parsing
            if exist('fid', 'var') && fid > 0
                fclose(fid); % Make sure to close the file handle even on error
            end
            warning('[get_plot_defaults] Error reading plot defaults: %s\nUsing default values.', err.message);
            cfg = get_default_values();
            return;
        end
    else
        debug_log('[get_plot_defaults] Using cached plot defaults\n');
    end
    
    cfg = cached_cfg;
    
catch err
    warning('Unexpected error in get_plot_defaults: %s\nUsing default values.', err.message);
    cfg = get_default_values();
end

end

function cfg = get_default_values()
% Provides hardcoded default values when config file can't be read
cfg = struct();
cfg.colors = struct();
cfg.colors.arena_edge = [0, 0, 0];
cfg.colors.init_zone_edge = [1, 1, 0];
cfg.colors.success_circle = [0, 0.7, 0];
cfg.colors.success_star = [0, 0.7, 0];
cfg.colors.scale_bar = [0, 0, 0];
cfg.colors.source = [1, 0, 0];

cfg.line_widths = struct();
cfg.line_widths.arena_edge = 2;
cfg.line_widths.init_zone_edge = 3;
cfg.line_widths.success_circle = 1.5;
cfg.line_widths.success_star = 1.5;
cfg.line_widths.source = 0;
cfg.line_widths.scale_bar = 3;

cfg.scale_bar_length_cm = 5;
cfg.markers = struct();
cfg.markers.source = 'o';
end
