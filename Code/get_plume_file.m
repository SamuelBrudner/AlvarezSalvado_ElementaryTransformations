function [plume_file, plume_config] = get_plume_file()
%GET_PLUME_FILE Return plume HDF5 filename from stored paths configuration

% First priority: explicit environment override
config_path = getenv('PLUME_CONFIG');
plume_file  = getenv('MATLAB_PLUME_FILE');

% If PLUME_CONFIG is not provided or invalid, fall back to paths.json via load_paths
if isempty(config_path) || ~exist(config_path,'file')
    try
        paths = load_paths();
        config_path = paths.plume_config;
        if isempty(plume_file)
            plume_file = paths.plume_file;
        end
    catch
        if isempty(config_path)
            error('Could not determine plume config (PLUME_CONFIG not set and load_paths failed).');
        end
    end
end

% Initialize default config
plume_config = struct();
plume_config.mm_per_pixel = 0.74;
plume_config.pixel_scale = 0.74;
plume_config.frame_rate = 15;
plume_config.time_scale_50hz = 15/50;
plume_config.time_scale_15hz = 1.0;
plume_config.plume_xlims = [1, 216];
plume_config.plume_ylims = [1, 406];
plume_config.dataset_name = '/dataset2';

% Load config file
if exist(config_path, 'file')
    try
        raw_json = fileread(config_path);
        disp('DEBUG: Config path:');
        disp(config_path);
        disp('DEBUG: Raw JSON:');
        disp(raw_json);
        try
            cfg = jsondecode(raw_json);
        catch err
            warning('Could not parse config file: %s', err.message);
            disp('DEBUG: Failed to parse config, config_path:');
            disp(config_path);
            rethrow(err);
        end
        disp('DEBUG: Loaded config struct:');
        disp(cfg);
        disp('DEBUG: Top-level fields:');
        disp(fieldnames(cfg));
        
        % Update config from file
        % Preserve the full data_path struct for downstream access (e.g. navigation_model_vec expects
        % plume_config.data_path.path to exist).
        if isfield(cfg, 'data_path')
            plume_config.data_path = cfg.data_path;
            if isfield(cfg.data_path, 'path')
                plume_file = cfg.data_path.path;
            end
        end
        if isfield(cfg, 'spatial')
            if isfield(cfg.spatial, 'mm_per_pixel')
                plume_config.mm_per_pixel = cfg.spatial.mm_per_pixel;
                plume_config.pixel_scale = cfg.spatial.mm_per_pixel;
            end
            if isfield(cfg.spatial, 'resolution')
                plume_config.plume_xlims = [1, cfg.spatial.resolution.width];
                plume_config.plume_ylims = [1, cfg.spatial.resolution.height];
            end
        end
        
        % Copy temporal field from config to plume_config
        if isfield(cfg, 'temporal')
            plume_config.temporal = cfg.temporal;
            disp('DEBUG: plume_config.temporal assigned!');
            disp(plume_config.temporal);
            if isfield(cfg.temporal, 'frame_rate')
                plume_config.frame_rate = cfg.temporal.frame_rate;
                plume_config.time_scale_50hz = cfg.temporal.frame_rate / 50.0;
                plume_config.time_scale_15hz = cfg.temporal.frame_rate / 15.0;
            end
        end
        
        % Copy spatial field from config to plume_config
        if isfield(cfg, 'spatial')
            plume_config.spatial = cfg.spatial;
            disp('DEBUG: plume_config.spatial assigned!');
            disp(plume_config.spatial);
        end
        
        if isfield(cfg, 'data_path') && isfield(cfg.data_path, 'dataset_name')
            plume_config.dataset_name = cfg.data_path.dataset_name;
        end
        
        if isfield(cfg, 'simulation')
            % Preserve full simulation sub-struct (includes agent_initialization)
            plume_config.simulation = cfg.simulation;
        end
        
    catch err
        warning('Could not parse config file: %s', err.message);
    end
else
    warning('Config file not found: %s', config_path);
end

% Fallback: if plume_file is still empty, try to use paths.plume_file
if isempty(plume_file) && exist('paths', 'var') && isfield(paths, 'plume_file')
    plume_file = paths.plume_file;
end

fprintf('Using plume file: %s\n', plume_file);

if nargout < 2
    clear plume_config;
end

end
