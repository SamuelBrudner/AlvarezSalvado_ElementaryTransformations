function [plume_file, plume_config] = get_plume_file()
%GET_PLUME_FILE Return plume HDF5 filename from stored paths configuration

% Load stored paths
try
    paths = load_paths();
    config_path = paths.plume_config;
    plume_file = '';
catch
    % Fallback to environment variables
    plume_file = getenv('MATLAB_PLUME_FILE');
    config_path = getenv('PLUME_CONFIG');
    
    if isempty(plume_file) || isempty(config_path)
        error('No paths configuration found. Run setup_env_paths.sh first!');
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
        cfg = jsondecode(fileread(config_path));
        
        % Update config from file
        if isfield(cfg, 'data_path') && isfield(cfg.data_path, 'path')
            plume_file = cfg.data_path.path;
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
        
        if isfield(cfg, 'temporal')
            if isfield(cfg.temporal, 'frame_rate')
                plume_config.frame_rate = cfg.temporal.frame_rate;
                plume_config.time_scale_50hz = cfg.temporal.frame_rate / 50.0;
                plume_config.time_scale_15hz = cfg.temporal.frame_rate / 15.0;
            end
        end
        
        if isfield(cfg, 'data_path') && isfield(cfg.data_path, 'dataset_name')
            plume_config.dataset_name = cfg.data_path.dataset_name;
        end
        
        if isfield(cfg, 'simulation')
            if isfield(cfg.simulation, 'duration_seconds')
                plume_config.simulation.duration_seconds = cfg.simulation.duration_seconds;
            end
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
