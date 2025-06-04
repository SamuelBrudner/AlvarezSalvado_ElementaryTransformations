function [plume_file, plume_config] = get_plume_file()
%GET_PLUME_FILE Return plume HDF5 filename from configuration

% Simple approach - use environment or defaults
config_path = getenv('PLUME_CONFIG');
if isempty(config_path)
    % Use explicit path for HPC
    if exist('/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations', 'dir')
        config_path = '/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations/configs/plumes/crimaldi_10cms_bounded.json';
    else
        % Fallback to relative path
        config_path = 'configs/plumes/crimaldi_10cms_bounded.json';
    end
end

fprintf('Loading plume config from %s\n', config_path);

% Initialize defaults
plume_config = struct();
plume_config.mm_per_pixel = 0.74;
plume_config.pixel_scale = 0.74;
plume_config.frame_rate = 15;
plume_config.time_scale_50hz = 15/50;
plume_config.time_scale_15hz = 1.0;
plume_config.plume_xlims = [1, 216];
plume_config.plume_ylims = [1, 406];
plume_config.dataset_name = '/dataset2';

% Load config if it exists
if exist(config_path, 'file')
    try
        cfg = jsondecode(fileread(config_path));
        
        % Extract plume file path
        if isfield(cfg, 'data_path') && isfield(cfg.data_path, 'path')
            plume_file = cfg.data_path.path;
        elseif isfield(cfg, 'plume_file')
            plume_file = cfg.plume_file;
        else
            plume_file = 'data/plumes/10302017_10cms_bounded.hdf5';
        end
        
        % Update config parameters
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
        plume_file = 'data/plumes/10302017_10cms_bounded.hdf5';
    end
else
    warning('Config file not found, using defaults');
    plume_file = 'data/plumes/10302017_10cms_bounded.hdf5';
end

% Make path absolute if needed
if exist(plume_file, 'file')
    % File exists as is
elseif plume_file(1) ~= '/'
    % Try with explicit base path
    if exist('/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations', 'dir')
        plume_file = fullfile('/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations', plume_file);
    end
end

% Environment override
env_plume = getenv('MATLAB_PLUME_FILE');
if ~isempty(env_plume)
    plume_file = env_plume;
end

if nargout < 2
    clear plume_config;
end

end
