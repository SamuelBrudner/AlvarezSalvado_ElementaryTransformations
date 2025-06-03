function [plume_file, plume_config] = get_plume_file()
%GET_PLUME_FILE Return plume HDF5 filename from configuration
%   Uses environment variable PLUME_CONFIG to locate the configuration.
%
%   plume_file = get_plume_file()
%       Returns just the plume file path (backward compatible)
%
%   [plume_file, plume_config] = get_plume_file()  
%       Also returns plume configuration with mm/pixel, frame rate, etc.

% Initialize default config values
plume_config = struct();
plume_config.mm_per_pixel = 0.74;
plume_config.pixel_scale = 0.74;
plume_config.frame_rate = 15;
plume_config.time_scale_50hz = 15/50;
plume_config.time_scale_15hz = 1.0;
plume_config.plume_xlims = [1, 216];
plume_config.plume_ylims = [1, 406];
plume_config.dataset_name = '/dataset2';

config_path = getenv('PLUME_CONFIG');
if isempty(config_path)
    config_path = fullfile(fileparts(mfilename('fullpath')),'..','configs', ...
        'navigation_model','navigation_model_default.json');
end

% Also check for new config location
if ~exist(config_path, 'file')
    alt_path = fullfile(fileparts(mfilename('fullpath')),'..','configs', ...
        'plumes','crimaldi_10cms_bounded.json');
    if exist(alt_path, 'file')
        config_path = alt_path;
    end
end

fprintf('Loading plume config from %s\n', config_path);
if ~exist(config_path, 'file')
    warning('Config file %s not found. Using default.', config_path);
    plume_file = '10302017_10cms_bounded.hdf5';
    if nargout < 2
        clear plume_config;
    end
    return;
end

try
    cfg = jsondecode(fileread(config_path));
    
    % Handle old format
    if isfield(cfg,'plume_file')
        plume_file = cfg.plume_file;
        if isfield(cfg,'plume_path') && ~isempty(cfg.plume_path)
            plume_file = fullfile(cfg.plume_path, plume_file);
        end
    % Handle new format
    elseif isfield(cfg, 'data_path') && isfield(cfg.data_path, 'path')
        plume_file = cfg.data_path.path;
        
        % Update config struct with new format data
        if isfield(cfg, 'spatial')
            plume_config.mm_per_pixel = cfg.spatial.mm_per_pixel;
            plume_config.pixel_scale = cfg.spatial.mm_per_pixel;
            if isfield(cfg.spatial, 'resolution')
                plume_config.plume_xlims = [1, cfg.spatial.resolution.width];
                plume_config.plume_ylims = [1, cfg.spatial.resolution.height];
            end
        end
        
        if isfield(cfg, 'temporal')
            plume_config.frame_rate = cfg.temporal.frame_rate;
            plume_config.time_scale_50hz = cfg.temporal.frame_rate / 50.0;
            plume_config.time_scale_15hz = cfg.temporal.frame_rate / 15.0;
        end
        
        if isfield(cfg.data_path, 'dataset_name')
            plume_config.dataset_name = cfg.data_path.dataset_name;
        end
    else
        warning('Invalid config file %s. Using default.', config_path);
        plume_file = '10302017_10cms_bounded.hdf5';
    end
    
    % Handle relative paths
    if exist('plume_file', 'var') && ~isempty(plume_file) && plume_file(1) ~= '/'
        % Make relative to project root
        project_root = fileparts(fileparts(mfilename('fullpath')));
        plume_file = fullfile(project_root, plume_file);
    end
    
catch err
    warning('Failed to read config %s: %s. Using default.', config_path, err.message);
    plume_file = '10302017_10cms_bounded.hdf5';
end

% Return just filename if only one output requested (backward compatibility)
if nargout < 2
    clear plume_config;
end
end
