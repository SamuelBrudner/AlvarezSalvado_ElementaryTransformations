function [plume_file, plume_config] = get_plume_file()
%GET_PLUME_FILE Return plume HDF5 filename from configuration
%   STRICT VERSION - Fails if config not found

config_path = getenv('PLUME_CONFIG');
if isempty(config_path)
    % Default to plumes directory
    config_path = fullfile(fileparts(mfilename('fullpath')), '..', 'configs', 'plumes', 'crimaldi_10cms_bounded.json');
end

fprintf('Loading plume config from %s\n', config_path);

% FAIL if config doesn't exist
if ~exist(config_path, 'file')
    error('PLUME_CONFIG_ERROR: Config file not found: %s', config_path);
end

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

try
    cfg = jsondecode(fileread(config_path));
catch err
    error('PLUME_CONFIG_ERROR: Failed to parse JSON: %s', err.message);
end

% Extract plume file path
if isfield(cfg,'plume_file')
    % Old format
    plume_file = cfg.plume_file;
    if isfield(cfg,'plume_path') && ~isempty(cfg.plume_path)
        plume_file = fullfile(cfg.plume_path, plume_file);
    end
elseif isfield(cfg, 'data_path') && isfield(cfg.data_path, 'path')
    % New format
    plume_file = cfg.data_path.path;
    
    % Update config struct
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
    error('PLUME_CONFIG_ERROR: Invalid config format - missing plume_file or data_path.path');
end

% Handle relative paths
if ~isempty(plume_file) && plume_file(1) ~= '/'
    project_root = fileparts(fileparts(mfilename('fullpath')));
    plume_file = fullfile(project_root, plume_file);
end

% Return just filename if only one output requested
if nargout < 2
    clear plume_config;
end

end
