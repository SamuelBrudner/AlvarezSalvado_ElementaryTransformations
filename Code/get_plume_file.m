function plume_file = get_plume_file()
%GET_PLUME_FILE Return plume HDF5 filename from configuration
%   Uses environment variable PLUME_CONFIG to locate the configuration.

config_path = getenv('PLUME_CONFIG');
if isempty(config_path)
    config_path = fullfile(fileparts(mfilename('fullpath')),'..','configs', ...
        'navigation_model','navigation_model_default.json');
end
fprintf('Loading plume config from %s\n', config_path);
if ~exist(config_path, 'file')
    warning('Config file %s not found. Using default.', config_path);
    plume_file = '10302017_10cms_bounded.hdf5';
    return;
end

try
    cfg = jsondecode(fileread(config_path));
    if isfield(cfg,'plume_file')
        plume_file = cfg.plume_file;
        if isfield(cfg,'plume_path') && ~isempty(cfg.plume_path)
            plume_file = fullfile(cfg.plume_path, plume_file);
        end
    else
        warning('Invalid config file %s. Using default.', config_path);
        plume_file = '10302017_10cms_bounded.hdf5';
    end
catch err
    warning('Failed to read config %s: %s. Using default.', config_path, err.message);
    plume_file = '10302017_10cms_bounded.hdf5';
end
end
