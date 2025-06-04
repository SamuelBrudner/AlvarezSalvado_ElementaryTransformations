function [plume_file, plume_config] = get_plume_file_wrapper()
%GET_PLUME_FILE_WRAPPER Wrapper that preserves symlink paths on HPC

% First get the normal result
[plume_file, plume_config] = get_plume_file();

% Check for environment override first
env_override = getenv('MATLAB_PLUME_FILE');
if ~isempty(env_override)
    plume_file = env_override;
    fprintf('Using environment override: %s\n', plume_file);
    return;
end

% Fix common HPC path resolutions
path_mappings = {
    '/vast/palmer/home.grace/snb6/', '/home/snb6/';
    '/gpfs/loomis/home.grace/snb6/', '/home/snb6/';
    '/vast/palmer/scratch/snb6/', '/home/snb6/scratch/';
};

original_path = plume_file;
for i = 1:size(path_mappings, 1)
    if contains(plume_file, path_mappings{i, 1})
        plume_file = strrep(plume_file, path_mappings{i, 1}, path_mappings{i, 2});
        fprintf('Path resolution fix: %s -> %s\n', original_path, plume_file);
        break;
    end
end

end
