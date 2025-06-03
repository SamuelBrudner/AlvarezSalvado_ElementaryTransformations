function [plume_file, plume_config] = get_plume_file_hpc()
% GET_PLUME_FILE_HPC - HPC-aware wrapper for get_plume_file
%   Handles /vast/palmer/... to /home/snb6/... conversion

% First get the normal result
[plume_file, plume_config] = get_plume_file();

% Fix for HPC paths
if contains(plume_file, '/vast/palmer/home.grace/snb6/')
    plume_file = strrep(plume_file, '/vast/palmer/home.grace/snb6/', '/home/snb6/');
    fprintf('HPC path conversion: Using %s\n', plume_file);
end

% Also check environment override
env_override = getenv('MATLAB_PLUME_FILE');
if ~isempty(env_override) && exist(env_override, 'file')
    plume_file = env_override;
    fprintf('Using environment override: %s\n', plume_file);
end

end
