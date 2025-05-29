function update_plume_registry(path, min_val, max_val, yamlPath)
%UPDATE_PLUME_REGISTRY Update intensity range registry.
%   UPDATE_PLUME_REGISTRY(PATH, MIN_VAL, MAX_VAL) stores the provided
%   intensity range for PATH in the plume registry YAML file. If an
%   existing entry is found, the stored range is expanded to encompass
%   the new values. The default registry file is
%   'configs/plume_registry.yaml' relative to the repository root.
%
%   Example:
%       update_plume_registry('plume.h5', 0, 1.2);
%
%   See also: load_yaml

arguments
    path (1,:) char
    min_val (1,1) double
    max_val (1,1) double
    yamlPath (1,:) char = defaultYaml()
end

if exist(yamlPath, 'file')
    registry = load_yaml(yamlPath);
else
    registry = struct();
end

if isfield(registry, path)
    entry = registry.(path);
    min_val = min(double(entry.min), min_val);
    max_val = max(double(entry.max), max_val);
end

registry.(path) = struct('min', double(min_val), 'max', double(max_val));

try
    if exist('yamlwrite', 'file') == 2
        yamlwrite(yamlPath, registry);
    else
        fid = fopen(yamlPath, 'w');
        fwrite(fid, jsonencode(registry));
        fclose(fid);
    end
catch ME
    warning('update_plume_registry:WriteFailed', ...
        'Failed to save registry: %s', ME.message);
end
end

function p = defaultYaml
thisDir = fileparts(mfilename('fullpath'));
rootDir = fileparts(thisDir);
p = fullfile(rootDir, 'configs', 'plume_registry.yaml');
end
