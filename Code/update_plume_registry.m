function update_plume_registry(file, minVal, maxVal, yamlPath)
%UPDATE_PLUME_REGISTRY Insert or update intensity range for a plume file.
%   update_plume_registry(FILE, MINVAL, MAXVAL) records the intensity range
%   for FILE inside the plume registry. If an entry already exists, the
%   stored range is expanded to include the new MINVAL and MAXVAL.
%   update_plume_registry(..., YAMLPATH) writes to the specified registry
%   file. The default registry lives under configs/plume_registry.yaml at
%   the repository root.
%
%   Example:
%       update_plume_registry('plume.h5', 0, 1.2);
%
%   See also: load_yaml

arguments
    file (1,:) char
    minVal (1,1) double
    maxVal (1,1) double
    yamlPath (1,:) char = defaultRegistryPath()
end

% Load existing registry or start new
if exist(yamlPath, 'file') == 2
    try
        registry = load_yaml(yamlPath);
    catch
        registry = struct();
    end
else
    registry = struct();
end

% Update or insert the entry
if isfield(registry, file)
    entry = registry.(file);
    if isfield(entry, 'min')
        minVal = min(minVal, double(entry.min));
    end
    if isfield(entry, 'max')
        maxVal = max(maxVal, double(entry.max));
    end
end
registry.(file) = struct('min', double(minVal), 'max', double(maxVal));

% Ensure destination directory exists
[yDir, ~] = fileparts(yamlPath);
if ~isempty(yDir) && ~exist(yDir, 'dir')
    mkdir(yDir);
end

% Save registry back to disk
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

function p = defaultRegistryPath
thisDir = fileparts(mfilename('fullpath'));
rootDir = fileparts(thisDir);
p = fullfile(rootDir, 'configs', 'plume_registry.yaml');
end
