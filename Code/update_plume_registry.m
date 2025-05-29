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

function update_plume_registry(file, minVal, maxVal, yamlPath)
%UPDATE_PLUME_REGISTRY Insert or update plume intensity range.
%   UPDATE_PLUME_REGISTRY(FILE, MINVAL, MAXVAL) updates the plume registry
%   entry for FILE in the default registry YAML file. If FILE already
%   exists, the stored range is expanded to include MINVAL and MAXVAL.
%   UPDATE_PLUME_REGISTRY(..., YAMLPATH) specifies a custom registry file.
%
%   The registry is stored in YAML when the YAML toolbox is available.
%   Otherwise, JSON encoding is used as a fallback.

arguments
    file (1,:) char
    minVal (1,1) double
    maxVal (1,1) double
    yamlPath (1,:) char = defaultRegistryPath()
end

% Load existing registry if possible
registry = struct();
if exist(yamlPath, 'file') == 2
    try
        if exist('load_yaml', 'file') == 2
            registry = load_yaml(yamlPath);
        else
            fid = fopen(yamlPath, 'r');
            if fid ~= -1
                txt = fread(fid, '*char')';
                fclose(fid);
                registry = jsondecode(txt);
            end
        end
    catch ME %#ok<NASGU>
        registry = struct();
    end
end

if ~isstruct(registry)
    registry = struct();
end

% Update or insert entry
if isfield(registry, file)
    entry = registry.(file);
    if isfield(entry, 'min')
        minVal = min(minVal, double(entry.min));
    end
    if isfield(entry, 'max')
        maxVal = max(maxVal, double(entry.max));
    end
end
registry.(file) = struct('min', minVal, 'max', maxVal);

% Ensure directory exists
[yDir,~] = fileparts(yamlPath);
if ~isempty(yDir) && ~exist(yDir, 'dir')
    mkdir(yDir);
end

% Save registry
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
