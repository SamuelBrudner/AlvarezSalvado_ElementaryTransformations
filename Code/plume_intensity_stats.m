function stats = plume_intensity_stats(yamlPath)
%PLUME_INTENSITY_STATS Return stored plume intensity statistics.
%   STATS = PLUME_INTENSITY_STATS(YAMLPATH) loads plume intensity statistics
%   from the given YAML file. If omitted, the default configuration file
%   'configs/plume_intensity_stats.yaml' relative to the repository root is
%   used.

if nargin < 1 || isempty(yamlPath)
    thisDir = fileparts(mfilename('fullpath'));
    rootDir = fileparts(thisDir);
    yamlPath = fullfile(rootDir, 'configs', 'plume_intensity_stats.yaml');
end

stats = load_yaml(yamlPath);
end
