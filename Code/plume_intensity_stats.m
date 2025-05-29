function stats = plume_intensity_stats(yamlPath)
%PLUME_INTENSITY_STATS Return stored plume intensity statistics.
%   STATS = PLUME_INTENSITY_STATS(YAMLPATH) loads plume intensity statistics
%   from the given YAML file. If omitted, the default configuration file
%   'configs/plume_intensity_stats.yaml' relative to the repository root is
%   used.
%
%   Example:
%       stats = plume_intensity_stats;
%       stats2 = plume_intensity_stats('my_stats.yaml');

arguments
    yamlPath (1,:) char = defaultStatsYaml()
end

stats = load_yaml(yamlPath);
end

function p = defaultStatsYaml
thisDir = fileparts(mfilename('fullpath'));
rootDir = fileparts(thisDir);
p = fullfile(rootDir, 'configs', 'plume_intensity_stats.yaml');
end
