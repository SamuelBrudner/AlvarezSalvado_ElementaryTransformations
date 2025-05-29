function stats = plume_intensity_stats
%PLUME_INTENSITY_STATS Return stored plume intensity statistics.
%   STATS = PLUME_INTENSITY_STATS() reads the YAML file 'plume_intensity_stats.yaml'
%   located in the configs folder and returns a struct with fields 'SMOKE' and
%   'CRIM'.
%
%   Example
%       stats = plume_intensity_stats;
%       disp(stats.CRIM.min)

rootDir = fileparts(fileparts(mfilename('fullpath')));
yamlPath = fullfile(rootDir, 'configs', 'plume_intensity_stats.yaml');
stats = load_yaml(yamlPath);
end

