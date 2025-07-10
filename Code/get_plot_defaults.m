function cfg = get_plot_defaults()
%GET_PLOT_DEFAULTS Loads and caches plot style defaults from JSON config.
%   cfg = get_plot_defaults() returns a struct with color, line width, and
%   scale bar length settings for arena plotting, loaded from
%   configs/plot_defaults.json. Uses persistent caching for efficiency.

persistent cached_cfg cached_time
config_path = fullfile('configs', 'plot_defaults.json');

% Only reload if file changed or not loaded yet
if isempty(cached_cfg) || isempty(cached_time) || ...
        dir(config_path).datenum > cached_time
    fid = fopen(config_path, 'r');
    raw = fread(fid, inf, 'uint8=>char')';
    fclose(fid);
    json_cfg = jsondecode(raw);
    cached_cfg = json_cfg;
    cached_time = dir(config_path).datenum;
end
cfg = cached_cfg;
end
