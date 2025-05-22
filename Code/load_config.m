function cfg = load_config(path)
%LOAD_CONFIG Load simulation parameters from a YAML file.
%   CFG = LOAD_CONFIG(PATH) reads the YAML file specified by PATH and returns
%   a struct with the decoded parameters using the YAML toolbox.

% Add YAML toolbox path if available via load_yaml
cfg = load_yaml(path);
if ~isstruct(cfg)
    cfg = struct(cfg);
end
end
