function s = load_yaml(path)
%LOAD_YAML Load a YAML file using the YAML toolbox.
%   S = LOAD_YAML(PATH) reads the YAML file specified by PATH and returns
%   a struct with the decoded contents. The YAML toolbox is added to the
%   path automatically if a 'yaml' folder exists next to this function.

% Add YAML toolbox path if needed
thisDir = fileparts(mfilename('fullpath'));
rootDir = fileparts(thisDir);
yamlDir = fullfile(rootDir, 'yaml');
if exist(yamlDir, 'dir') && isempty(which('yaml.ReadYaml'))
    addpath(yamlDir);
end

s = yaml.ReadYaml(path);
end
