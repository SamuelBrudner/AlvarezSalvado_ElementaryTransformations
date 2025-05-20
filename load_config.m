function cfg = load_config(path)
%LOAD_CONFIG Load simulation parameters from a JSON file
%   CFG = LOAD_CONFIG(PATH) reads the JSON file specified by PATH and
%   returns a struct with the parameters.

jsonText = fileread(path);
cfg = jsondecode(jsonText);
end
