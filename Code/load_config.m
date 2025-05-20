function cfg = load_config(path)
%LOAD_CONFIG Load simulation parameters from a JSON file.
%   CFG = LOAD_CONFIG(PATH) reads the JSON file specified by PATH and
%   returns a struct with the decoded parameters.

fid = fopen(path, 'r');
if fid == -1
    error('Could not open configuration file: %s', path);
end
raw = fread(fid, '*char')';
fclose(fid);

cfg = jsondecode(raw);
end
