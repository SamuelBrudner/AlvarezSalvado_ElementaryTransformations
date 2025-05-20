function cfg = load_config(path)
%LOAD_CONFIG Load simulation parameters from a YAML file.
%   CFG = LOAD_CONFIG(PATH) reads the YAML file specified by PATH and
%   returns a struct with the decoded parameters.

fid = fopen(path, 'r');
if fid == -1
    error('Could not open configuration file: %s', path);
end
lines = textscan(fid, '%s', 'Delimiter', '\n');
fclose(fid);

cfg = struct();
for i = 1:numel(lines{1})
    line = strtrim(lines{1}{i});
    if isempty(line) || startsWith(line, '#')
        continue
    end
    tokens = strsplit(line, ':', 2);
    if numel(tokens) < 2
        continue
    end
    key = strtrim(tokens{1});
    value = strtrim(tokens{2});
    numval = str2double(value);
    if ~isnan(numval)
        cfg.(key) = numval;
    else
        cfg.(key) = value;
    end
end
end
