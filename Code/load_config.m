function cfg = load_config(path)
%LOAD_CONFIG Load simulation parameters from a YAML file.
%   CFG = LOAD_CONFIG(PATH) reads the YAML file specified by PATH and
%   returns a struct with the decoded parameters. The parser supports simple
%   key:value pairs where values are numeric or strings.


fid = fopen(path, 'r');
if fid == -1
    error('Could not open configuration file: %s', path);
end
lines = textscan(fid, '%s', 'Delimiter', '\n', 'Whitespace', '');
fclose(fid);
lines = lines{1};

cfg = struct();
for i = 1:numel(lines)
    line = strtrim(lines{i});
    if isempty(line) || startsWith(line, '#')
        continue;
    end
    tokens = split(line, ':');
    if numel(tokens) < 2
        continue;
    end
    key = strtrim(tokens{1});
    value = strtrim(strjoin(tokens(2:end), ':'));
    num = str2double(value);
    if ~isnan(num)
        cfg.(key) = num;
    else
        cfg.(key) = value;
    end
end
end
