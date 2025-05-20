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
        continue;
    end
    tokens = regexp(line, '^([^:]+):\s*(.*)$', 'tokens', 'once');
    if isempty(tokens)
        continue;
    end
    key = strtrim(tokens{1});
    valStr = strtrim(tokens{2});
    numVal = str2double(valStr);
    if ~isnan(numVal)
        cfg.(key) = numVal;
    else
        if (startsWith(valStr, '"') && endsWith(valStr, '"')) || ...
           (startsWith(valStr, '''') && endsWith(valStr, ''''))
            valStr = valStr(2:end-1);
        end
        cfg.(key) = valStr;
    end
end
end
