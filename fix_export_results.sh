#!/bin/bash
# Fix export_results.m to handle nested result structure

cp Code/export_results.m Code/export_results.m.bak

# Add handling for nested result structure
sed -i '/result = load(input_file);/,/^if ~isfield(result, '\''x'\'')/ {
    s/result = load(input_file);/data = load(input_file);/
    /data = load(input_file);/a\
% Handle nested result structure\
if isfield(data, '\''result'\'')\
    result = data.result;\
elseif isfield(data, '\''out'\'')\
    result = data.out;\
else\
    result = data;\
end
}' Code/export_results.m

echo "Fixed export_results.m to handle nested structures"
