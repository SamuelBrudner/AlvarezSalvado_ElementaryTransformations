% Check structure of result.mat
result_file = 'data/raw/test_batch_v3/crimaldi/agent_0001/result.mat';
fprintf('Loading: %s\n', result_file);

% Load and show variables
data = load(result_file);
fprintf('\nVariables in file:\n');
whos -file result_file

fprintf('\nChecking for trajectory data:\n');
if isfield(data, 'x')
    fprintf('  ✓ Found x directly\n');
elseif isfield(data, 'out') && isfield(data.out, 'x')
    fprintf('  ✓ Found x in out structure\n');
elseif isfield(data, 'R') && isfield(data.R, 'x')
    fprintf('  ✓ Found x in R structure\n');
else
    fprintf('  ✗ No x trajectory found\n');
    fprintf('\nFile structure:\n');
    disp(data);
end

exit
