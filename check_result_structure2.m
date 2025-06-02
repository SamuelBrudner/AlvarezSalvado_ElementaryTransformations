% Check structure of result.mat
result_file = 'data/raw/test_batch_v3/crimaldi/agent_0001/result.mat';
fprintf('Loading: %s\n', result_file);

% Load the file
data = load(result_file);
fprintf('\nVariables loaded:\n');
disp(fieldnames(data));

% Check what's in the 'result' variable
if isfield(data, 'result')
    fprintf('\nFound "result" variable. Its fields are:\n');
    disp(fieldnames(data.result));
    
    fprintf('\nChecking for trajectories:\n');
    if isfield(data.result, 'x')
        fprintf('  ✓ Found x: size %s\n', mat2str(size(data.result.x)));
    end
    if isfield(data.result, 'y')
        fprintf('  ✓ Found y: size %s\n', mat2str(size(data.result.y)));
    end
    if isfield(data.result, 'theta')
        fprintf('  ✓ Found theta: size %s\n', mat2str(size(data.result.theta)));
    end
end

exit
