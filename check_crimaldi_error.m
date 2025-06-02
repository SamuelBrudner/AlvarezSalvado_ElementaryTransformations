% Check what's in the Crimaldi error files
error_file = 'data/raw/test_batch_v3/crimaldi/error_agent_0001.mat';
fprintf('Loading error file: %s\n', error_file);

error_data = load(error_file);
if isfield(error_data, 'error_info')
    fprintf('\nError: %s\n', error_data.error_info.error);
    fprintf('Agent: %d\n', error_data.error_info.agent);
    
    % Show stack trace
    if isfield(error_data.error_info, 'stack')
        fprintf('\nStack trace:\n');
        for i = 1:min(5, length(error_data.error_info.stack))
            fprintf('  %s (line %d)\n', ...
                    error_data.error_info.stack(i).name, ...
                    error_data.error_info.stack(i).line);
        end
    end
else
    % Display all fields
    disp(error_data);
end
