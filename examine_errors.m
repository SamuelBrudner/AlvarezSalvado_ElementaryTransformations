% Examine the error files from failed Crimaldi runs
error_files = dir('data/raw/test_batch_v3/crimaldi/error_agent_*.mat');

for i = 1:min(3, length(error_files))  % Look at first 3 errors
    fprintf('\n=== Error file %d: %s ===\n', i, error_files(i).name);
    error_data = load(fullfile(error_files(i).folder, error_files(i).name));
    
    % Display the error message
    if isfield(error_data, 'ME')
        fprintf('Error message: %s\n', error_data.ME.message);
        fprintf('Error identifier: %s\n', error_data.ME.identifier);
        
        % Show stack trace (first few entries)
        if isfield(error_data.ME, 'stack') && ~isempty(error_data.ME.stack)
            fprintf('Stack trace:\n');
            for j = 1:min(3, length(error_data.ME.stack))
                fprintf('  %s (line %d)\n', error_data.ME.stack(j).name, error_data.ME.stack(j).line);
            end
        end
    else
        % Display all fields in the error file
        fields = fieldnames(error_data);
        for j = 1:length(fields)
            fprintf('Field %s:\n', fields{j});
            disp(error_data.(fields{j}));
        end
    end
end
