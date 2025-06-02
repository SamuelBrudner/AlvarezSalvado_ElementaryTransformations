error_file = 'data/raw/test_batch_v3/crimaldi/error_agent_0001.mat';
load(error_file);

if exist('error_info', 'var')
    disp('=== Error Information ===');
    disp(error_info);
    
    % Display all fields
    fields = fieldnames(error_info);
    for i = 1:length(fields)
        fprintf('\n--- %s ---\n', fields{i});
        disp(error_info.(fields{i}));
    end
end
exit
