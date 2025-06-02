% Check error files
error_file = 'data/raw/test_batch/crimaldi/error_agent_0001.mat';
if exist(error_file, 'file')
    load(error_file);
    fprintf('Error for agent %d: %s\n', error_info.agent, error_info.error);
end
exit;
