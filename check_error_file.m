load('data/raw/test_batch_v3/crimaldi/error_agent_0001.mat');
whos
if exist('error_info', 'var')
    disp(error_info);
elseif exist('ME', 'var')
    disp(ME);
else
    disp('Unknown variables in error file');
end
exit
