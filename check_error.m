error_file = 'data/raw/test_batch_v3/crimaldi/error_agent_0001.mat';
try
    load(error_file);
    whos
    if exist('ME', 'var')
        disp('Error message:');
        disp(ME.message);
        disp('Error identifier:');
        disp(ME.identifier);
        if ~isempty(ME.stack)
            disp('Stack trace:');
            for i = 1:length(ME.stack)
                fprintf('  In %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
            end
        end
    end
catch err
    disp('Could not load error file');
    disp(err.message);
end
exit
