% Test export_results with a sample file
addpath('Code');

input_file = 'data/raw/test_batch_v3/crimaldi/agent_0001/result.mat';
output_dir = 'test_export';

fprintf('Testing export_results on: %s\n', input_file);

try
    export_results(input_file, output_dir);
    fprintf('✓ Export successful!\n');
    
    % Check what was created
    fprintf('\nCreated files:\n');
    dir(fullfile(output_dir, '*.csv'));
    dir(fullfile(output_dir, '*.json'));
    
    % Clean up
    rmdir(output_dir, 's');
catch ME
    fprintf('✗ Export failed: %s\n', ME.message);
end

exit
