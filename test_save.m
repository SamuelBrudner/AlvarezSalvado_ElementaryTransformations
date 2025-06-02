% Test the save functionality
outputDir = 'test_output/test_save';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Create a test structure
result.x = rand(100,1);
result.y = rand(100,1);
result.successrate = 0.5;

% Try both save methods
fprintf('Saving with -struct option...\n');
save(fullfile(outputDir, 'result_struct.mat'), '-struct', 'result');

fprintf('Saving without -struct option...\n');
save(fullfile(outputDir, 'result_normal.mat'), 'result', '-v7');

fprintf('Done! Check the output directory.\n');
