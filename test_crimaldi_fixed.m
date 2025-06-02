% Quick test of Crimaldi with fixed path
addpath('Code');
try
    fprintf('Testing Crimaldi plume with fixed path...\n');
    out = navigation_model_vec(3600, 'Crimaldi', 0, 1);
    fprintf('SUCCESS! Success rate: %.2f\n', out.successrate);
    save('crimaldi_test_success.mat', 'out');
catch ME
    fprintf('ERROR: %s\n', ME.message);
    fprintf('In: %s at line %d\n', ME.stack(1).name, ME.stack(1).line);
end
exit;
