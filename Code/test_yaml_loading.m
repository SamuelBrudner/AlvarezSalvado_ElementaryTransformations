%% Test YAML Configuration Loading
% This script tests the YAML configuration loading functionality

% Add necessary paths
addpath(genpath('..'));  % Add project root to path

% Initialize environment
startup();

% Test loading the default config
try
    fprintf('Loading default configuration...\n');
    config = load_config(fullfile('..', 'configs', 'default_config.yaml'));
    disp('Configuration loaded successfully:');
    disp(config);
    
    % Test saving a modified config
    test_config = config;
    test_config.experiment.name = 'test_experiment';
    
    test_config_path = fullfile('..', 'configs', 'test_config.yaml');
    yaml.WriteYaml(test_config_path, test_config);
    fprintf('Test configuration saved to: %s\n', test_config_path);
    
    % Test loading the modified config
    loaded_config = load_config(test_config_path);
    assert(strcmp(loaded_config.experiment.name, 'test_experiment'), 'Config loading test failed');
    fprintf('✅ All tests passed!\n');
    
catch ME
    fprintf('❌ Test failed: %s\n', ME.message);
    rethrow(ME);
end
