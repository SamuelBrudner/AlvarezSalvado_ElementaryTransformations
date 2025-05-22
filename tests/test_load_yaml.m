function tests = test_load_yaml
    tests = functiontests(localfunctions);
end

function setupOnce(~)
    addpath(fullfile(pwd, 'Code'));
end

function testLoadYaml(testCase)
    cfg = load_yaml(fullfile('tests', 'sample_config.yaml'));
    verifyEqual(testCase, cfg.environment, 'gaussian');
    verifyEqual(testCase, cfg.triallength, 100);
end
