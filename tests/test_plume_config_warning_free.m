function tests = test_plume_config_warning_free
    tests = functiontests(localfunctions);
end

function setupOnce(~)
    addpath(fullfile(pwd, 'Code'));
end

function testNoWarningForPlumeConfig(testCase)
    cfgPath = fullfile('configs', 'my_complex_plume_config.yaml');
    f = @() load_config(cfgPath);
    verifyWarningFree(testCase, f);
end
