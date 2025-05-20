function tests = test_run_navigation_cfg
    tests = functiontests(localfunctions);
end

function setupOnce(~)
    addpath(fullfile(pwd,'Code'));
end

function testRunUnilateral(testCase)
    cfg = load_config(fullfile('tests','sample_config.json'));
    result = run_navigation_cfg(cfg);
    verifyEqual(testCase, isfield(result,'x'), true);
end

function testRunBilateral(testCase)
    cfg = load_config(fullfile('tests','sample_config_bilateral.json'));
    result = run_navigation_cfg(cfg);
    verifyEqual(testCase, isfield(result,'x'), true);
end
