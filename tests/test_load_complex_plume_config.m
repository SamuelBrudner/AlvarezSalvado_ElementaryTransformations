function tests = test_load_complex_plume_config
    tests = functiontests(localfunctions);
end

function setupOnce(~)
    addpath(fullfile(pwd, 'Code'));
end

function testLoadComplexConfig(testCase)
    cfg = load_config(fullfile('configs', 'my_complex_plume_config.yaml'));
    verifyEqual(testCase, cfg.environment, 'video');
    verifyEqual(testCase, cfg.plume_video, 'data/smoke_1a_orig_backgroundsubtracted.avi');
    verifyEqual(testCase, cfg.px_per_mm, 6.536, 'AbsTol', 1e-8);
    verifyEqual(testCase, cfg.frame_rate, 60);
    verifyEqual(testCase, cfg.plotting, 0);
    verifyEqual(testCase, cfg.ntrials, 10);
    verifyEqual(testCase, cfg.ws, 1);
end
