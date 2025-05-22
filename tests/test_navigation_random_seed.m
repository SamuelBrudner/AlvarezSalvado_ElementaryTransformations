function tests = test_navigation_random_seed
    tests = functiontests(localfunctions);
end

function setupOnce(~)
    addpath(fullfile(pwd,'Code'));
end

function testSeedReproducibility(testCase)
    cfg = load_config(fullfile('tests','sample_config.yaml'));
    cfg.randomSeed = 123;
    out1 = run_navigation_cfg(cfg);
    out2 = run_navigation_cfg(cfg);
    verifyEqual(testCase, out1.x, out2.x);
    verifyEqual(testCase, out1.y, out2.y);
    verifyEqual(testCase, out1.theta, out2.theta);
end

function testDifferentSeeds(testCase)
    cfg = load_config(fullfile('tests','sample_config.yaml'));
    cfg.randomSeed = 1;
    out1 = run_navigation_cfg(cfg);
    cfg.randomSeed = 2;
    out2 = run_navigation_cfg(cfg);
    % At least one field should differ
    testCase.assertFalse(isequal(out1.x, out2.x) && isequal(out1.y, out2.y) && isequal(out1.theta, out2.theta));
end
