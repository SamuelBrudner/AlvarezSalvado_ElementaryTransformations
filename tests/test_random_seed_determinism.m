function tests = test_random_seed_determinism
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    addpath(fullfile(pwd, 'Code'));
end

function testSameSeedReproducible(testCase)
    cfg = load_config(fullfile('tests','sample_config.yaml'));
    cfg.randomSeed = 123;
    out1 = run_navigation_cfg(cfg);
    out2 = run_navigation_cfg(cfg);
    verifyEqual(testCase, out1.x, out2.x);
    verifyEqual(testCase, out1.y, out2.y);
end
