
function tests = test_load_config
    tests = functiontests(localfunctions);
end

function setupOnce(~)
    addpath(fullfile(pwd, 'Code'));
end

function testLoadSampleConfig(testCase)
    cfg = load_config(fullfile('tests','sample_config.yaml'));
    verifyEqual(testCase, cfg.environment, "gaussian");
    verifyEqual(testCase, cfg.triallength, 100);
    verifyEqual(testCase, cfg.plotting, 0);
    verifyEqual(testCase, cfg.ntrials, 5);

end

function testLoadPaperConfig(testCase)
    cfg = load_config(fullfile('configs','alvarez_salvado_2018.yaml'));
    verifyEqual(testCase, cfg.beta, 0.01, 'AbsTol', 1e-8);
    verifyEqual(testCase, cfg.tau_Aon, 490);
    verifyEqual(testCase, cfg.kdown, 0.5);
end

function testBooleanParsing(testCase)
    cfg = load_config(fullfile('tests','sample_config_bilateral.yaml'));
    verifyTrue(testCase, cfg.bilateral);
end
