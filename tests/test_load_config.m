
function tests = test_load_config
    tests = functiontests(localfunctions);
end

function setupOnce(~)
    % Add the repository Code directory to the MATLAB path regardless of
    % the current working directory when the tests are invoked.
    testDir = fileparts(mfilename('fullpath'));
    repoRoot = fileparts(testDir);
    addpath(fullfile(repoRoot, 'Code'));

end

function testLoadSampleConfig(testCase)
    cfg = load_config(fullfile('tests','sample_config.json'));
    verifyEqual(testCase, cfg.environment, "gaussian");
    verifyEqual(testCase, cfg.triallength, 100);
    verifyEqual(testCase, cfg.plotting, 0);
    verifyEqual(testCase, cfg.ntrials, 5);

end
