function tests = test_config_file_written
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    addpath(fullfile(pwd, 'Code'));
    testDir = tempname;
    mkdir(testDir);
    testCase.TestData.outputDir = testDir;
end

function teardownOnce(testCase)
    rmdir(testCase.TestData.outputDir, 's');
end

function testConfigIsSaved(testCase)
    cfg.environment = 'gaussian';
    cfg.triallength = 10;
    cfg.plotting = 0;
    cfg.ntrials = 1;
    cfg.outputDir = testCase.TestData.outputDir;
    run_navigation_cfg(cfg);
    filePath = fullfile(cfg.outputDir, 'config_used.yaml');
    verifyEqual(testCase, exist(filePath, 'file'), 2);
end
