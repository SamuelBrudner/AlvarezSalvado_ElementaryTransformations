classdef test_load_config < matlab.unittest.TestCase
    methods(Test)
        function testSampleConfig(testCase)
            cfg = load_config('tests/sample_config.json');
            testCase.verifyEqual(cfg.environment, 'gaussian');
            testCase.verifyEqual(cfg.triallength, 100);
            testCase.verifyEqual(cfg.plotting, 0);
            testCase.verifyEqual(cfg.ntrials, 5);
        end
    end
end
