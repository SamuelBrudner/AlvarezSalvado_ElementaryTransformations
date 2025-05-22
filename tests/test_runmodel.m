function tests = test_runmodel
    tests = functiontests(localfunctions);
end

function setupOnce(~)
    addpath(fullfile(pwd, 'Code'));
end

function testRunmodelExecutes(testCase)
    tl = 50;
    env = 'gaussian';
    nt = 1;
    [out, matrix] = runmodel(tl, env, nt, 1, 1);
    verifyEqual(testCase, size(matrix), [1 1]);
    verifyTrue(testCase, isfield(out, 'successrate'));
end
