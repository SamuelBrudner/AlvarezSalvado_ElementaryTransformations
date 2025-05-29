function tests = test_plume_intensity_stats_validation
    tests = functiontests(localfunctions);
end

function setupOnce(~)
    addpath(fullfile(pwd,'Code'));
end

function testInvalidPathType(testCase)
    f = @() plume_intensity_stats(123);
    try
        f();
        verifyFail(testCase,'Expected error');
    catch ME
        verifyTrue(testCase, startsWith(ME.identifier,'MATLAB:'), ...
            'Unexpected error identifier');
    end
end
