function tests = test_rescale_plume_range_validation
    tests = functiontests(localfunctions);
end

function setupOnce(~)
    addpath(fullfile(pwd,'Code'));
end

function testNonNumericData(testCase)
    f = @() rescale_plume_range('bad',0,1);
    try
        f();
        verifyFail(testCase,'Expected error');
    catch ME
        verifyTrue(testCase, startsWith(ME.identifier,'MATLAB:'), ...
            'Unexpected error identifier');
    end
end

function testNonScalarTarget(testCase)
    f = @() rescale_plume_range([1 2],[0 1],1);
    try
        f();
        verifyFail(testCase,'Expected error');
    catch ME
        verifyTrue(testCase, startsWith(ME.identifier,'MATLAB:'), ...
            'Unexpected error identifier');
    end
end
