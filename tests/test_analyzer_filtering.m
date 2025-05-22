function tests = test_analyzer_filtering
    tests = functiontests(localfunctions);
end

function setupOnce(~)
    startup; % add paths via startup script
end

function testFilteringAvailable(testCase)
    data(1).x = zeros(5,1);
    data(1).y = zeros(5,1);
    data(1).theta = zeros(5,1);
    res = analyzer(data);
    verifyTrue(testCase, isfield(res, 'xfilt'));
    verifyTrue(testCase, isfield(res, 'yfilt'));
    verifyTrue(testCase, isfield(res, 'thetafilt'));
end
