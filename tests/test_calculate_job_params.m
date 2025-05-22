function tests = test_calculate_job_params
    tests = functiontests(localfunctions);
end

function setupOnce(~)
    addpath(fullfile(pwd, 'Code'));
end

function testBasicDistribution(testCase)
    numConditions = 2;
    agentsPerCondition = 5;
    agentsPerJob = 2;

    expected = [ ...
        1 1 2; ...
        2 1 2; ...
        1 3 4; ...
        2 3 4; ...
        1 5 5; ...
        2 5 5];

    totalJobs = calculateTotalJobs(numConditions, agentsPerCondition, agentsPerJob);
    verifyEqual(testCase, totalJobs, size(expected,1));

    for taskId = 1:totalJobs
        params = calculateJobParams(taskId, numConditions, agentsPerCondition, agentsPerJob);
        verifyEqual(testCase, [params.conditionIndex params.startAgent params.endAgent], expected(taskId,:));
    end
end
