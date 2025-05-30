function tests = test_isSlurmCluster
    tests = functiontests(localfunctions);
end

function setupOnce(~)
    addpath(fullfile(pwd, 'Code'));
end

function teardownEach(~)
    unsetenv('SLURM_JOB_ID');
end

function testReturnsFalseWhenUnset(testCase)
    unsetenv('SLURM_JOB_ID');
    verifyFalse(testCase, isSlurmCluster());
end

function testReturnsTrueWhenSet(testCase)
    setenv('SLURM_JOB_ID', '123');
    verifyTrue(testCase, isSlurmCluster());
end
