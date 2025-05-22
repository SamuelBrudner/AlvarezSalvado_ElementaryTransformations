function tests = test_agents_per_job_validation
    tests = functiontests(localfunctions);
end

function setupOnce(~)
    addpath(fullfile(pwd, 'Code'));
end

function testInvalidExperimentValue(testCase)
    cfgPath = fullfile('configs','batch_job_config.yaml');
    f = @() load_experiment_config(cfgPath, 'agents_per_job', 0);
    verifyError(testCase, f, 'common:InvalidAgentsPerJob');
end

function testInvalidConfigValue(testCase)
    tmp = [tempname '.yaml'];
    fid = fopen(tmp,'w');
    fprintf(fid,'agents_per_job: -1\n');
    fclose(fid);
    cleanup = onCleanup(@() delete(tmp));
    f = @() load_config(tmp);
    verifyError(testCase, f, 'common:InvalidAgentsPerJob');
end
