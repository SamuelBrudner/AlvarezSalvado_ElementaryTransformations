function tests = test_run_agent_simulation_skip_existing
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    addpath(fullfile(pwd, 'Code'));
    tmpDir = fullfile(tempname);
    mkdir(tmpDir);
    cfgPath = fullfile(tmpDir, 'experiment_config.yaml');
    fid = fopen(cfgPath, 'w');
    fprintf(fid, 'experiment:\n');
    fprintf(fid, '  output_base: %s\n', tmpDir);
    fprintf(fid, '  agents_per_condition: 1\n');
    fprintf(fid, '  agents_per_job: 1\n');
    fprintf(fid, '  plume_types: [dummy]\n');
    fprintf(fid, '  sensing_modes: [bilateral]\n');
    fprintf(fid, 'plume_config: tests/sample_config.yaml\n');
    fprintf(fid, 'matlab: {}\n');
    fprintf(fid, 'slurm: {}\n');
    fclose(fid);
    testCase.TestData.tmpDir = tmpDir;
    testCase.TestData.cfgFile = cfgPath;
end

function teardownOnce(testCase)
    rmdir(testCase.TestData.tmpDir, 's');
end

function testSkipExistingResult(testCase)
    job_id = 1;
    agent_id = 1;
    run_agent_simulation(job_id, agent_id, testCase.TestData.cfgFile);
    outputDir = fullfile(testCase.TestData.tmpDir, 'dummy_bilateral', '1_1');
    resultFile = fullfile(outputDir, 'result.mat');
    info1 = dir(resultFile);
    run_agent_simulation(job_id, agent_id, testCase.TestData.cfgFile);
    info2 = dir(resultFile);
    verifyEqual(testCase, info1.datenum, info2.datenum);
end
