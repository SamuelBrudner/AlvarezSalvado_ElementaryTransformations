function tests = test_run_batch_job_error_logging
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    addpath(fullfile(pwd, 'Code'));
    stubDir = tempname;
    mkdir(stubDir);

    %% load_experiment_config stub
    fid = fopen(fullfile(stubDir, 'load_experiment_config.m'), 'w');
    fprintf(fid, ['function cfg = load_experiment_config(~)\n', ...
        'cfg.experiment.plume_types = {''p1''};\n', ...
        'cfg.experiment.sensing_modes = {''s1''};\n', ...
        'cfg.experiment.jobs_per_condition = 1;\n', ...
        'cfg.experiment.num_plumes = 1;\n', ...
        'cfg.experiment.num_sensing = 1;\n', ...
        'cfg.experiment.output_base = ''%s'';\n', ...
        'cfg.get_output_dir = @(~,~,a,~) fullfile(''%s'', sprintf(''agent%d'', a));\n', ...
        'cfg.plume_config = fullfile(''tests'', ''sample_config.yaml'');\n', ...
        'end'], stubDir, stubDir);
    fclose(fid);

    %% load_config stub
    fid = fopen(fullfile(stubDir, 'load_config.m'), 'w');
    fprintf(fid, ['function cfg = load_config(~)\n', ...
        'cfg.bilateral = false;\n', ...
        'end']);
    fclose(fid);

    %% run_navigation_cfg stub
    fid = fopen(fullfile(stubDir, 'run_navigation_cfg.m'), 'w');
    fprintf(fid, ['function out = run_navigation_cfg(cfg)\n', ...
        'if contains(cfg.outputDir, ''agent1'')\n', ...
        '    error(''SimFail:Test'', ''intentional error'');\n', ...
        'end\n', ...
        'out.result = true;\n', ...
        'end']);
    fclose(fid);

    addpath(stubDir);
    testCase.TestData.stubDir = stubDir;
    cfgPath = fullfile(stubDir, 'dummy.yaml');
    fid = fopen(cfgPath, 'w'); fclose(fid);
    testCase.TestData.cfgPath = cfgPath;
end

function teardownOnce(testCase)
    rmpath(testCase.TestData.stubDir);
    rmdir(testCase.TestData.stubDir, 's');
end

function testLogsCreatedAndBatchContinues(testCase)
    out = evalc('run_batch_job(testCase.TestData.cfgPath, 1, 1, 2, false)');
    errFile = fullfile(testCase.TestData.stubDir, 'agent1', 'error.log');
    resultFile = fullfile(testCase.TestData.stubDir, 'agent2', 'result.mat');
    verifyTrue(testCase, isfile(errFile), 'Error log should be created');
    verifyTrue(testCase, isfile(resultFile), 'Second agent should finish');
    verifyNotEmpty(testCase, regexp(out, 'Error', 'once'), 'Batch summary missing');
end
