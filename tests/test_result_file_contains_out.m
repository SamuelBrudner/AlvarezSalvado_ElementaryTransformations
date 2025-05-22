function tests = test_result_file_contains_out
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    addpath(fullfile(pwd, 'Code'));
    stubDir = tempname;
    mkdir(stubDir);
    testCase.TestData.stubDir = stubDir;
    fid = fopen(fullfile(stubDir, 'load_experiment_config.m'), 'w');
    fprintf(fid, ['function cfg = load_experiment_config(~)\n', ...
        'cfg.experiment.jobs_per_condition = 1;\n', ...
        'cfg.experiment.num_plumes = 1;\n', ...
        'cfg.experiment.num_sensing = 1;\n', ...
        'cfg.experiment.plume_types = {''gaussian''};\n', ...
        'cfg.experiment.sensing_modes = {''bilateral''};\n', ...
        'cfg.experiment.output_base = fullfile(tempdir, ''result_out_test'');\n', ...
        'cfg.plume_config = fullfile(''tests'', ''sample_config.yaml'');\n', ...
        'cfg.get_output_dir = @(plume, sensing, agent_id, seed) ...\n', ...
        '    fullfile(cfg.experiment.output_base, sprintf(''%s_%s'', plume, sensing), sprintf(''%d_%d'', agent_id, seed));\n', ...
        'end']);
    fclose(fid);
    addpath(stubDir);
    cfg = load_experiment_config();
    testCase.TestData.cfg = cfg;
end

function teardownOnce(testCase)
    rmpath(testCase.TestData.stubDir);
    if exist(testCase.TestData.cfg.experiment.output_base, 'dir')
        rmdir(testCase.TestData.cfg.experiment.output_base, 's');
    end
    rmdir(testCase.TestData.stubDir, 's');
end

function testResultFileHasOut(testCase)
    cfg = testCase.TestData.cfg;
    outDirBase = cfg.experiment.output_base;
    if exist(outDirBase, 'dir')
        rmdir(outDirBase, 's');
    end
    run_agent_simulation(1, 1, 'dummy.yaml');
    resultFile = fullfile(outDirBase, 'gaussian_bilateral', '1_1', 'result.mat');
    info = whos('-file', resultFile);
    verifyTrue(testCase, any(strcmp({info.name}, 'out')), 'Variable ''out'' missing from result.mat');
end
