function tests = test_run_agent_simulation_seed
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    addpath(fullfile(pwd,'Code'));
    stubDir = fullfile(tempname);
    mkdir(stubDir);
    % create stub load_experiment_config
    fid = fopen(fullfile(stubDir,'load_experiment_config.m'),'w');
    fprintf(fid, ['function cfg = load_experiment_config(~)\n' ...
        'cfg.experiment.plume_types = {\'plumeA\',\'plumeB\'};\n' ...
        'cfg.experiment.sensing_modes = {\'sense1\'};\n' ...
        'cfg.experiment.jobs_per_condition = 1;\n' ...
        'cfg.experiment.num_plumes = 2;\n' ...
        'cfg.experiment.num_sensing = 1;\n' ...
        'cfg.experiment.output_base = tempdir;\n' ...
        'cfg.get_output_dir = @(p,s,a,seed) fullfile(tempdir,[p ''_'' s],[sprintf(''%d_%d'',a,seed)]);\n' ...
        'cfg.plume_config = fullfile(''tests'',''sample_config.yaml'');\n' ...
        'end']);
    fclose(fid);
    % stub load_config
    fid = fopen(fullfile(stubDir,'load_config.m'),'w');
    fprintf(fid, ['function cfg = load_config(~)\n' ...
        'cfg.environment = ''gaussian'';\n' ...
        'cfg.triallength = 1;\n' ...
        'cfg.plotting = 0;\n' ...
        'cfg.ntrials = 1;\n' ...
        'end']);
    fclose(fid);
    % stub run_navigation_cfg
    fid = fopen(fullfile(stubDir,'run_navigation_cfg.m'),'w');
    fprintf(fid, ['function out = run_navigation_cfg(cfg)\n' ...
        'out = cfg;\n' ...
        'end']);
    fclose(fid);
    addpath(stubDir);
    testCase.TestData.stubDir = stubDir;
end

function teardownOnce(testCase)
    rmpath(testCase.TestData.stubDir);
    delete(fullfile(testCase.TestData.stubDir,'*.m'));
    rmdir(testCase.TestData.stubDir);
end

function testSeedDependsOnCondition(testCase)
    cfgFile = 'dummy.yaml';
    run_agent_simulation(1,1,cfgFile);
    d = dir(fullfile(tempdir,'plumeA_sense1','1_*'));
    seed1 = sscanf(d(1).name,'1_%d');
    run_agent_simulation(2,1,cfgFile);
    d2 = dir(fullfile(tempdir,'plumeB_sense1','1_*'));
    seed2 = sscanf(d2(1).name,'1_%d');
    verifyNotEqual(testCase, seed1, seed2);
end
