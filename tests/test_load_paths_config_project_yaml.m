function tests = test_load_paths_config_project_yaml
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    addpath(fullfile(pwd, 'scripts'));
    addpath(fullfile(pwd, 'Code'));
    cfgFile = fullfile('configs', 'project_paths.yaml');
    copyfile(fullfile('configs', 'project_paths.yaml.template'), cfgFile);
    testCase.TestData.cfgFile = cfgFile;
end

function teardownOnce(testCase)
    if exist(testCase.TestData.cfgFile, 'file')
        delete(testCase.TestData.cfgFile);
    end
end

function testLoadsProjectPathsYaml(testCase)
    cfg = load_paths_config();
    verifyTrue(testCase, isstruct(cfg));
    verifyTrue(testCase, isfield(cfg, 'scripts'));
end
