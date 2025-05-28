function tests = test_load_paths_config_paths_yaml
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    addpath(fullfile(pwd, 'Code'));

    tmpRoot = fullfile(tempname);
    mkdir(tmpRoot);
    mkdir(fullfile(tmpRoot, 'configs'));
    mkdir(fullfile(tmpRoot, 'scripts'));

    copyfile(fullfile('scripts', 'load_paths_config.m'), ...
        fullfile(tmpRoot, 'scripts', 'load_paths_config.m'));

    template = fileread(fullfile('configs', 'project_paths.yaml.template'));
    template = strrep(template, '${PROJECT_DIR}', tmpRoot);
    cfgFile = fullfile(tmpRoot, 'configs', 'paths.yaml');
    fid = fopen(cfgFile, 'w');
    fwrite(fid, template);
    fclose(fid);

    testCase.TestData.tmpRoot = tmpRoot;
end

function teardownOnce(testCase)
    rmdir(testCase.TestData.tmpRoot, 's');
end

function testPathsExpandedCorrectly(testCase)
    addpath(fullfile(testCase.TestData.tmpRoot, 'scripts'));
    cfg = load_paths_config();

    expVideo = fullfile(testCase.TestData.tmpRoot, ...
        'data', 'smoke_1a_bgsub_raw.avi');
    verifyEqual(testCase, cfg.data.video, expVideo);

    expPlume = fullfile(testCase.TestData.tmpRoot, ...
        'configs', 'my_complex_plume_config.yaml');
    verifyEqual(testCase, cfg.configs.plume, expPlume);
end
