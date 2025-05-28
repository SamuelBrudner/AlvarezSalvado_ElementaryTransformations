function tests = test_process_smoke_video_load_errors
    tests = functiontests(localfunctions);
end

function setup(testCase)
    addpath(fullfile(pwd, 'Code'));
    addpath(fullfile(pwd, 'scripts'));
    tmpRoot = tempname;
    mkdir(tmpRoot);
    mkdir(fullfile(tmpRoot, 'configs'));
    mkdir(fullfile(tmpRoot, 'Code'));
    mkdir(fullfile(tmpRoot, 'scripts'));
    copyfile('process_smoke_video.m', tmpRoot);
    testCase.TestData.tmpRoot = tmpRoot;
end

function teardown(testCase)
    rmdir(testCase.TestData.tmpRoot, 's');
end

function testPathsLoadFailure(testCase)
    tmpRoot = testCase.TestData.tmpRoot;
    fid = fopen(fullfile(tmpRoot, 'configs', 'project_paths.yaml'), 'w'); fclose(fid);
    fid = fopen(fullfile(tmpRoot, 'scripts', 'load_paths_config.m'), 'w');
    fprintf(fid, 'function cfg = load_paths_config\nerror(''stub:fail'',''boom'');\nend');
    fclose(fid);
    orig_script_dir = tmpRoot; %#ok<NASGU>
    f = @() run(fullfile(tmpRoot, 'process_smoke_video.m'));
    try
        f();
        verifyFail(testCase, 'Expected error');
    catch ME
        verifyEqual(testCase, ME.identifier, 'process_smoke_video:LoadPathsFailed');
        verifyEqual(testCase, ME.cause{1}.identifier, 'stub:fail');
    end
end

function testConfigLoadFailure(testCase)
    tmpRoot = testCase.TestData.tmpRoot;
    pathsFile = fullfile(tmpRoot, 'configs', 'project_paths.yaml');
    fid = fopen(pathsFile, 'w');
    fprintf(fid, 'data:\n  video: %s\n', fullfile(tmpRoot, 'dummy.avi'));
    fprintf(fid, 'configs:\n  plume: %s\n', fullfile(tmpRoot, 'configs', 'plume.yaml'));
    fprintf(fid, 'output:\n  matlab_temp: %s\n', tmpRoot);
    fclose(fid);
    fid = fopen(fullfile(tmpRoot, 'configs', 'plume.yaml'), 'w'); fclose(fid);
    fid = fopen(fullfile(tmpRoot, 'Code', 'load_config.m'), 'w');
    fprintf(fid, 'function cfg = load_config(~)\nerror(''stub:fail'',''boom'');\nend');
    fclose(fid);
    orig_script_dir = tmpRoot; %#ok<NASGU>
    f = @() run(fullfile(tmpRoot, 'process_smoke_video.m'));
    try
        f();
        verifyFail(testCase, 'Expected error');
    catch ME
        verifyEqual(testCase, ME.identifier, 'process_smoke_video:LoadConfigFailed');
        verifyEqual(testCase, ME.cause{1}.identifier, 'stub:fail');
    end
end

function testPlumeLoadFailure(testCase)
    tmpRoot = testCase.TestData.tmpRoot;
    pathsFile = fullfile(tmpRoot, 'configs', 'project_paths.yaml');
    fid = fopen(pathsFile, 'w');
    fprintf(fid, 'data:\n  video: %s\n', fullfile(tmpRoot, 'dummy.avi'));
    fprintf(fid, 'configs:\n  plume: %s\n', fullfile(tmpRoot, 'configs', 'plume.yaml'));
    fprintf(fid, 'output:\n  matlab_temp: %s\n', tmpRoot);
    fclose(fid);
    fid = fopen(fullfile(tmpRoot, 'configs', 'plume.yaml'), 'w');
    fprintf(fid, 'px_per_mm: 1\nframe_rate: 1\n');
    fclose(fid);
    fid = fopen(fullfile(tmpRoot, 'dummy.avi'), 'w'); fclose(fid);
    fid = fopen(fullfile(tmpRoot, 'Code', 'load_plume_video.m'), 'w');
    fprintf(fid, 'function plume = load_plume_video(varargin)\nerror(''stub:fail'',''boom'');\nend');
    fclose(fid);
    orig_script_dir = tmpRoot; %#ok<NASGU>
    f = @() run(fullfile(tmpRoot, 'process_smoke_video.m'));
    try
        f();
        verifyFail(testCase, 'Expected error');
    catch ME
        verifyEqual(testCase, ME.identifier, 'process_smoke_video:LoadPlumeFailed');
        verifyEqual(testCase, ME.cause{1}.identifier, 'stub:fail');
    end
end
