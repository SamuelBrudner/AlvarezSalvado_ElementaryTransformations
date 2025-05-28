function tests = test_process_smoke_video_missing_file
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    addpath(fullfile(pwd, 'Code'));
    addpath(fullfile(pwd, 'scripts'));

    tmpRoot = tempname;
    mkdir(tmpRoot);
    mkdir(fullfile(tmpRoot, 'configs'));

    % create required configs
    pathsFile = fullfile(tmpRoot, 'configs', 'paths.yaml');
    fid = fopen(pathsFile, 'w');
    fprintf(fid, 'data:\n  video: %s\n', fullfile(tmpRoot, 'missing.avi'));
    fprintf(fid, 'configs:\n  plume: %s\n', fullfile(tmpRoot, 'configs', 'plume.yaml'));
    fprintf(fid, 'output:\n  matlab_temp: %s\n', tmpRoot);
    fclose(fid);

    plumeFile = fullfile(tmpRoot, 'configs', 'plume.yaml');
    fid = fopen(plumeFile, 'w');
    fprintf(fid, 'px_per_mm: 1\nframe_rate: 1\n');
    fclose(fid);

    projPaths = fullfile(tmpRoot, 'configs', 'project_paths.yaml');
    fid = fopen(projPaths, 'w'); fclose(fid);

    copyfile('process_smoke_video.m', tmpRoot);

    testCase.TestData.tmpRoot = tmpRoot;
end

function teardownOnce(testCase)
    rmdir(testCase.TestData.tmpRoot, 's');
end

function testMissingVideoError(testCase)
    orig_script_dir = testCase.TestData.tmpRoot; %#ok<NASGU>
    f = @() run(fullfile(testCase.TestData.tmpRoot, 'process_smoke_video.m'));
    verifyError(testCase, f, 'process_smoke_video:VideoNotFound');
end
