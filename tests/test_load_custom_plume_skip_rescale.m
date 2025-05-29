function tests = test_load_custom_plume_skip_rescale
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    addpath(fullfile(pwd, 'Code'));
    tmpDir = tempname;
    mkdir(tmpDir);
    vw = VideoWriter(fullfile(tmpDir, 'plume.avi'));
    open(vw);
    writeVideo(vw, uint8([0 255; 128 64]));
    close(vw);
    meta = fullfile(tmpDir, 'meta.yaml');
    fid = fopen(meta, 'w');
    fprintf(fid, 'output_directory: %s\n', tmpDir);
    fprintf(fid, 'output_filename: plume.avi\n');
    fprintf(fid, 'vid_mm_per_px: 1\n');
    fprintf(fid, 'fps: 1\n');
    fprintf(fid, 'scaled_to_crim: true\n');
    fclose(fid);
    testCase.TestData.tmpDir = tmpDir;
    testCase.TestData.meta = meta;
    testCase.TestData.video = fullfile(tmpDir, 'plume.avi');
end

function teardownOnce(testCase)
    rmdir(testCase.TestData.tmpDir, 's');
end

function testSkipRescaling(testCase)
    orig = load_plume_video(testCase.TestData.video, 1, 1);
    plume = load_custom_plume(testCase.TestData.meta);
    verifyEqual(testCase, plume.data, orig.data);
end
