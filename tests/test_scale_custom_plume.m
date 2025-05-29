function tests = test_scale_custom_plume
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)

    addpath(fullfile(pwd, 'Code'));
    tmpDir = tempname;
    mkdir(tmpDir);
    vw = VideoWriter(fullfile(tmpDir, 'orig.avi'));
    open(vw);
    writeVideo(vw, uint8([0 255;128 64]));
    writeVideo(vw, uint8([255 128;64 0]));
    close(vw);
    meta = fullfile(tmpDir, 'meta.yaml');
    fid = fopen(meta, 'w');
    fprintf(fid, 'output_directory: %s\n', tmpDir);
    fprintf(fid, 'output_filename: orig.avi\n');
    fprintf(fid, 'vid_mm_per_px: 1\n');
    fprintf(fid, 'fps: 1\n');
    fprintf(fid, 'extra_field: 42\n');
    fclose(fid);
    testCase.TestData.tmpDir = tmpDir;
    testCase.TestData.meta = meta;
    testCase.TestData.outVideo = fullfile(tmpDir, 'scaled.avi');
    testCase.TestData.outMeta = fullfile(tmpDir, 'scaled.yaml');
end

function teardownOnce(testCase)
    rmdir(testCase.TestData.tmpDir, 's');
end

function testScalingAndMetadata(testCase)
    outMeta = scale_custom_plume(testCase.TestData.meta, ...
                                 testCase.TestData.outVideo, ...
                                 testCase.TestData.outMeta);
    verifyEqual(testCase, outMeta, testCase.TestData.outMeta);
    verifyEqual(testCase, exist(testCase.TestData.outVideo, 'file'), 2);
    info = load_config(testCase.TestData.outMeta);
    verifyEqual(testCase, info.output_directory, testCase.TestData.tmpDir);
    verifyEqual(testCase, info.output_filename, 'scaled.avi');
    verifyEqual(testCase, info.extra_field, 42);
    verifyTrue(testCase, info.scaled_to_crim);
    plume = load_custom_plume(testCase.TestData.outMeta);
    stats = plume_intensity_stats();
    verifyEqual(testCase, min(plume.data(:)), stats.CRIM.min, 'AbsTol', 1e-12);
    verifyEqual(testCase, max(plume.data(:)), stats.CRIM.max, 'AbsTol', 1e-12);

end
