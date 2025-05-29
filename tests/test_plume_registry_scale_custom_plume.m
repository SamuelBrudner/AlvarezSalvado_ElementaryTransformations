function tests = test_plume_registry_scale_custom_plume
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    addpath(fullfile(pwd, 'Code'));
    tmpDir = tempname;
    mkdir(tmpDir);
    vw = VideoWriter(fullfile(tmpDir, 'orig.avi'));
    open(vw);
    writeVideo(vw, uint8([0 255; 128 64]));
    writeVideo(vw, uint8([255 128; 64 0]));
    close(vw);
    meta = fullfile(tmpDir, 'meta.yaml');
    fid = fopen(meta, 'w');
    fprintf(fid, 'output_directory: %s\n', tmpDir);
    fprintf(fid, 'output_filename: orig.avi\n');
    fprintf(fid, 'vid_mm_per_px: 1\n');
    fprintf(fid, 'fps: 1\n');
    fclose(fid);
    testCase.TestData.tmpDir = tmpDir;
    testCase.TestData.meta = meta;
    testCase.TestData.outVideo = fullfile(tmpDir, 'scaled.avi');
    testCase.TestData.outMeta = fullfile(tmpDir, 'scaled.yaml');
end

function teardownOnce(testCase)
    rmdir(testCase.TestData.tmpDir, 's');
end

function testPlumeRegistryUpdated(testCase)
    scale_custom_plume(testCase.TestData.meta, ...
                       testCase.TestData.outVideo, ...
                       testCase.TestData.outMeta);
    registry = load_yaml(fullfile('configs', 'plume_registry.yaml'));
    verifyTrue(testCase, isfield(registry, 'scaled.avi'));
    entry = registry.("scaled.avi");
    stats = plume_intensity_stats();
    verifyEqual(testCase, entry.min, stats.CRIM.min, 'AbsTol', 1e-12);
    verifyEqual(testCase, entry.max, stats.CRIM.max, 'AbsTol', 1e-12);
    verifyTrue(testCase, isfield(registry, 'orig.avi'));
    plume = load_plume_video(fullfile(testCase.TestData.tmpDir, 'orig.avi'), 1, 1);
    % Original range should be stored without rescaling
    expMin = min(plume.data(:));
    expMax = max(plume.data(:));
    origEntry = registry.("orig.avi");
    verifyEqual(testCase, origEntry.min, expMin, 'AbsTol', 1e-12);
    verifyEqual(testCase, origEntry.max, expMax, 'AbsTol', 1e-12);
end
