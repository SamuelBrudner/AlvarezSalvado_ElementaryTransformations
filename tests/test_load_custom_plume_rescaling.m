function tests = test_load_custom_plume_rescaling
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    addpath(fullfile(pwd,'Code'));
    tmpDir = tempname;
    mkdir(tmpDir);
    vw = VideoWriter(fullfile(tmpDir,'plume.avi'));
    open(vw);
    writeVideo(vw,uint8([0 255;128 64]));
    writeVideo(vw,uint8([255 128;64 0]));
    close(vw);
    meta = fullfile(tmpDir,'meta.yaml');
    fid = fopen(meta,'w');
    fprintf(fid,'output_directory: %s\n',tmpDir);
    fprintf(fid,'output_filename: plume.avi\n');
    fprintf(fid,'vid_mm_per_px: 1\n');
    fprintf(fid,'fps: 1\n');
    fclose(fid);
    testCase.TestData.tmpDir = tmpDir;
    testCase.TestData.meta = meta;
    testCase.TestData.video = fullfile(tmpDir,'plume.avi');
end

function teardownOnce(testCase)
    rmdir(testCase.TestData.tmpDir,'s');
end

function testRescalingToCrimaldiRange(testCase)
    stats = plume_intensity_stats();
    orig = load_plume_video(testCase.TestData.video,1,1);
    preMin = min(orig.data(:));
    preMax = max(orig.data(:));
    plume = load_custom_plume(testCase.TestData.meta);
    verifyEqual(testCase,min(plume.data(:)),stats.CRIM.min,'AbsTol',1e-12);
    verifyEqual(testCase,max(plume.data(:)),stats.CRIM.max,'AbsTol',1e-12);
    verifyTrue(testCase, abs(preMin - stats.CRIM.min) > 1e-6 || ...
        abs(preMax - stats.CRIM.max) > 1e-6);
end
