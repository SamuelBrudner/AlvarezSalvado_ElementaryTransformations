function tests = test_load_custom_plume
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    addpath(fullfile(pwd, 'Code'));
    tmpDir = fullfile(tempname);
    mkdir(tmpDir);
    vw = VideoWriter(fullfile(tmpDir, 'dummy.avi'));
    open(vw);
    writeVideo(vw, uint8(zeros(2,2,3)));
    close(vw);
    meta.output_directory = tmpDir;
    meta.output_filename = 'dummy.avi';
    meta.vid_mm_per_px = 0.5;
    meta.fps = 30;
    fid = fopen(fullfile(tmpDir, 'meta.json'), 'w');
    fwrite(fid, jsonencode(meta));
    fclose(fid);
    testCase.TestData.tmpDir = tmpDir;
    testCase.TestData.metaFile = fullfile(tmpDir, 'meta.json');
end

function teardownOnce(testCase)
    rmdir(testCase.TestData.tmpDir, 's');
end

function testLoadCustomPlumeStruct(testCase)
    plume = load_custom_plume(testCase.TestData.metaFile);
    verifyEqual(testCase, plume.frame_rate, 30);
    verifyEqual(testCase, plume.px_per_mm, 1/0.5);
    verifySize(testCase, plume.data, [2,2,1]);
end
