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
    meta_path = fullfile(tmpDir, 'meta.yaml');
    fid = fopen(meta_path, 'w');
    fprintf(fid, 'output_directory: %s\n', tmpDir);
    fprintf(fid, 'output_filename: dummy.avi\n');
    fprintf(fid, 'vid_mm_per_px: 0.5\n');
    fprintf(fid, 'fps: 30\n');
    fclose(fid);
    testCase.TestData.tmpDir = tmpDir;
    testCase.TestData.metaFile = meta_path;
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
