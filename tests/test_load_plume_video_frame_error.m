function tests = test_load_plume_video_frame_error
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    addpath(fullfile(pwd, 'Code'));
    tmpDir = tempname;
    mkdir(tmpDir);
    vw = VideoWriter(fullfile(tmpDir, 'mismatch.avi'));
    open(vw);
    writeVideo(vw, uint8(zeros(2,2,1)));
    writeVideo(vw, uint8(zeros(3,3,1)));
    close(vw);
    testCase.TestData.video = fullfile(tmpDir, 'mismatch.avi');
    testCase.TestData.tmpDir = tmpDir;
end

function teardownOnce(testCase)
    rmdir(testCase.TestData.tmpDir, 's');
end

function testErrorThrown(testCase)
    f = @() load_plume_video(testCase.TestData.video, 1, 1);
    verifyError(testCase, f, 'load_plume_video:FrameSizeMismatch');
end
