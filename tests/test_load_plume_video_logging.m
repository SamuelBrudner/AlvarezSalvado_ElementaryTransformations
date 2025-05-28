function tests = test_load_plume_video_logging
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    addpath(fullfile(pwd, 'Code'));
    tmpDir = tempname;
    mkdir(tmpDir);
    vw = VideoWriter(fullfile(tmpDir, 'tiny.avi'));
    open(vw);
    writeVideo(vw, uint8(zeros(2,3,1)));
    writeVideo(vw, uint8(255*ones(2,3,1)));
    close(vw);
    testCase.TestData.video = fullfile(tmpDir, 'tiny.avi');
    testCase.TestData.tmpDir = tmpDir;
end

function teardownOnce(testCase)
    rmdir(testCase.TestData.tmpDir, 's');
end

function testLogsDimensions(testCase)
    out = evalc('load_plume_video(testCase.TestData.video, 1, 1)');
    hasDims = ~isempty(regexp(out, '2\s*x\s*3', 'once'));
    verifyTrue(testCase, hasDims, 'Output should include frame dimensions');
end
