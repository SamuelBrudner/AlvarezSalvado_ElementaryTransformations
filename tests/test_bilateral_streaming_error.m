function tests = test_bilateral_streaming_error
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    addpath(fullfile(pwd,'Code'));
    tmpDir = tempname;
    mkdir(tmpDir);
    vw = VideoWriter(fullfile(tmpDir,'stream.avi'));
    open(vw);
    writeVideo(vw,uint8(zeros(2,2,1)));
    close(vw);
    testCase.TestData.video = fullfile(tmpDir,'stream.avi');
    testCase.TestData.tmpDir = tmpDir;
end

function teardownOnce(testCase)
    if exist(testCase.TestData.tmpDir,'dir')
        rmdir(testCase.TestData.tmpDir,'s');
    end
end

function testErrorIdentifier(testCase)
    cfg.plume_video = testCase.TestData.video;
    cfg.px_per_mm = 1;
    cfg.frame_rate = 1;
    cfg.plotting = 0;
    cfg.use_streaming = true;
    cfg.ntrials = 1;
    cfg.bilateral = true;
    f = @() run_navigation_cfg(cfg);
    verifyError(testCase,f,'run_navigation_cfg:BilateralStreamingUnsupported');
end
