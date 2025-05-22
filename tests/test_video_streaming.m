function tests = test_video_streaming
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    addpath(fullfile(pwd,'Code'));
    tmpDir = tempname;
    mkdir(tmpDir);
    vw = VideoWriter(fullfile(tmpDir,'stream.avi'));
    vw.FrameRate = 1;
    open(vw);
    writeVideo(vw, uint8(255*cat(4,ones(1,1,1),zeros(1,1,1))));
    close(vw);
    testCase.TestData.video = fullfile(tmpDir,'stream.avi');
    testCase.TestData.tmpDir = tmpDir;
end

function teardownOnce(testCase)
    rmdir(testCase.TestData.tmpDir,'s');
end

function testStreamingEqualsLoaded(testCase)
    rng(42);
    plume = load_plume_video(testCase.TestData.video,1,1);
    out1 = navigation_model_vec(4,'video',0,1,plume);
    rng(42);
    vr = VideoReader(testCase.TestData.video);
    params.px_per_mm = 1;
    params.frame_rate = 1;
    out2 = navigation_model_vec_stream(4,'video',0,1,vr,params);
    verifyEqual(testCase,out1.x,out2.x,'AbsTol',1e-12);
    verifyEqual(testCase,out1.ON,out2.ON,'AbsTol',1e-12);
    verifyEqual(testCase,out1.OFF,out2.OFF,'AbsTol',1e-12);
    verifyEqual(testCase,out1.turn,out2.turn);
end
