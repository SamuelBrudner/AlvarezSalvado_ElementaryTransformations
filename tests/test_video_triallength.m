function tests = test_video_triallength
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    addpath(fullfile(pwd, 'Code'));
    tmp = tempname;
    mkdir(tmp);
    vw = VideoWriter(fullfile(tmp, 'tl.avi'));
    open(vw);
    writeVideo(vw, uint8(zeros(2,2,2))); % two-frame video
    close(vw);
    testCase.TestData.video = fullfile(tmp, 'tl.avi');
    testCase.TestData.tmp = tmp;
end

function teardownOnce(testCase)
    rmdir(testCase.TestData.tmp, 's');
end

function testRespectsTriallength(testCase)
    cfg.environment = 'video';
    cfg.plume_video = testCase.TestData.video;
    cfg.px_per_mm = 10;
    cfg.frame_rate = 50;
    cfg.plotting = 0;
    cfg.ntrials = 1;
    cfg.triallength = 5;
    out = run_navigation_cfg(cfg);
    verifySize(testCase, out.x, [cfg.triallength, cfg.ntrials]);
end
