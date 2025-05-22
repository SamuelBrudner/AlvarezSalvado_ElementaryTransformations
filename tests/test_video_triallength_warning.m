function tests = test_video_triallength_warning
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    addpath(fullfile(pwd, 'Code'));
    tmp = tempname;
    mkdir(tmp);
    vw = VideoWriter(fullfile(tmp, 'short.avi'));
    open(vw);
    writeVideo(vw, uint8(zeros(2,2,1)));
    close(vw);
    testCase.TestData.tmp = tmp;
    testCase.TestData.video = fullfile(tmp, 'short.avi');
end

function teardownOnce(testCase)
    rmdir(testCase.TestData.tmp, 's');
end

function testWarningEmitted(testCase)
    cfg.environment = 'video';
    cfg.plume_video = testCase.TestData.video;
    cfg.px_per_mm = 10;
    cfg.frame_rate = 50;
    cfg.plotting = 0;
    cfg.ntrials = 1;
    lastwarn('');
    run_navigation_cfg(cfg);
    [msg,~] = lastwarn;
    verifyEqual(testCase, msg, ...
        'Trial truncated to movie length; set cfg.loop=true to repeat.');
end
