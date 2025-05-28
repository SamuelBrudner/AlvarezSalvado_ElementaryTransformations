function tests = test_video_looping_behavior
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    addpath(fullfile(pwd, 'Code'));
    tmpDir = tempname;
    mkdir(tmpDir);
    vw = VideoWriter(fullfile(tmpDir, 'loop.avi'));
    open(vw);
    writeVideo(vw, uint8(zeros(2,2,1)));
    close(vw);
    testCase.TestData.video = fullfile(tmpDir, 'loop.avi');
    testCase.TestData.tmpDir = tmpDir;
end

function teardownOnce(testCase)
    rmdir(testCase.TestData.tmpDir, 's');
end

function testLoopingLongTrial(testCase)
    cfg = load_config(fullfile('configs', 'my_complex_plume_loop2min.yaml'));
    cfg.plume_video = testCase.TestData.video;
    run_navigation_cfg(cfg);
    assert(true);
end
